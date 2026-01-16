# Step 5: Multiple Command Queues

## Objective

Use multiple command queues to overlap I/O with compute operations.

## Overview

Metal supports 2 command queues that execute independently:
- Overlap input writes with operation execution
- Overlap output reads with next iteration
- Requires event synchronization between queues

## When to Use

- Model is device-bound with slow I/O
- Large input tensors take significant write time
- Want to hide I/O latency

## APIs

### Device Configuration

```python
# Configure 2 command queues
@pytest.mark.parametrize("device_params", [{
    "num_command_queues": 2
}], indirect=True)

# Or programmatically
device = ttnn.open_device(device_id=0, num_command_queues=2)
```

### Event Synchronization

```python
# Record event on command queue
event = ttnn.record_event(device, cq_id=0)

# Wait for event on different queue
ttnn.wait_for_event(cq_id=1, event=event)
```

## Pattern 1: Writes on CQ1, Compute+Reads on CQ0

```python
# Allocate persistent input
input_dram = ttnn.allocate_tensor_on_device(spec, device, dram_config)

# Dummy event for first iteration
op_event = ttnn.record_event(device, 0)

outputs = []
for batch in batches:
    # CQ1: Wait for compute to finish reading input
    ttnn.wait_for_event(1, op_event)

    # CQ1: Write next input
    ttnn.copy_host_to_device_tensor(batch, input_dram, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    # CQ0: Wait for write to complete
    ttnn.wait_for_event(0, write_event)

    # CQ0: Run first op (consumes input)
    input_l1 = ttnn.to_memory_config(input_dram, l1_config)
    op_event = ttnn.record_event(device, 0)  # Signal input consumed

    # CQ0: Run rest of model + read output
    output = model(input_l1)
    outputs.append(output.cpu(blocking=False))

ttnn.synchronize_device(device)
```

## Pattern 2: Compute on CQ0, I/O on CQ1

```python
# Allocate persistent input and output
input_dram = ttnn.allocate_tensor_on_device(input_spec, device, dram_config)
output_dram = ttnn.allocate_tensor_on_device(output_spec, device, dram_config)

# Dummy events
first_op_event = ttnn.record_event(device, 0)
read_event = ttnn.record_event(device, 1)

# Initial write (outside loop)
ttnn.wait_for_event(1, first_op_event)
ttnn.copy_host_to_device_tensor(host_input, input_dram, cq_id=1)
write_event = ttnn.record_event(device, 1)

outputs = []
for i in range(iterations):
    # CQ0: Wait for input write
    ttnn.wait_for_event(0, write_event)

    # CQ0: First op
    input_l1 = ttnn.to_memory_config(input_dram, l1_config)
    first_op_event = ttnn.record_event(device, 0)

    # CQ0: Rest of model
    output = model(input_l1)

    # CQ0: Wait for previous read to complete
    ttnn.wait_for_event(0, read_event)

    # CQ0: Write output to DRAM
    output_dram = ttnn.reshard(output, output_dram_config, output_dram)
    last_op_event = ttnn.record_event(device, 0)

    # CQ1: Write next input (overlaps with compute)
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(next_input, input_dram, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    # CQ1: Read output (overlaps with next iteration)
    ttnn.wait_for_event(1, last_op_event)
    outputs.append(output_dram.cpu(blocking=False, cq_id=1))
    read_event = ttnn.record_event(device, 1)

ttnn.synchronize_device(device)
```

## Combining with Metal Trace

```python
# Capture trace (doesn't include events)
tid = ttnn.begin_trace_capture(device, cq_id=0)
output = model(input_l1)
ttnn.end_trace_capture(device, tid, cq_id=0)

# Execute with multi-CQ
for batch in batches:
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(batch, input_dram, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    ttnn.wait_for_event(0, write_event)
    input_l1 = ttnn.reshard(input_dram, l1_config, input_l1)  # In-place
    op_event = ttnn.record_event(device, 0)

    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    outputs.append(output.cpu(blocking=False))
```

## Event Flow Diagram

```
CQ0 (Compute)     CQ1 (I/O)
     │                │
     │    ┌───────────┤ Write input
     │    │           │
     ▼    │           ▼
  Wait ◄──┘      write_event
     │
     ▼
  Compute
     │
     ▼
  op_event ───────────►  Wait
                          │
                          ▼
                      Read output
```

## TT-CNN Pipeline API

For vision models, use the Pipeline API for simplified multi-CQ:

```python
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

# Standard 2CQ: Input on CQ1, compute+output on CQ0
config = PipelineConfig(
    use_trace=True,
    num_command_queues=2,
    all_transfers_on_separate_command_queue=False,
)

# Full I/O separation: All transfers on CQ1, only compute on CQ0
config = PipelineConfig(
    use_trace=True,
    num_command_queues=2,
    all_transfers_on_separate_command_queue=True,
)

pipeline = create_pipeline_from_config(
    config=config,
    model=my_model,
    device=device,
    dram_input_memory_config=dram_config,
    dram_output_memory_config=dram_output_config,  # Required for all_transfers_on_separate_command_queue
    l1_input_memory_config=l1_config,
)

pipeline.compile(sample_input)
outputs = pipeline.enqueue(inputs).pop_all()
pipeline.cleanup()
```

### Choosing Pipeline Configuration

| Configuration | Use When |
|---------------|----------|
| `num_command_queues=1` | Simple models, debugging |
| `num_command_queues=2, all_transfers=False` | Significant input transfer time |
| `num_command_queues=2, all_transfers=True` | Significant bidirectional I/O |

### Host Memory Pre-allocation

For workloads bottlenecked by host memory allocation:

```python
# Pre-allocate host memory for outputs
pipeline.preallocate_output_tensors_on_host(num_iterations)
```

## Performance Analysis

### Identifying I/O Bottlenecks

```
Symptoms of I/O bottleneck:
- Large gap between device and e2e performance
- Device utilization < 100%
- Vision models with large input tensors

Solutions:
- Use 2CQ to overlap transfers
- Reduce input/output padding
- Use DRAM sharding for persistent tensors
```

### Measuring Overlap Effectiveness

```python
import time

# Without multi-CQ
device = ttnn.open_device(device_id=0, num_command_queues=1)
# ... run model ...
single_cq_time = measure_e2e_time()

# With multi-CQ
device = ttnn.open_device(device_id=0, num_command_queues=2)
# ... run model with event synchronization ...
multi_cq_time = measure_e2e_time()

print(f"Speedup: {single_cq_time / multi_cq_time:.2f}x")
```

## Checklist

- [ ] Device configured with 2 command queues
- [ ] Events properly synchronize CQs
- [ ] Persistent tensors used for I/O
- [ ] No deadlocks in event flow
- [ ] Performance improvement verified
- [ ] Consider Pipeline API for vision models
- [ ] all_transfers_on_separate_command_queue evaluated
- [ ] Host memory pre-allocation considered
