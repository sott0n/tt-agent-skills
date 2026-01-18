# Step 4: Metal Trace Optimization

## Contents
- When to Use Metal Trace
- APIs (begin_trace_capture, end_trace_capture, execute_trace)
- Basic Pattern: DRAM Input
- Advanced Pattern: L1 Input
- Constraints and Limitations
- Determining Trace Region Size
- Device vs End-to-End Performance
- TT-CNN Pipeline API
- Checklist

## Objective

Use Metal Trace to eliminate host overhead by capturing and replaying device operations.

## Overview

Metal Trace records operation dispatch commands into DRAM, enabling replay without host involvement. Benefits:
- Removes host overhead for operation dispatch
- Reduces gaps between operations on device
- Ideal for host-bound models with static shapes

## When to Use

- Model is host-bound (host slower than device)
- Input/output shapes are static
- Same operations run repeatedly

## APIs

### Device Configuration

```python
# Configure trace region size in pytest
@pytest.mark.parametrize("device_params", [{
    "l1_small_size": 24576,
    "trace_region_size": 800768  # Size for trace buffers
}], indirect=True)
```

### Trace Capture

```python
# Begin trace capture
tid = ttnn.begin_trace_capture(device, cq_id=0)

# ... operations to trace ...

# End trace capture
ttnn.end_trace_capture(device, tid, cq_id=0)
```

### Trace Execution

```python
# Execute captured trace
ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
```

### Static Tensor Allocation

```python
# Allocate persistent input tensor
input_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)

# Copy data to pre-allocated tensor
ttnn.copy_host_to_device_tensor(host_tensor, input_tensor, cq_id=0)
```

## Basic Pattern: DRAM Input

```python
# Allocate persistent DRAM input
input_dram = ttnn.allocate_tensor_on_device(
    spec, device, memory_config=sharded_dram_config
)

# First run to compile (warm up program cache)
ttnn.copy_host_to_device_tensor(host_data, input_dram)
input_l1 = ttnn.to_memory_config(input_dram, l1_config)
output = model(input_l1)

# Capture trace
ttnn.copy_host_to_device_tensor(host_data, input_dram)
tid = ttnn.begin_trace_capture(device, cq_id=0)
input_l1 = ttnn.to_memory_config(input_dram, l1_config)
output = model(input_l1)  # Keep output tensor reference
ttnn.end_trace_capture(device, tid, cq_id=0)

# Execute trace in loop
for batch in data_loader:
    ttnn.copy_host_to_device_tensor(batch, input_dram)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    result = output.cpu(blocking=False)

ttnn.synchronize_device(device)
```

## Advanced Pattern: L1 Input

For models that don't fit with persistent L1 input:

```python
# Compile run
input_l1 = host_tensor.to(device, l1_config)
output = model(input_l1)

# Capture trace with address matching
input_l1 = host_tensor.to(device, l1_config)
input_trace_addr = input_l1.buffer_address()
spec = input_l1.spec

# Deallocate output to free space for input allocation
output.deallocate(force=True)

tid = ttnn.begin_trace_capture(device, cq_id=0)
output = model(input_l1)

# Reallocate input at same address
input_l1 = ttnn.allocate_tensor_on_device(spec, device)
assert input_trace_addr == input_l1.buffer_address()  # Verify address match

ttnn.end_trace_capture(device, tid, cq_id=0)

# Execute
ttnn.copy_host_to_device_tensor(host_tensor, input_l1)
ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
```

## Constraints

1. **Static shapes**: All tensor shapes fixed during trace
2. **No I/O in trace**: Cannot capture read/write commands
3. **Program cache required**: Compile operations before capture
4. **Fixed addresses**: Input/output tensor addresses must match

## Determining Trace Region Size

If trace capture fails:

```
FATAL | Creating trace buffers of size 751616B on device 0,
       but only 0B is allocated for trace region.
```

Use the reported size to configure `trace_region_size`.

## Device vs End-to-End Performance

Understanding the difference is critical for optimization:

**Device Performance**: Only measures kernel execution time
- Best-case scenario
- Excludes host-device transfers

**End-to-End Performance**: Total time including all overheads
- Host dispatch latency
- Host-to-device transfers
- Device-to-host transfers
- Can be significantly lower than device performance

```
Without Trace:
  Python → C++ → Command Gen → Device Execution → Return

With Trace:
  First run: Python → C++ → Record Commands → Device Execution
  Later runs: Replay Commands → Device Execution (no Python/C++ overhead)
```

**When to use Trace:**
- Model is host-bound (device faster than host can dispatch)
- Large gap between device and e2e performance
- Static input/output shapes

## TT-CNN Pipeline API

For vision models, the Pipeline API simplifies traced execution:

```python
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

# Configure pipeline with tracing
config = PipelineConfig(
    use_trace=True,
    num_command_queues=1,  # or 2 for multi-CQ
)

pipeline = create_pipeline_from_config(
    config=config,
    model=my_model,
    device=device,
    dram_input_memory_config=dram_config,
    l1_input_memory_config=l1_config,
)

# Compile (captures trace)
pipeline.compile(sample_input)

# Execute (replays trace)
outputs = pipeline.enqueue(inputs).pop_all()

# Cleanup
pipeline.cleanup()
```

See `step-05-multi-cq.md` for multi-CQ pipeline configuration.

## Checklist

- [ ] Model shapes are static
- [ ] trace_region_size configured
- [ ] Program cache enabled (operations compiled)
- [ ] Persistent tensors allocated
- [ ] Address matching verified for L1 inputs
- [ ] Performance improvement measured
- [ ] Device vs e2e performance gap analyzed
- [ ] Consider Pipeline API for vision models
