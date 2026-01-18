# Quick Reference

## Memory Configs

```python
# Interleaved (default)
ttnn.DRAM_MEMORY_CONFIG  # DRAM interleaved
ttnn.L1_MEMORY_CONFIG    # L1 interleaved

# Sharded
ttnn.create_sharded_memory_config(
    shape=(height, width),
    core_grid=ttnn.CoreGrid(y=4, x=4),
    strategy=ttnn.ShardStrategy.BLOCK
)
```

## Compute Config

```python
compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,  # LoFi(4T)/HiFi2(2T)/HiFi3(1.33T)/HiFi4(1T)
    math_approx_mode=True,   # Faster exp/gelu/sqrt with approximations
    fp32_dest_acc_en=False,  # True: FP32 accumulation (half tile capacity)
    packer_l1_acc=False,     # True: L1 accumulation for higher precision
)

# Apply to matmul
output = ttnn.matmul(a, b, compute_kernel_config=compute_config)
```

## Metal Trace

```python
# Allocate persistent input
input_tensor = ttnn.allocate_tensor_on_device(spec, device)

# Capture trace
tid = ttnn.begin_trace_capture(device, cq_id=0)
output = model(input_tensor)
ttnn.end_trace_capture(device, tid, cq_id=0)

# Execute trace
ttnn.copy_host_to_device_tensor(host_data, input_tensor)
ttnn.execute_trace(device, tid, cq_id=0)
```

## Multiple Command Queues

```python
# Configure device with 2 CQs
device = ttnn.open_device(device_id=0, num_command_queues=2)

# CQ0: compute, CQ1: I/O
write_event = ttnn.record_event(device, cq_id=1)
ttnn.wait_for_event(cq_id=0, event=write_event)
```

## TT-CNN Pipeline API (Vision Models)

```python
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

# Configure pipeline with trace and multi-CQ
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
    l1_input_memory_config=l1_config,
)

pipeline.compile(sample_input)
outputs = pipeline.enqueue(inputs).pop_all()
pipeline.cleanup()
```
