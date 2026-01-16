---
name: ttnn-model-optimization
description: "Guide for optimizing TTNN model performance on Tenstorrent hardware. Covers data formats, tensor layouts, sharding, memory management, Metal Trace, multi-CQ, Conv2d tuning, and multi-device optimization."
---

# TTNN Model Optimization

This skill provides guidance for optimizing model performance after functional bring-up on Tenstorrent hardware.

## When to Use This Skill

- After completing model bringup (see `ttnn-model-bringup` skill)
- When model is functionally correct but needs performance improvement
- When optimizing memory usage or data movement
- When implementing Metal Trace or multiple command queues
- When scaling to multiple devices

## Optimization Workflow

| Step | Name | Description | Document |
|------|------|-------------|----------|
| 1 | Data Formats | Choose optimal precision (bfloat16, bfloat8_b) | `step-01-data-formats.md` |
| 2 | Tensor Layouts | Configure tile layout and shapes | `step-02-tensor-layouts.md` |
| 3 | Memory & Sharding | Optimize L1/DRAM usage, double buffering | `step-03-memory-sharding.md` |
| 4 | Metal Trace | Capture and replay operations | `step-04-metal-trace.md` |
| 5 | Multi-CQ | Overlap I/O with compute | `step-05-multi-cq.md` |
| 6 | Conv2d Optimization | Sharding, block tuning, BN folding | `step-06-conv2d-optimization.md` |
| 7 | Multi-Device | Scale across multiple devices | `step-07-multi-device.md` |

## Key Optimization Concepts

### Data Formats

| Format | Bits | Use Case |
|--------|------|----------|
| `float32` | 32 | High precision, debugging |
| `bfloat16` | 16 | Default for activations |
| `bfloat8_b` | 8 | Weights (block float, shared exponent) |
| `bfloat4_b` | 4 | Aggressive compression |

### Tensor Layouts

- **ROW_MAJOR_LAYOUT**: Standard row-major, flexible shapes
- **TILE_LAYOUT**: 32x32 tiles with 16x16 faces, required for compute

### Memory Hierarchy

| Storage | Capacity | Speed | Use Case |
|---------|----------|-------|----------|
| L1 | ~1MB/core | Fast | Activations, intermediates |
| DRAM | 12-32GB | Slower | Weights, large tensors |

### Sharding Strategies

| Type | Description | Best For |
|------|-------------|----------|
| Height | Split along height | Row-wise operations |
| Width | Split along width | Column-wise operations |
| Block | 2D grid split | Spatial locality |

### Math Fidelity

| Fidelity | TFLOPS | Use Case |
|----------|--------|----------|
| LoFi | 4.0 | High performance, low accuracy |
| HiFi2 | 2.0 | Balanced performance/accuracy |
| HiFi3 | 1.33 | Higher precision |
| HiFi4 | 1.0 | Highest accuracy operations |

Note: Accumulation precision (FP16/FP32) is controlled separately via `fp32_dest_acc_en` in the compute config.

### Conv2d Optimization

| Technique | Description |
|-----------|-------------|
| Double Buffering | Overlap memory access with compute |
| act_block_h_override | Tune activation block height for L1 |
| BN Folding | Fold BatchNorm into Conv2d weights |
| Sharding Strategy | Match sharding to spatial dimensions |

### Device vs End-to-End Performance

| Metric | Includes |
|--------|----------|
| Device Performance | Kernel execution only |
| End-to-End Performance | Host dispatch, transfers, kernels |

Use Metal Trace and Multi-CQ to close the gap between device and e2e performance.

### Multi-Device Scaling

| Strategy | Description |
|----------|-------------|
| Data Parallelism | Shard batch across devices |
| Weight Replication | Same weights on all devices |

## Instructions for Claude

1. **Verify functional correctness** before optimizing
2. **Profile first** to identify bottlenecks
3. **Read step files** for detailed optimization techniques
4. **Test after each optimization** to verify PCC maintained

## Quick Reference

### Memory Configs

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

### Compute Config

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

### Metal Trace

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

### Multiple Command Queues

```python
# Configure device with 2 CQs
device = ttnn.open_device(device_id=0, num_command_queues=2)

# CQ0: compute, CQ1: I/O
write_event = ttnn.record_event(device, cq_id=1)
ttnn.wait_for_event(cq_id=0, event=write_event)
```

### TT-CNN Pipeline API (Vision Models)

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

## Files in This Skill

```
.claude/skills/ttnn-model-optimization/
├── SKILL.md                        # This file
├── step-01-data-formats.md         # Data format optimization
├── step-02-tensor-layouts.md       # Tensor layout optimization
├── step-03-memory-sharding.md      # Memory, sharding, double buffering
├── step-04-metal-trace.md          # Metal Trace guide
├── step-05-multi-cq.md             # Multiple command queues
├── step-06-conv2d-optimization.md  # Conv2d tuning (vision models)
└── step-07-multi-device.md         # Multi-device scaling
```

## Reference

### Tech Reports
- `tech_reports/AdvancedPerformanceOptimizationsForModels/` - Metal Trace & Multi-CQ
- `tech_reports/data_formats/data_formats.md` - Data format details
- `tech_reports/tensor_layouts/tensor_layouts.md` - Layout concepts
- `tech_reports/tensor_sharding/tensor_sharding.md` - Sharding strategies
- `tech_reports/memory/allocator.md` - Memory allocation
- `tech_reports/matrix_engine/matrix_engine.md` - Compute config

### TT-CNN Library (Vision Models)
- `models/tt_cnn/README.md` - Pipeline and Builder API
- `models/tt_cnn/TT_CNN_MODEL_BRINGUP_GUIDE.md` - Vision model optimization guide

### Documentation
- `docs/source/ttnn/ttnn/api.rst` - TTNN API reference
