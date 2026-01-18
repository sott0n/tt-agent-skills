# Key Optimization Concepts

## Data Formats

| Format | Bits | Use Case |
|--------|------|----------|
| `float32` | 32 | High precision, debugging |
| `bfloat16` | 16 | Default for activations |
| `bfloat8_b` | 8 | Weights (block float, shared exponent) |
| `bfloat4_b` | 4 | Aggressive compression |

## Tensor Layouts

- **ROW_MAJOR_LAYOUT**: Standard row-major, flexible shapes
- **TILE_LAYOUT**: 32x32 tiles with 16x16 faces, required for compute

## Memory Hierarchy

| Storage | Capacity | Speed | Use Case |
|---------|----------|-------|----------|
| L1 | ~1MB/core | Fast | Activations, intermediates |
| DRAM | 12-32GB | Slower | Weights, large tensors |

## Sharding Strategies

| Type | Description | Best For |
|------|-------------|----------|
| Height | Split along height | Row-wise operations |
| Width | Split along width | Column-wise operations |
| Block | 2D grid split | Spatial locality |

## Math Fidelity

| Fidelity | TFLOPS | Use Case |
|----------|--------|----------|
| LoFi | 4.0 | High performance, low accuracy |
| HiFi2 | 2.0 | Balanced performance/accuracy |
| HiFi3 | 1.33 | Higher precision |
| HiFi4 | 1.0 | Highest accuracy operations |

Note: Accumulation precision (FP16/FP32) is controlled separately via `fp32_dest_acc_en` in the compute config.

## Conv2d Optimization

| Technique | Description |
|-----------|-------------|
| Double Buffering | Overlap memory access with compute |
| act_block_h_override | Tune activation block height for L1 |
| BN Folding | Fold BatchNorm into Conv2d weights |
| Sharding Strategy | Match sharding to spatial dimensions |

## Device vs End-to-End Performance

| Metric | Includes |
|--------|----------|
| Device Performance | Kernel execution only |
| End-to-End Performance | Host dispatch, transfers, kernels |

Use Metal Trace and Multi-CQ to close the gap between device and e2e performance.

## Multi-Device Scaling

| Strategy | Description |
|----------|-------------|
| Data Parallelism | Shard batch across devices |
| Weight Replication | Same weights on all devices |
