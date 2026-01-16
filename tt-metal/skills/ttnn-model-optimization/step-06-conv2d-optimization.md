# Step 6: Conv2d Optimization

## Objective

Optimize convolution operations for vision models.

## Overview

Convolutions are core operations in vision models. Key optimization areas:
- Sharding strategy selection
- Activation block sizing
- Double buffering
- Weight preprocessing
- Data format selection

## Sharding Strategies for Conv2d

### Height Sharding (Recommended for most cases)

Best for tensors with large spatial dimensions (H×W):

```python
# Height sharding configuration
conv_config = ttnn.Conv2dConfig(
    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    act_block_h_override=15 * 32,  # Must be multiple of 32
)
```

### Width Sharding

Best for inputs with deep channels:

```python
# Width sharding for channel-heavy tensors
conv_config = ttnn.Conv2dConfig(
    shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
)
```

### Block Sharding

For 2D distribution across cores:

```python
conv_config = ttnn.Conv2dConfig(
    shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)
```

## Activation Block Tuning

Tune `act_block_h_override` for maximum performance:

```python
# Start small, increase until L1 runs out
act_block_h_values = [32, 64, 128, 256, 480, 640]

for act_block_h in act_block_h_values:
    try:
        conv_config = ttnn.Conv2dConfig(
            act_block_h_override=act_block_h,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        output = ttnn.conv2d(input, weight, config=conv_config)
        print(f"act_block_h={act_block_h}: OK")
    except RuntimeError as e:
        if "L1" in str(e):
            print(f"act_block_h={act_block_h}: Out of L1")
            break
```

**Guidelines:**
- Larger blocks = better performance (fewer kernel launches)
- Must fit in L1 with weights, intermediates, and double buffers
- Always multiple of 32

## Double Buffering

Enable for overlapping memory access with compute:

```python
conv_config = ttnn.Conv2dConfig(
    enable_act_double_buffer=True,    # Double buffer activations
    enable_weights_double_buffer=True, # Double buffer weights
)
```

**When to use:**
- Compute-bound convolutions
- When L1 has sufficient headroom (2x buffer size needed)

**When to disable:**
- Memory-constrained scenarios
- When L1 is already at capacity

## Batch Norm Folding

Fold BatchNorm into Conv2d for inference:

```python
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

# Fold BN into preceding Conv
conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
    conv_layer,      # nn.Conv2d
    bn_layer         # nn.BatchNorm2d
)

# Convert to TTNN tensors
weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat8_b)
bias = ttnn.from_torch(
    conv_bias.reshape(1, 1, 1, -1),  # Reshape for TTNN format
    dtype=ttnn.bfloat16
)
```

**Benefits:**
- Eliminates BN operation entirely
- Reduces memory bandwidth
- Fuses computation

## Data Formats for Conv2d

### Weights

```python
# Use bfloat8_b for weights (2x memory reduction)
weight = ttnn.from_torch(
    torch_weight,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT
)
```

### Activations

```python
# Use bfloat16 for activations (default)
# Use bfloat8_b for intermediate layers if accuracy allows

# First/last layers: higher precision
activation_dtype=ttnn.bfloat16

# Intermediate layers: can use lower precision
activation_dtype=ttnn.bfloat8_b
```

### Mixed Precision Strategy

```python
# First layer: full precision activations
encoder1_config = Conv2dConfig(
    activation_dtype=ttnn.bfloat16,
    weights_dtype=ttnn.bfloat8_b,
    output_dtype=ttnn.bfloat16,
)

# Intermediate layers: lower precision
encoder2_config = Conv2dConfig(
    activation_dtype=ttnn.bfloat8_b,
    weights_dtype=ttnn.bfloat8_b,
    output_dtype=ttnn.bfloat8_b,
)

# Final layer: higher precision output
final_config = Conv2dConfig(
    activation_dtype=ttnn.bfloat8_b,
    weights_dtype=ttnn.bfloat8_b,
    output_dtype=ttnn.bfloat16,
)
```

## Input Format Conversion

TTNN convolutions expect HWC format (Height-Width-Channel):

```python
# Convert from PyTorch NCHW to TTNN NHWC
input_hwc = ttnn.experimental.convert_to_hwc(input_tensor)

# Run convolution
output = conv(input_hwc)

# Convert back to NCHW for PyTorch compatibility
output_chw = ttnn.experimental.convert_to_chw(output, dtype=ttnn.bfloat16)
```

## Reshard Minimization

Plan sharding to minimize reshards through the network:

```python
# Good: All convolutions use same sharding strategy
layer1_config = Conv2dConfig(shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
layer2_config = Conv2dConfig(shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
layer3_config = Conv2dConfig(shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

# Bad: Frequent sharding changes (causes reshards)
layer1_config = Conv2dConfig(shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
layer2_config = Conv2dConfig(shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED)  # Reshard
layer3_config = Conv2dConfig(shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED)  # Reshard
```

## TT-CNN Builder API

For vision models, consider using the TT-CNN Builder API:

```python
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    HeightShardedStrategyConfiguration
)

# Create configuration from PyTorch layer
torch_conv = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
config = Conv2dConfiguration.from_torch(
    torch_layer=torch_conv,
    input_height=224,
    input_width=224,
    batch_size=1,
    sharding_strategy=HeightShardedStrategyConfiguration(
        act_block_h_override=64
    )
)

# Instantiate and execute
layer = TtConv2d(config, device)
output = layer(ttnn_input)
```

## Performance Profiling

Profile convolution performance:

```python
import time

# Warmup
for _ in range(3):
    output = conv(input)
ttnn.synchronize_device(device)

# Measure
start = time.perf_counter()
for _ in range(100):
    output = conv(input)
ttnn.synchronize_device(device)
end = time.perf_counter()

avg_time = (end - start) / 100
print(f"Average conv time: {avg_time*1000:.2f} ms")
```

## Checklist

- [ ] Sharding strategy selected (height/width/block)
- [ ] act_block_h_override tuned for L1 utilization
- [ ] Double buffering enabled where beneficial
- [ ] Batch normalization folded into convolutions
- [ ] Weight dtype set to bfloat8_b
- [ ] Activation dtype optimized per layer
- [ ] Reshard operations minimized
- [ ] Input/output format conversions handled
