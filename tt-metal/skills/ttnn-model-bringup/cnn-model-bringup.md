# CNN Model Bringup Guide

## Contents
- Prerequisites and Reference Implementation
- CNN-Specific Considerations (HWC format, sharding)
- CNN Operator Mapping (Conv2d, BatchNorm, Pooling)
- Systematic Module-by-Module Bring-Up
- Object Detection Models (YOLO-style)
- Testing CNN Models
- Performance Optimization
- Checklist

## Prerequisites

- Access to TT-Hardware ([Buy TT-Hardware](https://tenstorrent.com/hardware/wormhole))
- Knowledge of PyTorch and CNN architectures
- Familiarity with [TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html) and [TT-NN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html)
- Completed general model bringup steps 1-3 (Reference Model Analysis, Operator Mapping, Per-Operator Testing)

## Reference Implementation

For CNN models, use the following as reference:
- **YOLOv4**: `models/demos/yolov4/` - Object detection model
- **ResNet**: `models/demos/resnet/` - Image classification model
- **Model Demos**: `models/demos/` - Various CNN implementations

## CNN-Specific Considerations

### Data Format

**PyTorch vs TTNN Format:**
| Framework | Format | Shape |
|-----------|--------|-------|
| PyTorch | NCHW | (batch, channels, height, width) |
| TTNN | NHWC | (batch, height, width, channels) |

**Format Conversion:**
```python
# PyTorch NCHW to TTNN NHWC
def nchw_to_nhwc(tensor):
    return tensor.permute(0, 2, 3, 1).contiguous()

# TTNN NHWC to PyTorch NCHW
def nhwc_to_nchw(tensor):
    return tensor.permute(0, 3, 1, 2).contiguous()
```

### Tensor Shape Requirements

TTNN operations have specific shape requirements:
- Height and width should be padded to multiples of 32 for TILE_LAYOUT
- Channel dimension may need padding for optimal performance

```python
def pad_to_tile(tensor, tile_size=32):
    """Pad tensor dimensions to multiples of tile_size"""
    b, h, w, c = tensor.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    pad_c = (tile_size - c % tile_size) % tile_size

    if pad_h > 0 or pad_w > 0 or pad_c > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_c, 0, pad_w, 0, pad_h))

    return tensor
```

## CNN Operator Mapping

### Convolution Operations

| PyTorch | TTNN | Notes |
|---------|------|-------|
| `nn.Conv2d` | `ttnn.conv2d` | Input must be NHWC format |
| `nn.ConvTranspose2d` | `ttnn.conv_transpose2d` | For upsampling |
| `F.conv2d` | `ttnn.conv2d` | Functional API |

**Conv2d Example:**
```python
def ttnn_conv2d(
    input_tensor,
    weight,
    bias=None,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    device,
):
    """TTNN Conv2d wrapper"""

    # Input should already be in NHWC format
    batch_size, height, width, in_channels = input_tensor.shape

    # Create conv config
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="",  # or "relu" for fused conv+relu
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    # Run convolution
    output = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight,
        device=device,
        in_channels=in_channels,
        out_channels=weight.shape[0],
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        kernel_size=(weight.shape[2], weight.shape[3]),
        stride=(stride, stride) if isinstance(stride, int) else stride,
        padding=(padding, padding) if isinstance(padding, int) else padding,
        dilation=(dilation, dilation) if isinstance(dilation, int) else dilation,
        groups=groups,
        bias_tensor=bias,
        conv_config=conv_config,
    )

    return output
```

### Pooling Operations

| PyTorch | TTNN | Notes |
|---------|------|-------|
| `nn.MaxPool2d` | `ttnn.max_pool2d` | |
| `nn.AvgPool2d` | `ttnn.avg_pool2d` | |
| `nn.AdaptiveAvgPool2d` | `ttnn.adaptive_avg_pool2d` | Global pooling |

**MaxPool2d Example:**
```python
def ttnn_max_pool2d(input_tensor, kernel_size, stride=None, padding=0):
    """TTNN MaxPool2d"""
    if stride is None:
        stride = kernel_size

    return ttnn.max_pool2d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
```

### Normalization

| PyTorch | TTNN | Notes |
|---------|------|-------|
| `nn.BatchNorm2d` | `ttnn.batch_norm` | Per-channel normalization |
| `nn.GroupNorm` | `ttnn.group_norm` | |
| `nn.InstanceNorm2d` | Use group_norm with groups=channels | |

### Activation Functions

| PyTorch | TTNN |
|---------|------|
| `nn.ReLU` | `ttnn.relu` |
| `nn.LeakyReLU` | `ttnn.leaky_relu` |
| `nn.SiLU` / Swish | `ttnn.silu` |
| `nn.Mish` | `ttnn.mish` |
| `nn.Sigmoid` | `ttnn.sigmoid` |
| `nn.Hardswish` | `ttnn.hardswish` |

## Systematic Module-by-Module CNN Bring-Up

### Step 1: Basic Blocks

Start with the smallest building blocks:

**Convolution + BatchNorm + Activation (CBR Block):**
```python
def conv_bn_relu(
    input_tensor,
    *,
    parameters,
    device,
):
    """Conv + BatchNorm + ReLU block"""

    # Convolution
    output = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=parameters.conv.weight,
        bias_tensor=None,  # BN has bias
        device=device,
        # ... other conv params
    )

    # BatchNorm
    output = ttnn.batch_norm(
        output,
        running_mean=parameters.bn.running_mean,
        running_var=parameters.bn.running_var,
        weight=parameters.bn.weight,
        bias=parameters.bn.bias,
        eps=1e-5,
    )

    # ReLU
    output = ttnn.relu(output)

    return output
```

**Test Basic Block:**
```python
def test_conv_bn_relu(device):
    torch.manual_seed(0)

    # Create PyTorch reference
    torch_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
    torch_bn = nn.BatchNorm2d(128)

    # Input (NCHW for PyTorch)
    torch_input = torch.randn(1, 64, 56, 56)

    # Reference output
    with torch.no_grad():
        torch_output = F.relu(torch_bn(torch_conv(torch_input)))

    # Convert to NHWC for TTNN
    ttnn_input = nchw_to_nhwc(torch_input)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # TTNN output
    ttnn_output = conv_bn_relu(ttnn_input, parameters=parameters, device=device)

    # Convert back and compare
    output = nhwc_to_nchw(ttnn.to_torch(ttnn_output))
    assert_with_pcc(torch_output, output, 0.999)
```

### Step 2: Residual Blocks

Build up to more complex blocks:

**ResNet Residual Block:**
```python
def residual_block(
    input_tensor,
    *,
    parameters,
    device,
    downsample=False,
):
    """ResNet residual block"""

    identity = input_tensor

    # First conv
    out = conv_bn_relu(input_tensor, parameters=parameters.conv1, device=device)

    # Second conv (no ReLU before add)
    out = ttnn.conv2d(out, parameters.conv2.weight, device=device, ...)
    out = ttnn.batch_norm(out, ...)

    # Downsample identity if needed
    if downsample:
        identity = ttnn.conv2d(identity, parameters.downsample.weight, device=device, ...)
        identity = ttnn.batch_norm(identity, ...)

    # Residual connection
    out = ttnn.add(out, identity)
    out = ttnn.relu(out)

    return out
```

### Step 3: Feature Extraction Stages

Compose blocks into stages:

**Stage (Multiple Blocks):**
```python
def stage(input_tensor, *, parameters, device, num_blocks):
    """Stage containing multiple residual blocks"""
    output = input_tensor

    for i in range(num_blocks):
        block_params = getattr(parameters, f"block{i}")
        downsample = (i == 0)  # Downsample only in first block
        output = residual_block(output, parameters=block_params, device=device, downsample=downsample)

    return output
```

### Step 4: Full Model

Compose all stages:

**Complete CNN Model:**
```python
def resnet(input_tensor, *, parameters, device, config):
    """Full ResNet model"""

    # Stem
    x = ttnn.conv2d(input_tensor, parameters.stem.conv.weight, device=device, ...)
    x = ttnn.batch_norm(x, ...)
    x = ttnn.relu(x)
    x = ttnn.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # Stages
    x = stage(x, parameters=parameters.layer1, device=device, num_blocks=config.layers[0])
    x = stage(x, parameters=parameters.layer2, device=device, num_blocks=config.layers[1])
    x = stage(x, parameters=parameters.layer3, device=device, num_blocks=config.layers[2])
    x = stage(x, parameters=parameters.layer4, device=device, num_blocks=config.layers[3])

    # Global average pooling
    x = ttnn.adaptive_avg_pool2d(x, output_size=(1, 1))

    # Flatten and classify
    x = ttnn.reshape(x, (x.shape[0], -1))
    x = ttnn.linear(x, parameters.fc.weight, bias=parameters.fc.bias)

    return x
```

## Object Detection Models (YOLO-style)

For object detection models like YOLO:

### Backbone
```python
def yolo_backbone(input_tensor, *, parameters, device):
    """YOLO backbone (e.g., CSPDarknet)"""

    # Downsample blocks
    x = downsample1(input_tensor, parameters=parameters.ds1, device=device)
    x = downsample2(x, parameters=parameters.ds2, device=device)
    x = downsample3(x, parameters=parameters.ds3, device=device)

    # Feature maps at different scales
    p3 = x  # Small objects
    x = downsample4(x, parameters=parameters.ds4, device=device)
    p4 = x  # Medium objects
    x = downsample5(x, parameters=parameters.ds5, device=device)
    p5 = x  # Large objects

    return p3, p4, p5
```

### Neck (FPN/PANet)
```python
def yolo_neck(p3, p4, p5, *, parameters, device):
    """Feature Pyramid Network"""

    # Top-down path
    p5_up = ttnn.upsample(p5, scale_factor=2)
    p4 = ttnn.concat([p4, p5_up], dim=-1)  # NHWC, concat on channels
    p4 = csp_block(p4, parameters=parameters.fpn_p4, device=device)

    p4_up = ttnn.upsample(p4, scale_factor=2)
    p3 = ttnn.concat([p3, p4_up], dim=-1)
    p3 = csp_block(p3, parameters=parameters.fpn_p3, device=device)

    # Bottom-up path
    p3_down = ttnn.conv2d(p3, ..., stride=2)  # Downsample
    p4 = ttnn.concat([p4, p3_down], dim=-1)
    p4 = csp_block(p4, parameters=parameters.pan_p4, device=device)

    p4_down = ttnn.conv2d(p4, ..., stride=2)
    p5 = ttnn.concat([p5, p4_down], dim=-1)
    p5 = csp_block(p5, parameters=parameters.pan_p5, device=device)

    return p3, p4, p5
```

### Detection Head
```python
def yolo_head(p3, p4, p5, *, parameters, device, num_classes):
    """YOLO detection heads"""

    # Each head predicts: (x, y, w, h, objectness, class_probs)
    output_channels = (5 + num_classes) * 3  # 3 anchors per scale

    det_p3 = ttnn.conv2d(p3, parameters.head_p3.weight, device=device, ...)
    det_p4 = ttnn.conv2d(p4, parameters.head_p4.weight, device=device, ...)
    det_p5 = ttnn.conv2d(p5, parameters.head_p5.weight, device=device, ...)

    return det_p3, det_p4, det_p5
```

## Testing CNN Models

### Per-Module Testing
```python
def test_backbone(device):
    torch_model = load_torch_backbone()
    ttnn_model = create_ttnn_backbone(device)

    # Test input (NCHW)
    torch_input = torch.randn(1, 3, 416, 416)

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    # Convert to NHWC
    ttnn_input = nchw_to_nhwc(torch_input)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)

    ttnn_output = ttnn_model(ttnn_input)

    output = nhwc_to_nchw(ttnn.to_torch(ttnn_output))
    assert_with_pcc(torch_output, output, 0.99)
```

### End-to-End Testing
```python
def test_e2e_inference(device):
    """Test full model on real image"""
    from PIL import Image
    import torchvision.transforms as transforms

    # Load and preprocess image
    image = Image.open("test_image.jpg")
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Reference
    torch_model = load_torch_model()
    with torch.no_grad():
        torch_output = torch_model(input_tensor)

    # TTNN
    ttnn_input = nchw_to_nhwc(input_tensor)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)

    ttnn_output = ttnn_model(ttnn_input, device=device)

    # Compare detections
    compare_detections(torch_output, ttnn_output)
```

## Performance Optimization

See: [CNN Bring-up & Optimization in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/cnn_optimizations.md)

Key optimizations:
- **Operator Fusion**: Conv + BN + ReLU fusion
- **Memory Layout**: Optimal sharding strategies
- **Data Movement**: Minimize host-device transfers

## Reference Documentation

- [CNN Bring-up & Optimization](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/cnn_optimizations.md)
- [YOLOv4 Demo](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov4)
- [ResNet Demo](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/resnet)

## Checklist

- [ ] Data format conversion (NCHW ↔ NHWC) implemented
- [ ] Basic blocks (Conv+BN+ReLU) tested with PCC >= 0.999
- [ ] Residual/skip connections verified
- [ ] Pooling operations tested
- [ ] Feature pyramid (if applicable) working
- [ ] Full backbone tested with PCC >= 0.99
- [ ] Detection/classification head working
- [ ] End-to-end inference validated
- [ ] Real image inputs produce reasonable outputs
