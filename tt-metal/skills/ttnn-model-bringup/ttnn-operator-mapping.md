# PyTorch to TTNN Operator Mapping

## Contents
- Neural Network Layers (Linear, Conv2d, Embedding)
- Normalization (LayerNorm, BatchNorm, RMSNorm)
- Activation Functions (ReLU, GELU, Sigmoid, Softmax)
- Pooling (MaxPool, AvgPool, AdaptivePool)
- Matrix Operations (matmul, einsum, bmm)
- Pointwise Operations (binary, unary, comparison)
- Reduction Operations (sum, mean, max, min)
- Tensor Manipulation (reshape, transpose, concat)
- Transformer Operations (attention, MLP, KV cache)
- Loss Functions and Notes

## Neural Network Layers

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.nn.Conv1d` | `ttnn.conv1d` | |
| `torch.nn.Conv2d` | `ttnn.conv2d` | |
| `torch.nn.ConvTranspose2d` | `ttnn.conv_transpose2d` | |
| `torch.nn.Linear` | `ttnn.linear` | Fused matmul + bias |
| `torch.nn.Embedding` | `ttnn.embedding` | |

## Normalization

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.nn.BatchNorm2d` | `ttnn.batch_norm` | |
| `torch.nn.LayerNorm` | `ttnn.layer_norm` | |
| `torch.nn.GroupNorm` | `ttnn.group_norm` | |
| `torch.nn.RMSNorm` | `ttnn.rms_norm` | |

## Activation Functions

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.nn.GELU` / `torch.nn.functional.gelu` | `ttnn.gelu` | |
| `torch.nn.ReLU` / `torch.nn.functional.relu` | `ttnn.relu` | |
| `torch.nn.ReLU6` | `ttnn.relu6` | |
| `torch.nn.LeakyReLU` | `ttnn.leaky_relu` | |
| `torch.nn.PReLU` | `ttnn.prelu` | |
| `torch.nn.ELU` | `ttnn.elu` | |
| `torch.nn.SELU` | `ttnn.selu` | |
| `torch.nn.CELU` | `ttnn.celu` | |
| `torch.nn.Sigmoid` / `torch.sigmoid` | `ttnn.sigmoid` | |
| `torch.nn.Tanh` / `torch.tanh` | `ttnn.tanh` | |
| `torch.nn.Softmax` | `ttnn.softmax` | |
| `torch.nn.LogSoftmax` | `ttnn.log_sigmoid` | |
| `torch.nn.SiLU` / `torch.nn.functional.silu` | `ttnn.silu` | Also known as Swish |
| `torch.nn.Mish` | `ttnn.mish` | |
| `torch.nn.Softplus` | `ttnn.softplus` | |
| `torch.nn.Softsign` | `ttnn.softsign` | |
| `torch.nn.Hardsigmoid` | `ttnn.hardsigmoid` | |
| `torch.nn.Hardswish` | `ttnn.hardswish` | |
| `torch.nn.Hardtanh` | `ttnn.hardtanh` | |
| `torch.nn.GLU` | `ttnn.glu` | |
| `torch.nn.GEGLU` | `ttnn.geglu` | |

## Pooling

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.nn.MaxPool2d` | `ttnn.max_pool2d` | |
| `torch.nn.AvgPool2d` | `ttnn.avg_pool2d` | |
| `torch.nn.AdaptiveAvgPool2d` | `ttnn.adaptive_avg_pool2d` | |
| `torch.nn.AdaptiveMaxPool2d` | `ttnn.adaptive_max_pool2d` | |
| `torch.nn.functional.interpolate` | `ttnn.upsample` | |

## Matrix Operations

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.matmul` / `@` | `ttnn.matmul` | |
| `torch.mm` | `ttnn.matmul` | |
| `torch.bmm` | `ttnn.matmul` | Batched matmul |
| `torch.addmm` | `ttnn.addmm` | |
| `torch.outer` | `ttnn.outer` | |

## Pointwise Binary Operations

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.add` / `+` | `ttnn.add` | |
| `torch.sub` / `-` | `ttnn.subtract` | |
| `torch.mul` / `*` | `ttnn.multiply` | |
| `torch.div` / `/` | `ttnn.divide` | |
| `torch.pow` | `ttnn.pow` | |
| `torch.maximum` | `ttnn.maximum` | |
| `torch.minimum` | `ttnn.minimum` | |
| `torch.fmod` | `ttnn.fmod` | |
| `torch.remainder` | `ttnn.remainder` | |
| `torch.floor_divide` | `ttnn.floor_div` | |
| `torch.atan2` | `ttnn.atan2` | |
| `torch.hypot` | `ttnn.hypot` | |
| `torch.ldexp` | `ttnn.ldexp` | |
| `torch.xlogy` | `ttnn.xlogy` | |

## Pointwise Unary Operations

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.abs` | `ttnn.abs` | |
| `torch.neg` | `ttnn.neg` | |
| `torch.sign` | `ttnn.sign` | |
| `torch.sqrt` | `ttnn.sqrt` | |
| `torch.rsqrt` | `ttnn.rsqrt` | |
| `torch.square` | `ttnn.square` | |
| `torch.reciprocal` | `ttnn.reciprocal` | |
| `torch.exp` | `ttnn.exp` | |
| `torch.exp2` | `ttnn.exp2` | |
| `torch.expm1` | `ttnn.expm1` | |
| `torch.log` | `ttnn.log` | |
| `torch.log2` | `ttnn.log2` | |
| `torch.log10` | `ttnn.log10` | |
| `torch.log1p` | `ttnn.log1p` | |
| `torch.sin` | `ttnn.sin` | |
| `torch.cos` | `ttnn.cos` | |
| `torch.tan` | `ttnn.tan` | |
| `torch.asin` | `ttnn.asin` | |
| `torch.acos` | `ttnn.acos` | |
| `torch.atan` | `ttnn.atan` | |
| `torch.sinh` | `ttnn.sinh` | |
| `torch.cosh` | `ttnn.cosh` | |
| `torch.tanh` | `ttnn.tanh` | |
| `torch.asinh` | `ttnn.asinh` | |
| `torch.acosh` | `ttnn.acosh` | |
| `torch.atanh` | `ttnn.atanh` | |
| `torch.erf` | `ttnn.erf` | |
| `torch.erfc` | `ttnn.erfc` | |
| `torch.erfinv` | `ttnn.erfinv` | |
| `torch.floor` | `ttnn.floor` | |
| `torch.ceil` | `ttnn.ceil` | |
| `torch.round` | `ttnn.round` | |
| `torch.trunc` | `ttnn.trunc` | |
| `torch.frac` | `ttnn.frac` | |
| `torch.clamp` | `ttnn.clamp` | |
| `torch.clip` | `ttnn.clip` | |
| `torch.isnan` | `ttnn.isnan` | |
| `torch.isinf` | `ttnn.isinf` | |
| `torch.isfinite` | `ttnn.isfinite` | |
| `torch.logical_not` | `ttnn.logical_not` | |
| `torch.bitwise_not` | `ttnn.bitwise_not` | |
| `torch.lgamma` | `ttnn.lgamma` | |
| `torch.digamma` | `ttnn.digamma` | |
| `torch.i0` | `ttnn.i0` | |
| `torch.cbrt` | `ttnn.cbrt` | |

## Comparison Operations

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.eq` / `==` | `ttnn.eq` | |
| `torch.ne` / `!=` | `ttnn.ne` | |
| `torch.lt` / `<` | `ttnn.lt` | |
| `torch.le` / `<=` | `ttnn.le` | |
| `torch.gt` / `>` | `ttnn.gt` | |
| `torch.ge` / `>=` | `ttnn.ge` | |
| `torch.isclose` | `ttnn.isclose` | |

## Logical Operations

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.logical_and` | `ttnn.logical_and` | |
| `torch.logical_or` | `ttnn.logical_or` | |
| `torch.logical_xor` | `ttnn.logical_xor` | |
| `torch.bitwise_and` | `ttnn.bitwise_and` | |
| `torch.bitwise_or` | `ttnn.bitwise_or` | |
| `torch.bitwise_xor` | `ttnn.bitwise_xor` | |

## Reduction Operations

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.sum` | `ttnn.sum` | |
| `torch.mean` | `ttnn.mean` | |
| `torch.max` | `ttnn.max` | |
| `torch.min` | `ttnn.min` | |
| `torch.prod` | `ttnn.prod` | |
| `torch.std` | `ttnn.std` | |
| `torch.var` | `ttnn.var` | |
| `torch.argmax` | `ttnn.argmax` | |
| `torch.topk` | `ttnn.topk` | |
| `torch.cumsum` | `ttnn.cumsum` | |
| `torch.cumprod` | `ttnn.cumprod` | |

## Tensor Manipulation

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.cat` | `ttnn.concat` | |
| `torch.stack` | `ttnn.stack` | |
| `torch.split` | `ttnn.split` | |
| `torch.chunk` | `ttnn.chunk` | |
| `torch.reshape` | `ttnn.reshape` | |
| `torch.view` | `ttnn.view` | |
| `torch.permute` | `ttnn.permute` | |
| `torch.transpose` | `ttnn.transpose` | |
| `torch.squeeze` | `ttnn.squeeze` | |
| `torch.unsqueeze` | `ttnn.unsqueeze` | |
| `torch.expand` | `ttnn.expand` | |
| `torch.repeat` | `ttnn.repeat` | |
| `torch.repeat_interleave` | `ttnn.repeat_interleave` | |
| `torch.gather` | `ttnn.gather` | |
| `torch.scatter` | `ttnn.scatter` | |
| `torch.scatter_add` | `ttnn.scatter_add` | |
| `torch.index_fill` | `ttnn.index_fill` | |
| `torch.roll` | `ttnn.roll` | |
| `torch.flip` | Use `ttnn.permute` | |
| `torch.pad` | `ttnn.pad` | |
| `torch.slice` / indexing | `ttnn.slice` | |
| `torch.tril` | `ttnn.tril` | |
| `torch.triu` | `ttnn.triu` | |
| `torch.sort` | `ttnn.sort` | |
| `torch.nonzero` | `ttnn.nonzero` | |

## Ternary Operations

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.where` | `ttnn.where` | |
| `torch.lerp` | `ttnn.lerp` | |
| `torch.addcdiv` | `ttnn.addcdiv` | |
| `torch.addcmul` | `ttnn.addcmul` | |

## Transformer Operations

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.nn.functional.scaled_dot_product_attention` | `ttnn.transformer.scaled_dot_product_attention` | |
| Attention softmax | `ttnn.transformer.attention_softmax` | |
| Split QKV and heads | `ttnn.transformer.split_query_key_value_and_split_heads` | |
| Concatenate heads | `ttnn.transformer.concatenate_heads` | |

## Tensor Creation

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.zeros` | `ttnn.zeros` | |
| `torch.ones` | `ttnn.ones` | |
| `torch.full` | `ttnn.full` | |
| `torch.empty` | `ttnn.empty` | |
| `torch.arange` | `ttnn.arange` | |
| `torch.zeros_like` | `ttnn.zeros_like` | |
| `torch.ones_like` | `ttnn.ones_like` | |
| `torch.full_like` | `ttnn.full_like` | |
| `torch.empty_like` | `ttnn.empty_like` | |
| `torch.rand` | `ttnn.rand` | |

## Loss Functions

| PyTorch Operator | TTNN Operator | Notes |
|------------------|---------------|-------|
| `torch.nn.L1Loss` | `ttnn.l1_loss` | |
| `torch.nn.MSELoss` | `ttnn.mse_loss` | |

## Notes

- This mapping is not exhaustive. See below for how to find additional operators.
- Some operators may have slight API differences (e.g., parameter names, default values).
- If an operator is not available in TTNN, file a GitHub issue to request implementation.
- For complex operations, multiple TTNN ops may be needed to replicate PyTorch behavior.

## Finding Operators Not Listed Here

If the operator you need is not in this mapping table, search in the following locations:

### 1. Documentation

- **API Reference**: `docs/source/ttnn/ttnn/api.rst` - Complete list of TTNN operations
- **Online Docs**: https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html

### 2. Source Code

Search for TTNN operator implementations in the source code:

```bash
# Search for operator by name in TTNN Python bindings
grep -r "def <operator_name>" ttnn/ttnn/

# Search in C++ operation implementations
ls ttnn-metal/ttnn/cpp/ttnn/operations/

# Search for specific operation patterns
grep -r "ttnn\.<operator>" tests/ttnn/unit_tests/
```

**Key source directories:**
- `ttnn/ttnn/` - Python API definitions
- `ttnn-metal/ttnn/cpp/ttnn/operations/` - C++ operation implementations
- `tests/ttnn/unit_tests/operations/` - Unit tests (useful for usage examples)

### 3. Existing Model Implementations

Check how existing models use TTNN operators:

- `models/demos/` - Demo model implementations (YOLOv4, BERT, Llama, etc.)
- `models/common/` - Common model utilities

### 4. If Operator Does Not Exist

If the operator is not available in TTNN:
1. Check if a combination of existing ops can achieve the same result
2. Use PyTorch fallback temporarily: convert tensor to torch, run op, convert back
3. File a GitHub issue requesting the operator implementation
