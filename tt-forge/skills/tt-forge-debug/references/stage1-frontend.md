# Stage 1: Frontend Tracing (TorchDynamo/Torch XLA)

- **Modifiable**: No (external), but workarounds available
- **Input**: PyTorch FX Graph
- **Output**: StableHLO/VHLO MLIR

## Contents

- [What Happens in This Stage](#what-happens-in-this-stage)
- [Common Issues](#common-issues)
  - [Graph Breaks](#graph-breaks)
  - [Unsupported Operations](#unsupported-operations)
- [Workarounds](#workarounds)
- [Debugging](#debugging)

## What Happens in This Stage

1. `torch.compile(backend="tt")` triggers TorchDynamo
2. TorchDynamo traces the model into an FX Graph
3. `tt_backend()` in `tt_torch/backend/backend.py` processes the graph
4. `torch_pass_pipeline()` runs preprocessing passes
5. Torch XLA converts FX Graph to StableHLO/VHLO

## Common Issues

### Graph Breaks

**Symptom**: Model compiles but runs slower than expected, or errors about "graph break"

**Cause**: TorchDynamo cannot trace certain operations and splits the graph

**Debug**:
```python
import torch._dynamo
torch._dynamo.config.verbose = True

# Or use explain mode
explanation = torch._dynamo.explain(model)(input_x)
print(explanation)
```

**Common causes of graph breaks**:
- Data-dependent control flow (`if tensor.item() > 0`)
- Dynamic shapes
- Unsupported Python operations
- In-place mutations in certain contexts

### Unsupported Operations in Dynamo

**Symptom**: `unsupported` errors from TorchDynamo

**Workaround**: Use decompositions or rewrite the operation

```python
# Example: Replace problematic operation with equivalent
# Instead of:
x = torch.special.some_op(input)

# Use:
x = manual_implementation(input)
```

### Torch XLA Trace Failures

**Symptom**: Errors from `torch_xla` or `bridge.extract_compiled_graph()`

**Debug**:
```bash
export XLA_HLO_DEBUG=1
export TTXLA_LOGGER_LEVEL=DEBUG
```

## tt-xla Workarounds

### Decompositions

tt-xla provides custom decompositions in `tt_torch/backend/decompositions.py`:

```python
# Operations that are decomposed:
aten.copy.default       # Handles broadcast semantics
aten.matmul.default     # 4D+ tensor matmul via einsum
aten.dot.default        # 1D dot product to matmul
aten.squeeze.dims       # Squeeze to reshape
aten.avg_pool2d         # Special case for global pooling

# Standard decompositions enabled:
aten.native_layer_norm
aten.native_group_norm
aten.addmm
aten._adaptive_avg_pool2d
aten.grid_sampler_2d
aten._log_softmax
```

### Frontend Passes

`torch_pass_pipeline()` in `tt_torch/backend/backend.py` runs:

1. **run_fusion_passes()** - Detects and fuses multi-op patterns
2. **handle_composite_ops()** - Wraps patterns as composite ops
3. **torch.export.export()** - Exports to ExportedProgram
4. **run_decompositions()** - Applies decomposition table
5. **insert_argument_type_markers()** - Marks input types
6. **bypass_dtype_promotion_and_redundant_cast()** - Removes unnecessary casts
7. **bypass_redundant_getitem()** - Simplifies getitem operations
8. **bypass_assert_tensor_metadata()** - Removes assertions

### Disabling Problematic Features

```python
options = {
    # Disable composite ops if causing issues
    "tt_enable_composite_ops": False,

    # Disable fusion passes
    "tt_enable_torch_fx_fusion_pass": False,

    # Use legacy compile flow
    "tt_legacy_compile": True,

    # Use AOTAutograd (may help with some models)
    "tt_use_aot_autograd": True,
}
```

## Specific Error Patterns

### `as_strided_` Error

**Symptom**: Error in AdaptiveAveragePool with XLA tensors

**Location**: `tt_torch/backend/backend.py:301`

**Workaround**: Already handled by `rewrite_adaptive_avgpool_to_mean()`

### `batch_norm_training` Sharding Error

**Symptom**: LayerNorm/BatchNorm fails in multichip scenarios

**Location**: `tt_torch/backend/backend.py:310`

**Workaround**: Decomposition added for `_native_batch_norm_legit.no_stats`

### Metadata Injection Errors

**Symptom**: Errors during metadata propagation

**Debug**:
```bash
export XLA_HLO_DEBUG=0  # Disable metadata injection
```

## Adding Custom Decompositions

If an operation fails, you can add a custom decomposition:

```python
# In your code or as a PR to tt-xla
from torch._decomp import register_decomposition

@register_decomposition(torch.ops.aten.problematic_op)
def decompose_problematic_op(input, ...):
    # Implement using supported ops
    return supported_ops(input, ...)
```

## Environment Variables

```bash
# TorchDynamo debug
export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE=1

# Torch XLA debug
export XLA_HLO_DEBUG=1
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"

# tt-xla debug
export TTXLA_LOGGER_LEVEL=DEBUG
```

## Checking if Stage 1 Succeeded

If `export_path` is set, check for:
- `vhlo_*.mlir` - VHLO module was created
- `shlo_*.mlir` - StableHLO conversion succeeded

If neither exists, the failure is in Stage 1 (frontend tracing).

## Related Files

- `python_package/tt_torch/backend/backend.py` - Main backend entry point
- `python_package/tt_torch/backend/decompositions.py` - Custom decompositions
- `python_package/tt_torch/backend/passes.py` - FX graph passes
- `python_package/torch_plugin_tt/__init__.py` - TTPlugin class
