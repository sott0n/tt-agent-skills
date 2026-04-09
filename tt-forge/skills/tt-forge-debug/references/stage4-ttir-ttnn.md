# Stage 4: TTIR → TTNN (tt-mlir)

- **Modifiable**: Yes (via tt-mlir)
- **Input**: TTIR MLIR module
- **Output**: TTNN MLIR module

## Contents

- [What Happens in This Stage](#what-happens-in-this-stage)
- [Common Issues](#common-issues)
- [Compile Options Affecting This Stage](#compile-options-affecting-this-stage)
- [Math Fidelity Options](#math-fidelity-options)
- [Debugging TTIR/TTNN IR](#debugging-ttirttnn-ir)
- [Error Messages Reference](#error-messages-reference)

## What Happens in This Stage

1. `convertFromTTIRToTTNN()` - Main conversion entry point
2. `TTIRToTTNNBackendPipeline` - tt-mlir's TTIR to TTNN pipeline
   - Legalization passes
   - Layout conversion
   - Memory optimization
   - Operator lowering
   - Optional: Optimizer passes (optimization_level >= 1)

## Common Issues

### TTIR to TTNN Conversion Failures

**Symptom**: `Failed to convert from TTIR to TTNN module`

**Cause**: Unsupported TTIR operation or invalid configuration

**Debug**:
```bash
export TTXLA_LOGGER_LEVEL=VERBOSE
# Check ttir_*.mlir for the input TTIR
```

### Unsupported Operations

**Symptom**: Legalization errors for specific operations

**Common unsupported patterns**:
- Complex tensor reshapes
- Certain reduction patterns
- Specific convolution configurations

**Fix options**:
1. Add decomposition in tt-xla frontend
2. Add support in tt-mlir
3. Modify model to use supported operations

### Mesh Shape Errors

**Symptom**: `Invalid mesh shape size: N. Shape must have two dimensions!`

**Cause**: Mesh shape inference failed or invalid configuration

**Fix**: Ensure mesh shape is always 2D (e.g., `[1, 1]` for single device)

### Optimizer Pass Errors

**Symptom**: Errors when `optimization_level >= 1`

**Note**: `Optimizer passes are not supported in distributed runtime`

**Fix**: Use `optimization_level = 0` in distributed mode
```python
options = {"optimization_level": 0}
```

### Weight Dtype Errors

**Symptom**: `Unknown experimental_weight_dtype`

**Valid values**:
```python
options = {
    "experimental_weight_dtype": "bfp8",  # BFP_BFloat8
    # or
    "experimental_weight_dtype": "bfp4",  # BFP_BFloat4
}
```

## Compile Options Affecting This Stage

Most compile options affect this stage. Key options:

```python
options = {
    # Optimization level (most impactful)
    "optimization_level": 0,           # 0=none, 1=basic, 2=advanced

    # Compute configuration
    "math_fidelity": "hifi4",          # lofi, hifi2, hifi3, hifi4, ttnn_default
    "fp32_dest_acc_en": True,          # FP32 accumulation

    # Weight compression
    "experimental_weight_dtype": "bfp8",  # or "bfp4"

    # Constant evaluation
    "enable_const_eval": True,
    "enable_const_eval_on_cpu": True,  # CPU uses 32-bit precision

    # Tracing (requires TT_RUNTIME_TRACE_REGION_SIZE env var)
    "enable_trace": False,

    # Experimental
    "experimental_enable_fusing_conv2d_with_multiply_pattern": False,
    "experimental_enable_permute_matmul_fusion": True,
    "experimental_enable_dram_space_saving_optimization": False,
}
```

See [compile-options.md](compile-options.md) for complete options reference.

## Math Fidelity Options

| Value | Description |
|-------|-------------|
| `HiFi4` | Highest precision (default) |
| `HiFi3` | High precision |
| `HiFi2` | Medium precision |
| `LoFi` | Lowest precision, fastest |

## Key Source Files

- `pjrt_implementation/src/api/module_builder/module_builder.cc`
  - `convertFromTTIRToTTNN()`
- tt-mlir (external):
  - `ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h`
  - `ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h`

## tt-mlir Pipeline Options

```cpp
mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;
options.optimizationLevel = compile_options.optimization_level;
options.experimentalWeightDtype = ...;
options.computeCfgMathFidelity = ...;
options.computeCfgFp32DestAccEn = ...;
options.enableFusingConv2dWithMultiplyPattern = ...;
options.enablePermuteMatmulFusion = ...;
options.enableTrace = ...;
options.systemDescPath = system_descriptor_path;
options.enableConstEval = ...;
options.enableCPUHoistedConstEval = ...;
options.dramSpaceSavingOptimizationEnabled = ...;
options.meshShape = {mesh_shape[0], mesh_shape[1]};
options.meshTopology = ...; // From fabric config
options.devicePtr = ...; // For optimizer passes
```

## Debugging TTIR/TTNN IR

### Reading TTIR

```mlir
// Example TTIR operations
%0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
%1 = "ttir.relu"(%0) : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
```

### Reading TTNN

```mlir
// Example TTNN operations
%0 = "ttnn.matmul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : ...
%1 = "ttnn.relu"(%0) : ...
```

## Error Messages Reference

| Error Message | Likely Cause | Fix |
|--------------|--------------|-----|
| `Failed to convert from TTIR to TTNN` | Unsupported op or config | Check unsupported ops |
| `Invalid mesh shape size` | Bad mesh config | Ensure 2D mesh shape |
| `Optimizer passes not supported in distributed` | Wrong optimization_level | Set optimization_level=0 |
| `Unknown experimental_weight_dtype` | Invalid dtype string | Use "bfp8" or "bfp4" |
| `Failed to parse*MathFidelity` | Invalid math fidelity | Use valid fidelity value |

## Checking if Stage 4 Succeeded

Check `<export_path>/irs/` for:
- `ttir_*.mlir` - Input TTIR (from Stage 3)
- `ttnn_*.mlir` - Output TTNN (success)

If `ttnn_*.mlir` doesn't exist but `ttir_*.mlir` does, the failure is in Stage 4.

## Common Patterns That May Fail

1. **Dynamic shapes** - TTNN requires static shapes
2. **Unsupported layouts** - Some tensor layouts not supported
3. **Large tensors** - May exceed device memory
4. **Complex reshapes** - Multi-step reshapes may fail
5. **Unsupported dtypes** - Check supported data types
