# Compile Options Reference

## Contents

- [Quick Reference](#quick-reference)
- [Detailed Options](#detailed-options)
  - [Debugging & Export](#debugging--export)
  - [Optimization](#optimization)
  - [Compute Configuration](#compute-configuration)
  - [Weight Compression](#weight-compression)
  - [Performance](#performance)
  - [Backend Selection](#backend-selection)
  - [Experimental Options](#experimental-options)
  - [Codegen Options](#codegen-options)

## Quick Reference

```python
options = {
    # === Debugging & Export ===
    "export_path": "/path/to/dump",           # Dump intermediate IR files
    "export_model_name": "my_model",          # Name prefix for exported files
    "export_tensors": False,                  # Save graph inputs to disk

    # === Optimization ===
    "optimization_level": 0,                  # 0=none, 1=basic, 2=advanced
    "enable_const_eval": True,                # Constant folding on device
    "enable_const_eval_on_cpu": True,         # Hoist const-eval to CPU (32-bit precision)

    # === Compute Configuration ===
    "math_fidelity": "hifi4",                 # lofi, hifi2, hifi3, hifi4, ttnn_default
    "fp32_dest_acc_en": True,                 # FP32 accumulation

    # === Weight Compression ===
    "experimental_weight_dtype": "",          # "bfp8" or "bfp4"

    # === Performance ===
    "enable_trace": False,                    # Trace hoisting for repeated ops
    "ttnn_perf_metrics_enabled": False,       # Collect performance metrics
    "ttnn_perf_metrics_output_file": "",      # Output path for metrics

    # === Backend Selection ===
    "backend": "TTNNFlatbuffer",              # TTNNFlatbuffer, codegen_py, codegen_cpp

    # === Experimental ===
    "experimental_enable_fusing_conv2d_with_multiply_pattern": False,
    "experimental_enable_permute_matmul_fusion": True,
    "experimental_enable_dram_space_saving_optimization": False,

    # === Codegen-specific ===
    "codegen_try_recover_structure": False,   # Readable codegen output
    "codegen_split_files": False,             # Split codegen into files
}
```

## Detailed Options

### Debugging & Export

#### `export_path`
- **Type**: string (path)
- **Default**: None (disabled)
- **Purpose**: Directory to dump intermediate IR files for debugging
- **Outputs**: Creates `<export_path>/irs/` with MLIR files at each stage

```python
options = {"export_path": "./debug_output"}
# Creates: ./debug_output/irs/vhlo_*.mlir, shlo_*.mlir, ttir_*.mlir, ttnn_*.mlir
```

#### `export_model_name`
- **Type**: string
- **Default**: "" (auto-generated)
- **Purpose**: Prefix for exported file names
- **Format**: `<export_model_name>_g<graph_number>_<timestamp>.mlir`

#### `export_tensors`
- **Type**: bool
- **Default**: False
- **Purpose**: Save graph inputs to disk during execution (for chisel/codegen debugging)

---

### Optimization

#### `optimization_level`
- **Type**: int (0, 1, or 2)
- **Default**: 0
- **Affects**: Stage 4 (TTIR→TTNN)

| Level | Description | Enabled Features |
|-------|-------------|------------------|
| 0 | No optimizations | Baseline compilation |
| 1 | Basic optimizations | Optimizer passes + Conv2d fusion |
| 2 | Advanced optimizations | Level 1 + memory layout optimization |

**Important**: Optimizer passes are NOT supported in distributed runtime. Use `optimization_level=0` when `TT_RUNTIME_ENABLE_DISTRIBUTED=1`.

#### `enable_const_eval`
- **Type**: bool
- **Default**: True
- **Purpose**: Enable constant evaluation (folding) on device
- **Warning**: Results stored on device until closed. Multiple graphs with same weights can cause OOM.

#### `enable_const_eval_on_cpu`
- **Type**: bool
- **Default**: True
- **Purpose**: Hoist const-eval to CPU instead of device
- **Benefit**: CPU uses 32-bit precision, improving accuracy for some models

---

### Compute Configuration

#### `math_fidelity`
- **Type**: string
- **Default**: None (MLIR default = HiFi4)
- **Affects**: All TTNN operations with compute kernel config

| Value | Description |
|-------|-------------|
| `"lofi"` | Lowest precision, fastest |
| `"hifi2"` | Medium precision |
| `"hifi3"` | High precision |
| `"hifi4"` | Highest precision (default) |
| `"ttnn_default"` | Let TTNN choose per-operation |

#### `fp32_dest_acc_en`
- **Type**: bool
- **Default**: None (MLIR default = True)
- **Purpose**: Enable FP32 destination accumulation for higher precision

---

### Weight Compression

#### `experimental_weight_dtype`
- **Type**: string
- **Default**: "" (disabled)
- **Valid values**: `"bfp8"`, `"bfp4"`
- **Purpose**: Convert weights to block floating point format for matmul/linear ops

```python
# Enable BFP8 weight compression
options = {"experimental_weight_dtype": "bfp8"}
```

---

### Performance

#### `enable_trace`
- **Type**: bool
- **Default**: False
- **Purpose**: Enable trace hoisting for TTNN pipeline
- **Requirement**: All non-consteval ops must be on device
- **Benefit**: Eliminates host overhead for repeated operations

**Environment variable required**:
```bash
export TT_RUNTIME_TRACE_REGION_SIZE=10000000  # Adjust as needed
```

#### `ttnn_perf_metrics_enabled`
- **Type**: bool
- **Default**: False
- **Purpose**: Collect TTNN performance metrics during execution

#### `ttnn_perf_metrics_output_file`
- **Type**: string
- **Default**: "" (saves to `perf_metrics/` directory)
- **Purpose**: Custom output path for performance metrics

---

### Backend Selection

#### `backend`
- **Type**: string
- **Default**: `"TTNNFlatbuffer"`

| Value | Description |
|-------|-------------|
| `"TTNNFlatbuffer"` | Default runtime using flatbuffer binary |
| `"codegen_py"` | Generate TTNN Python code |
| `"codegen_cpp"` | Generate TTNN C++ code |

**Codegen requirements**:
- Set `TT_MLIR_HOME` environment variable
- tt-alchemist library must be available

---

### Experimental Options

#### `experimental_enable_fusing_conv2d_with_multiply_pattern`
- **Type**: bool
- **Default**: False
- **Purpose**: Fuse Conv2d with multiply pattern in TTNN
- **Note**: Temporary option until [tt-mlir#4628](https://github.com/tenstorrent/tt-mlir/issues/4628) is fixed

#### `experimental_enable_permute_matmul_fusion`
- **Type**: bool
- **Default**: True
- **Purpose**: Fuse transpose + matmul/linear operations
- **Trade-off**: Disabled = transpose can be constevaled (better perf), but may cause OOM

#### `experimental_enable_dram_space_saving_optimization`
- **Type**: bool
- **Default**: False
- **Purpose**: Enable DRAM space saving optimization pass (TTNNMemoryManagement)

---

### Codegen Options

#### `codegen_try_recover_structure`
- **Type**: bool
- **Default**: False
- **Purpose**: Generate more readable codegen output by recovering original graph structure
- **Use case**: When inspecting generated code

#### `codegen_split_files`
- **Type**: bool
- **Default**: False
- **Purpose**: Split codegen output into separate files

---

## Options by Stage

| Option | Affects Stage |
|--------|---------------|
| `export_path`, `export_model_name` | All stages |
| `optimization_level` | Stage 4 (TTIR→TTNN) |
| `math_fidelity`, `fp32_dest_acc_en` | Stage 4 (TTIR→TTNN) |
| `experimental_weight_dtype` | Stage 4 (TTIR→TTNN) |
| `enable_const_eval`, `enable_const_eval_on_cpu` | Stage 4 (TTIR→TTNN) |
| `enable_trace` | Stage 4 (TTIR→TTNN), Stage 6 (Runtime) |
| `backend` | Stage 5 (Flatbuffer/Codegen) |
| `ttnn_perf_metrics_*` | Stage 6 (Runtime) |

---

## Common Configurations

### Debugging a failing model
```python
options = {
    "export_path": "./debug_output",
    "export_model_name": "my_model",
    "optimization_level": 0,  # Simplest path
}
```

### Maximum performance
```python
options = {
    "optimization_level": 2,
    "enable_trace": True,
    "math_fidelity": "lofi",
    "experimental_weight_dtype": "bfp8",
}
```

### Maximum accuracy
```python
options = {
    "optimization_level": 0,
    "math_fidelity": "hifi4",
    "fp32_dest_acc_en": True,
    "enable_const_eval_on_cpu": True,
}
```

### Memory-constrained model
```python
options = {
    "enable_const_eval": False,  # Avoid storing consteval results
    "experimental_enable_dram_space_saving_optimization": True,
    "experimental_enable_permute_matmul_fusion": False,
}
```

---

## Source Files

- `pjrt_implementation/inc/api/compile_options.h` - CompileOptions struct
- `pjrt_implementation/src/api/compile_options.cc` - Option parsing
- `pjrt_implementation/src/api/compile_options_parser.cc` - XLA compile options parsing
