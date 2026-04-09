# ttrt - Flatbuffer Debugging Tool

Essential for debugging Stage 5-6 issues.

- **Component**: tt-mlir
- **Required Build**: `-DTTMLIR_ENABLE_RUNTIME=ON` (for run/check/perf)

## Contents

- [Commands](#commands)
- [ttrt read](#ttrt-read) - Inspect flatbuffer (no hardware)
- [ttrt run](#ttrt-run) - Execute and verify
- [ttrt check](#ttrt-check) - Validate compatibility
- [ttrt query](#ttrt-query) - Get system descriptor
- [ttrt perf](#ttrt-perf) - Performance analysis
- [Common Workflows](#common-workflows)

## Commands

| Command | Purpose | Requires Silicon |
|---------|---------|------------------|
| `ttrt read` | Inspect flatbuffer contents | No |
| `ttrt run` | Execute flatbuffer | Yes |
| `ttrt check` | Validate against system | Yes |
| `ttrt query` | Get system descriptor | Yes |
| `ttrt perf` | Performance analysis | Yes |
| `ttrt emitpy` | Run generated Python code | Yes |

---

## ttrt read

Inspect sections of a flatbuffer binary. **Does not require hardware.**

```bash
# Available sections
ttrt read --section version out.ttnn
ttrt read --section system_desc out.ttnn
ttrt read --section mlir out.ttnn
ttrt read --section inputs out.ttnn
ttrt read --section outputs out.ttnn
ttrt read --section op_stats out.ttnn
ttrt read --section mesh_shape out.ttnn
ttrt read --section all out.ttnn

# Save extracted artifacts
ttrt read out.ttnn --save-artifacts
ttrt read out.ttnn --save-artifacts --artifact-dir /path/to/dir

# Read system descriptor file
ttrt read system_desc.ttsys

# Process directory of flatbuffers
ttrt read /dir/of/flatbuffers

# Output to JSON
ttrt read out.ttnn --result-file result.json
```

### When to Use

- Verify flatbuffer was generated correctly (Stage 5 debugging)
- Check input/output shapes and dtypes
- Inspect embedded MLIR
- Verify mesh shape configuration
- Compare system descriptor between compile and runtime

---

## ttrt run

Execute flatbuffer binary on silicon. Essential for Stage 6 debugging.

```bash
# Basic execution
ttrt run out.ttnn

# Multiple loops (for benchmarking)
ttrt run out.ttnn --loops 10

# Input initialization
ttrt run out.ttnn --seed 0           # Reproducible random
ttrt run out.ttnn --init arange       # Sequential values
ttrt run out.ttnn --identity          # Identity test

# Golden comparison (enabled by default)
ttrt run out.ttnn                     # Runs golden checks
ttrt run out.ttnn --disable-golden    # Skip golden checks
ttrt run out.ttnn --identity --rtol 1 --atol 1  # Custom tolerance

# Debugger mode (step through ops)
ttrt run out.ttnn --debugger

# Memory analysis
ttrt run out.ttnn --memory --save-artifacts
ttrt run out.ttnn --memory --check-memory-leak

# Save golden tensors for analysis
ttrt run out.ttnn --save-golden-tensors

# Print tensor values
ttrt run out.ttnn --print-input-output-tensors

# Run specific program in multi-program binary
ttrt run --program-index 0 out.ttnn
ttrt run --program-index all out.ttnn

# Run directory of flatbuffers
ttrt run /dir/of/flatbuffers
```

### Run Results

`run_results.json` is saved with execution details:

```json
{
  "file_path": "test.ttnn",
  "result": "pass",
  "exception": "",
  "program_results": {
    "program_index_0": {
      "loop_0": {
        "total_duration_ns": 3269341588
      }
    }
  }
}
```

### Golden Checks

Golden checks verify runtime accuracy. Results saved when `--save-artifacts` is used:

```json
{
  "loc(...)": {
    "expected_pcc": 0.99,
    "actual_pcc": 0.99123,
    "atol": 1e-08,
    "rtol": 1e-05,
    "allclose": true
  }
}
```

### Memory Analysis

With `--memory --save-artifacts`, detailed memory reports are saved:

```json
{
  "0": {
    "loc": "...",
    "dram": {
      "num_banks": 12,
      "total_bytes_per_bank": 1071181792,
      "total_bytes_allocated_per_bank": 16384,
      "total_bytes_free_per_bank": 1071167456
    },
    "l1": {
      "num_banks": 64,
      "total_bytes_per_bank": 1369120
    }
  }
}
```

### Debugger Mode

The `--debugger` flag enables pdb tracing after each op:

```bash
ttrt run out.ttnn --debugger
# Steps through execution, pausing after each operation
```

---

## ttrt check

Validate flatbuffer against a system descriptor.

```bash
# Check against current system
ttrt check out.ttnn

# Check against specific system descriptor
ttrt check out.ttnn --system-desc /path/to/system_desc.ttsys

# Check directory of flatbuffers
ttrt check /dir/of/flatbuffers --system-desc /dir/of/system_desc
```

### When to Use

- Verify flatbuffer is compatible with target hardware
- Debug "System desc does not match flatbuffer" errors
- Pre-validate before attempting execution

---

## ttrt query

Query the current system and save system descriptor.

```bash
# Query and display system info
ttrt query

# Save system descriptor
ttrt query --save-artifacts
# Creates: ttrt-artifacts/system_desc.ttsys

# Custom output directory
ttrt query --save-artifacts --artifact-dir /path/to/dir
```

### When to Use

- Generate system descriptor for compile-only mode
- Verify hardware configuration
- Compare system descriptors between machines

---

## ttrt perf

Run performance analysis. Requires perf-enabled build (`-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`).

```bash
# Run performance analysis
ttrt perf out.ttnn

# Host-only performance (no device-side data)
ttrt perf out.ttnn --host-only

# Save artifacts
ttrt perf out.ttnn --save-artifacts
```

### Output Files

When `--save-artifacts` is used:

| File | Description |
|------|-------------|
| `ops_perf_results.csv` | Compiled op performance results |
| `profile_log_device.csv` | Device-side profiled results |
| `tracy_ops_data.csv` | Op data in readable format |
| `tracy_ops_times.csv` | Op timing results |
| `tracy_profile_log_host.tracy` | Tracy GUI file |

### CSV Fields

Performance CSV includes:
- OP CODE, OP TYPE, GLOBAL CALL COUNT
- HOST START TS, HOST END TS, HOST DURATION [ns]
- DEVICE FW START CYCLE, DEVICE FW END CYCLE
- DEVICE KERNEL DURATION [ns]
- INPUT/OUTPUT shapes, layouts, datatypes, memory
- MATH FIDELITY, CORE COUNT

---

## Environment Variables

```bash
# ttrt logging levels
export TTRT_LOGGER_LEVEL=INFO      # Default
export TTRT_LOGGER_LEVEL=DEBUG     # Verbose
export TTRT_LOGGER_LEVEL=WARNING
export TTRT_LOGGER_LEVEL=ERROR

# tt-metal logging (for runtime debugging)
export TT_METAL_LOGGER_LEVEL=DEBUG
```

---

## Common Workflows

### Debug Stage 5 (Flatbuffer Generation)

```bash
# 1. Check if flatbuffer was generated
ls <export_path>/irs/ttnn_*.mlir  # TTNN IR exists?

# 2. Inspect flatbuffer structure
ttrt read out.ttnn --section all --save-artifacts

# 3. Verify input/output specs
ttrt read out.ttnn --section inputs
ttrt read out.ttnn --section outputs
```

### Debug Stage 6 (Runtime Execution)

```bash
# 1. Validate flatbuffer against system
ttrt check out.ttnn

# 2. Run with golden checks
ttrt run out.ttnn --save-artifacts

# 3. If fails, run with debugger
ttrt run out.ttnn --debugger

# 4. Check memory usage
ttrt run out.ttnn --memory --save-artifacts --check-memory-leak
```

### Performance Debugging

```bash
# 1. Run performance analysis
ttrt perf out.ttnn --save-artifacts

# 2. Analyze ops_perf_results.csv
# Look for:
# - High HOST DURATION vs DEVICE KERNEL DURATION (host bottleneck)
# - High OP TO OP LATENCY (dispatch overhead)
# - Low FPU UTIL (underutilized compute)
```

### Verify Cross-System Compatibility

```bash
# On compile machine
ttrt query --save-artifacts
# Transfer system_desc.ttsys to compile-only machine

# On target machine
ttrt check out.ttnn --system-desc /path/to/system_desc.ttsys
```

---

## Versioning

ttrt has strict version checking. Flatbuffer version must match ttrt version.

```bash
# Version format: vmajor.minor.patch
# Patch = commits since last tag

# Bypass version check (use with caution)
ttrt run out.ttnn --ignore-version
```

---

## GDB Integration

Debug C++ runtime components:

```bash
ttrt --gdb run out.ttnn
ttrt --gdb perf out.ttnn
```

---

## Python API

```python
from ttrt.common.api import API

API.initialize_apis()

# Configure and run
custom_args = {
    "--save-artifacts": True,
    "--loops": 10,
    "binary": "/path/to/out.ttnn"
}

run_instance = API.Run(args=custom_args)
result_code, results = run_instance()
```

---

## Error Messages Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `Flatbuffer version does not match ttrt version` | Version mismatch | Rebuild with same version |
| `System desc does not match flatbuffer` | Compiled for different system | Recompile for target system |
| Golden check fails (low PCC) | Accuracy issue | Check math fidelity, dtypes |
| Memory leak detected | L1/DRAM not freed | Report to tt-mlir team |
