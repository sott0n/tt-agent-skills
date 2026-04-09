# Stage 6: Runtime Execution (tt-metal)

- **Modifiable**: Yes
- **Input**: Flatbuffer binary, input tensors
- **Output**: Computed output tensors

## Contents

- [What Happens in This Stage](#what-happens-in-this-stage)
- [Common Issues](#common-issues)
- [Environment Variables](#environment-variables)
- [Execution Flow](#execution-flow)
- [Debugging with ttrt](#debugging-with-ttrt)
- [Debugging Runtime Crashes](#debugging-runtime-crashes)
- [Error Messages Reference](#error-messages-reference)

## What Happens in This Stage

1. `LoadedExecutableInstance::execute()` - Entry point for execution
2. Device/argument validation
3. `getOrCreateMeshDevice()` - Set up mesh device
4. `getInputRuntimeTensors()` - Prepare input tensors
5. `tt::runtime::submit()` - Submit to tt-metal runtime
6. `fillPJRTOutputLists()` - Extract and return outputs

## Common Issues

### Device Count Mismatch

**Symptom**: `Device count mismatch: N vs M`

**Cause**: Number of devices at execution differs from compilation

**Fix**: Ensure consistent device configuration between compile and run

### Argument Count Mismatch

**Symptom**: `Argument count mismatch: N vs M`

**Cause**: Number of inputs at execution differs from compilation

**Fix**: Ensure inputs match the compiled model's expected inputs

### Output Count Mismatch

**Symptom**: `Runtime produced different number of output tensors (N) than the compiler estimated number of outputs (M)`

**Cause**: tt-metal runtime produced unexpected number of outputs

**Debug**: This usually indicates a runtime bug; report to tt-metal team

### Mesh Device Errors

**Symptom**: Errors during `getOrCreateMeshDevice()`

**Cause**: Hardware configuration mismatch or device not available

**Debug**:
```bash
tt-smi  # Check device status
```

### Runtime Submit Failures

**Symptom**: `tt::runtime::submit()` returns error or crashes

**Cause**: Various runtime issues including:
- Unsupported operation at runtime
- Memory allocation failure
- Hardware fault
- Kernel execution error

**Debug**:
```bash
# Enable tt-metal debug logging
export TT_METAL_LOGGER_LEVEL=DEBUG

# Enable runtime tracing
export TTMLIR_ENABLE_PERF_TRACE=1
```

### Compile-Only Mode

**Symptom**: `Early aborting execution in compile-only mode`

**Cause**: Trying to execute when `TT_COMPILE_ONLY_SYSTEM_DESC` is set

**Fix**: Unset the environment variable for actual execution

### Input Tensor Preparation Errors

**Symptom**: `Failed to fill strategy map from sharding`

**Cause**: Input buffer sharding doesn't match expected sharding

**Debug**: Check input buffer device placement and sharding

### Layout Conversion Errors

**Symptom**: Errors during `ensure_layout()`

**Cause**: Input tensor layout doesn't match expected runtime layout

**Debug**: Check input tensor memory layout

## Environment Variables

```bash
# tt-metal debugging
export TT_METAL_LOGGER_LEVEL=DEBUG

# Performance tracing
export TTMLIR_ENABLE_PERF_TRACE=1

# Memory tracking (pytest)
pytest --log-memory path/to/test

# Compile-only mode (disables execution)
export TT_COMPILE_ONLY_SYSTEM_DESC=/path/to/system.ttsys
```

## Key Source Files

- `pjrt_implementation/src/api/flatbuffer_loaded_executable_instance.cc`
  - `execute()`
  - `prepareInputTensor()`
  - `fillPJRTOutputLists()`
- `pjrt_implementation/src/api/loaded_executable_instance.cc`
  - Base class for execution
- `pjrt_implementation/src/api/buffer_instance.cc`
  - Buffer management
- `pjrt_implementation/src/api/tensor.cc`
  - Tensor handling

## Execution Flow

```
execute()
├── Validate device count
├── Validate argument count
├── getOrCreateMeshDevice()
│   └── ClientInstance::getOrCreateMeshDevice()
├── getInputRuntimeTensors()
│   └── prepareInputTensor() for each input
│       ├── Get expected layout from flatbuffer
│       ├── Fill strategy map from sharding
│       └── Ensure tensor has correct layout
├── tt::runtime::submit()
│   └── Execute on tt-metal
├── Validate output count
└── fillPJRTOutputLists()
    └── Create BufferInstance for each output
```

## Runtime Tensor Preparation

### Input Tensor Requirements

1. **Correct shape**: Must match compiled input shape
2. **Correct dtype**: Must match compiled input dtype
3. **Correct layout**: Runtime may require specific memory layout
4. **Correct device**: Must be on expected device(s)

### Sharding Strategy

For sharded inputs, the runtime needs to know:
- Which devices hold which shards
- How data is distributed across devices

```cpp
mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
    fillStrategyMapFromSharding(sharding, num_devices);
```

## Error Messages Reference

| Error Message | Likely Cause | Fix |
|--------------|--------------|-----|
| `Device count mismatch` | Wrong number of devices | Match compile/run devices |
| `Argument count mismatch` | Wrong number of inputs | Check input count |
| `Runtime produced different number of outputs` | Runtime bug | Report to tt-metal |
| `Failed to fill strategy map` | Sharding mismatch | Check input sharding |
| `Early aborting execution in compile-only mode` | TT_COMPILE_ONLY_SYSTEM_DESC set | Unset env var |

## Debugging with ttrt

The `ttrt` tool provides powerful runtime debugging capabilities. See [tools-ttrt.md](tools-ttrt.md) for complete reference.

### Run with Golden Checks

```bash
# Run and verify accuracy against golden values
ttrt run out.ttnn

# Custom tolerance for accuracy check
ttrt run out.ttnn --identity --rtol 1 --atol 1

# Disable golden checks (for performance testing)
ttrt run out.ttnn --disable-golden

# Save golden tensors for analysis
ttrt run out.ttnn --save-golden-tensors --save-artifacts
```

### Step-Through Debugging

```bash
# Debugger mode: pause after each op with pdb
ttrt run out.ttnn --debugger
```

### Memory Analysis

```bash
# Analyze memory usage per operation
ttrt run out.ttnn --memory --save-artifacts

# Check for memory leaks
ttrt run out.ttnn --memory --check-memory-leak --save-artifacts

# Memory report shows:
# - DRAM/L1 allocation per op
# - Bytes per bank
# - Free/allocated memory
```

### Performance Profiling

Requires perf-enabled build (`-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`):

```bash
# Run performance analysis
ttrt perf out.ttnn --save-artifacts

# Host-only metrics (no device-side data)
ttrt perf out.ttnn --host-only

# Output files:
# - ops_perf_results.csv: per-op timing
# - tracy_profile_log_host.tracy: Tracy GUI file
```

### Print Tensor Values

```bash
# Print input/output tensor values for debugging
ttrt run out.ttnn --print-input-output-tensors
```

---

## Debugging Runtime Crashes

### Collect Debug Info

```bash
# Enable debug logging
export TTXLA_LOGGER_LEVEL=DEBUG
export TT_METAL_LOGGER_LEVEL=DEBUG
export TTRT_LOGGER_LEVEL=DEBUG

# Run with verbose output
pytest -svv path/to/test

# Or use ttrt directly
ttrt run out.ttnn --save-artifacts
```

### Check Hardware Status

```bash
# Device status
tt-smi

# Reset devices if needed (may require sudo)
tt-smi -r
```

### Memory Issues

- Large models may exceed device memory
- Check tensor sizes and total memory usage
- Use `ttrt run --memory` to analyze per-op memory
- Consider using smaller batch sizes or model sharding

### Hardware Faults

- Reset the device and retry
- Check for hardware errors in tt-smi
- Ensure proper cooling and power

## Checking if Stage 6 Failed

If compilation succeeds (all IR files generated including `ttnn_*.mlir`) but execution fails, the problem is in Stage 6.

**Success indicators**:
- Model compiles without errors
- IR files are generated
- But runtime execution crashes or produces wrong results
