# Stage 5: Flatbuffer Generation (tt-mlir)

- **Modifiable**: Yes (via tt-mlir)
- **Input**: TTNN MLIR module
- **Output**: Flatbuffer binary for runtime

## Contents

- [What Happens in This Stage](#what-happens-in-this-stage)
- [Common Issues](#common-issues)
  - [Flatbuffer Generation Failures](#flatbuffer-generation-failures)
  - [Verification Failures](#verification-failures)
- [Debugging with ttrt](#debugging-with-ttrt)
- [Error Messages Reference](#error-messages-reference)

## What Happens in This Stage

1. `createFlatbufferBinary()` - Generate flatbuffer from TTNN module
2. `ttnnToFlatbuffer()` - tt-mlir's serialization function
3. `verifyCreatedFlatbufferBinary()` - Validation checks
   - Verify input count matches shardings
   - Verify output count matches shardings
   - Check output sharding shapes

## Common Issues

### Flatbuffer Generation Failures

**Symptom**: `Failed to generate flatbuffer binary`

**Cause**: TTNN module has invalid structure or unsupported operations

**Debug**:
```bash
export TTXLA_LOGGER_LEVEL=DEBUG
# Check ttnn_*.mlir for the input TTNN module
```

### Input/Output Count Mismatch

**Symptom**:
- `Created flatbuffer binary contains different number of inputs N than expected from the m_input_shardings M`
- `Created flatbuffer binary contains different number of outputs N than expected from the m_output_shardings M`

**Cause**: Mismatch between frontend sharding info and compiled binary

**Debug**: Compare sharding annotations with flatbuffer program inputs/outputs

### Output Sharding Shape Errors

**Symptom**:
- `Output sharding shape (N) doesn't match the output shape (M)`
- `Output shape (N) is not divisible by the sharding shape (M)`

**Cause**: Invalid sharding configuration for output tensors

**Fix**: Ensure output dimensions are divisible by shard dimensions

## Verification Checks

### Input Count Verification

```cpp
if (num_inputs != input_shardings.size()) {
  // Error: input count mismatch
}
```

### Output Count Verification

```cpp
if (num_outputs != output_shardings.size()) {
  // Error: output count mismatch
}
```

### Output Sharding Shape Verification

For each sharded output:
1. Shard shape rank must match output shape rank
2. Each output dimension must be divisible by corresponding shard dimension

## Key Source Files

- `pjrt_implementation/src/api/module_builder/module_builder.cc`
  - `createFlatbufferBinary()`
  - `verifyCreatedFlatbufferBinary()`
  - `checkOutputShardingShapes()`
- tt-mlir (external):
  - `ttmlir/Target/TTNN/TTNNToFlatbuffer.h`

## Flatbuffer Binary Structure

The flatbuffer binary contains:
- **Programs**: List of executable programs
- **Inputs**: Input tensor specifications
- **Outputs**: Output tensor specifications
- **Operations**: Serialized TTNN operations
- **Metadata**: System info, mesh shape, etc.

## Debugging with ttrt

The `ttrt` tool from tt-mlir is essential for debugging flatbuffer issues. See [tools-ttrt.md](tools-ttrt.md) for complete reference.

### Inspect Flatbuffer Contents

```bash
# Read all sections
ttrt read out.ttnn --section all

# Check specific sections
ttrt read --section version out.ttnn      # Version info
ttrt read --section system_desc out.ttnn  # Target system
ttrt read --section mlir out.ttnn         # Embedded MLIR
ttrt read --section inputs out.ttnn       # Input specifications
ttrt read --section outputs out.ttnn      # Output specifications
ttrt read --section mesh_shape out.ttnn   # Mesh configuration

# Save extracted artifacts for analysis
ttrt read out.ttnn --save-artifacts --artifact-dir ./debug
```

### Validate Flatbuffer Against System

```bash
# Check if flatbuffer is compatible with current system
ttrt check out.ttnn

# Check against specific system descriptor
ttrt check out.ttnn --system-desc /path/to/system_desc.ttsys
```

### Verify Against TTNN IR

Compare `ttnn_*.mlir` with flatbuffer structure:
- Number of function arguments = Number of inputs
- Number of function results = Number of outputs

```bash
# Extract and compare
ttrt read --section inputs out.ttnn
# Compare with function arguments in ttnn_*.mlir
```

## Error Messages Reference

| Error Message | Likely Cause | Fix |
|--------------|--------------|-----|
| `Failed to generate flatbuffer binary` | Invalid TTNN module | Check ttnn_*.mlir |
| `different number of inputs` | Frontend/backend mismatch | Check sharding config |
| `different number of outputs` | Frontend/backend mismatch | Check sharding config |
| `Output sharding shape doesn't match` | Invalid sharding | Fix sharding dimensions |
| `Output shape not divisible by sharding` | Incompatible dimensions | Adjust output or shard shape |

## Checking if Stage 5 Succeeded

If `ttnn_*.mlir` exists but execution fails with flatbuffer-related errors, the failure is in Stage 5.

**Success indicators**:
- No flatbuffer generation errors
- Verification passes complete
- Program proceeds to runtime execution (Stage 6)

## Compile Options

This stage is primarily affected by the TTNN module structure from Stage 4. No specific compile options directly control this stage.

## Related to Runtime

The flatbuffer binary is the input to Stage 6 (Runtime). If the binary is generated but execution fails, the problem is in Stage 6, not Stage 5.
