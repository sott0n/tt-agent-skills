# Stage 3: StableHLO → TTIR (tt-xla + tt-mlir)

- **Modifiable**: Yes
- **Input**: StableHLO MLIR module
- **Output**: TTIR MLIR module

## Contents

- [What Happens in This Stage](#what-happens-in-this-stage)
- [Common Issues](#common-issues)
  - [Frontend Pipeline Failures](#frontend-pipeline-failures)
  - [VHLO to StableHLO Failures](#vhlo-to-stablehlo-failures)
  - [StableHLO to TTIR Failures](#stablehlo-to-ttir-failures)
- [Debugging](#debugging)
- [Error Messages Reference](#error-messages-reference)

## What Happens in This Stage

1. `runFrontendSHLOPipeline()` - tt-xla frontend passes
2. `collectInputShardings()` / `collectOutputShardings()` - Extract sharding info
3. `runCompilerStableHLOPipeline()` - tt-mlir StableHLO passes
4. `setProperSdyMeshAttributeInSpmdMode()` - Set mesh attributes
5. `convertFromSHLOToTTIR()` - Convert to TTIR dialect

## Common Issues

### Frontend Pipeline Failures

**Symptom**: Errors related to sharding attributes

**Error patterns**:
- `Failed to create input shardings from GSPMD attributes`
- `Failed to create input shardings from Shardy attributes`
- `Failed to create output shardings from GSPMD attributes`
- `Failed to create output shardings from Shardy attributes`

**Cause**: Invalid or incompatible sharding annotations

**Debug**:
```bash
export TTXLA_LOGGER_LEVEL=DEBUG
# Check shlo_frontend_*.mlir in export_path
```

### StableHLO Pipeline Failures

**Symptom**: `Failed to run stablehlo pipeline`

**Cause**: Unsupported StableHLO operation or pattern

**Debug**:
1. Check `shlo_compiler_*.mlir` for the last successful state
2. Enable verbose IR printing:
```bash
export TTXLA_LOGGER_LEVEL=VERBOSE
```

### Mesh Attribute Errors

**Symptom**: `Failed to set proper sdy.mesh attribute in SPMD mode`

**Cause**: Inconsistent mesh configuration between sharding annotations

**Debug**: Check mesh attributes in `shlo_set_mesh_attr_*.mlir`

### StableHLO to TTIR Conversion Failures

**Symptom**: `Failed to convert from SHLO to TTIR module`

**Cause**: Unsupported StableHLO operation

**Debug**:
1. Check which operation failed in verbose output
2. Look for the operation in tt-mlir's supported ops

## Frontend Passes (tt-xla)

### shlo_input_role_propagation

Propagates input argument attributes (parameter vs input roles).

**Related errors**: `Failed to uplift mark parameters custom call`

### shlo_clean_for_xla_ingestion

Cleans StableHLO for XLA compatibility.

### shlo_set_proper_sdy_mesh_attribute

Sets Shardy mesh attributes in SPMD mode.

## Sharding Modes

### GSPMD (Google SPMD)

Traditional XLA sharding using `mhlo.sharding` attributes.

**Attributes**:
- `mhlo.num_partitions`
- `mhlo.num_replicas`
- `xla.sharding` on arguments/results

### Shardy

Newer sharding dialect using `sdy.mesh` and `sdy.sharding`.

**Detection**: Module has `sdy.mesh` operation

**Attributes**:
- `sdy.mesh` operation
- `sdy.sharding` on arguments/results
- `sdy.manual_computation` operations

## Compile Options Affecting This Stage

```python
options = {
    "export_path": "/path/to/dump",  # Dump IR at each stage
}
```

## IR Files Generated

| File | Stage |
|------|-------|
| `shlo_*.mlir` | After VHLO→StableHLO |
| `shlo_frontend_*.mlir` | After frontend pipeline |
| `shlo_compiler_*.mlir` | After compiler StableHLO pipeline |
| `shlo_set_mesh_attr_*.mlir` | After mesh attribute setting |
| `ttir_*.mlir` | After SHLO→TTIR conversion |

## Key Source Files

- `pjrt_implementation/src/api/module_builder/module_builder.cc`
  - `runFrontendSHLOPipeline()`
  - `runCompilerStableHLOPipeline()`
  - `convertFromSHLOToTTIR()`
- `pjrt_implementation/src/api/module_builder/frontend_passes/`
  - `shlo_input_role_propagation.cc`
  - `shlo_clean_for_xla_ingestion.cc`
  - `shlo_set_proper_sdy_mesh_attribute.cc`

## tt-mlir Pipelines Used

### StableHLOPipeline

```cpp
mlir::tt::stablehlo::StableHLOPipelineOptions options;
options.resultPresharded = result_presharded;
mlir::tt::stablehlo::createStableHLOPipeline(pm, options);
```

### StableHLOToTTIRPipeline

```cpp
mlir::tt::ttir::StableHLOToTTIRPipelineOptions options;
options.arithDialectConversionsEnabled = true;
options.legalizeCompositeToCallEnabled = true;
mlir::tt::ttir::createStableHLOToTTIRPipeline(pm, options);
```

## Error Messages Reference

| Error Message | Stage | Fix |
|--------------|-------|-----|
| `Failed to create input shardings from GSPMD attributes` | Sharding collection | Check sharding annotations |
| `Failed to create input shardings from Shardy attributes` | Sharding collection | Check sdy.mesh/sdy.sharding |
| `Failed to run stablehlo pipeline` | Compiler StableHLO | Check unsupported ops |
| `Failed to set proper sdy.mesh attribute` | Mesh setup | Check mesh consistency |
| `Failed to convert from SHLO to TTIR module` | SHLO→TTIR | Check unsupported StableHLO ops |
| `Expected exactly one manual computation op, found: N` | Shardy processing | Check manual_computation ops |

## Checking if Stage 3 Succeeded

Check `<export_path>/irs/` for:
- `shlo_frontend_*.mlir` - Frontend pipeline succeeded
- `shlo_compiler_*.mlir` - Compiler pipeline succeeded
- `ttir_*.mlir` - TTIR conversion succeeded

If `ttir_*.mlir` doesn't exist, the failure is in Stage 3.
