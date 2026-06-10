---
name: tt-forge-debug
description: Debug and fix compilation/execution errors in tt-xla, tt-mlir, and tt-metal stack. Use when errors occur during model bringup or execution.
---

# TT-Forge Debug Skill

## When to Use

- Compilation error occurred
- Runtime error or crash
- Model produces incorrect results
- Called from tt-forge-bringup when errors occur

**Not for**: Initial model setup (use tt-forge-bringup first)

## Debug Workflow Checklist

```
- [ ] Step 0: Check known failures (grep YAML config)
- [ ] Step 1: Check IR dump directory for last successful stage
- [ ] Step 2: Match error message to stage
- [ ] Step 3: Read stage-specific reference file
- [ ] Step 4: Apply fix (workaround or proper fix)
- [ ] Step 5: Re-run and verify fix
```

## Important Principles

### No Workarounds in Tenstorrent Software

**Do NOT apply workaround fixes to Tenstorrent software components (tt-xla, tt-mlir, tt-metal).**

- Stage 2-6 issues require **proper fixes**, not hacks
- If a bug is found in tt-xla/tt-mlir/tt-metal, report it and fix it correctly
- Workarounds are only acceptable in:
  - **Stage 0** (Model code) - e.g., restructure model to avoid unsupported patterns
  - **Stage 1** (Frontend) - e.g., use `torch._dynamo.allow_in_graph` for graph breaks

### Read Component CLAUDE.md When Modifying

**When debugging or fixing issues in submodule components, read their CLAUDE.md first:**

- **tt-mlir** (Stage 4-5): Read `third_party/tt-mlir/src/tt-mlir/CLAUDE.md`
  - Contains build commands, test commands, code style guidelines
  - Required for proper tt-mlir development workflow

---

## Quick Diagnosis: Where Did It Fail?

### Step 0: Check Known Failures First

**Before debugging, check if it's a known issue with documented reason:**

```bash
# Find status for a specific model in YAML config
grep -A5 "model_name.*single_device" tests/runner/test_config/torch/test_config_inference_single_device.yaml

# Find all known failures with reasons
grep -B2 -A2 "KNOWN_FAILURE_XFAIL" tests/runner/test_config/torch/test_config_inference_single_device.yaml

# Search for xfail markers in test files
grep -r "xfail" tests/torch/models/
```

**Key files:**
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` - Centralized status + reasons
- `tests/torch/models/*/test_*.py` - Individual test xfail markers

If found, the `reason` field contains the error message and often a GitHub issue link.

See [stage0-models.md](references/stage0-models.md#known-failures-and-xfail-tracking) for details.

### Step 1: Check IR Dump Directory

If `export_path` was set, check which IR files exist in `<export_path>/irs/`:

| File exists? | Last successful stage | Failed at |
|--------------|----------------------|-----------|
| None | - | Stage 1 (Frontend) or Stage 2 (PJRT init) |
| `vhlo_*.mlir` only | VHLO parse | Stage 2-3 (VHLO→StableHLO) |
| `shlo_*.mlir` | StableHLO | Stage 3 (StableHLO pipeline) |
| `shlo_frontend_*.mlir` | Frontend pipeline | Stage 3 (Compiler StableHLO) |
| `shlo_compiler_*.mlir` | Compiler StableHLO | Stage 3 (SHLO→TTIR) |
| `ttir_*.mlir` | TTIR | Stage 4 (TTIR→TTNN) |
| `ttnn_*.mlir` | TTNN | Stage 5 (Flatbuffer) or Stage 6 (Runtime) |

### Step 2: Match Error Message Pattern

| Error message contains | Stage | Reference |
|------------------------|-------|-----------|
| `graph break`, `Dynamo`, `unsupported.*torch` | 1 | [stage1-frontend.md](references/stage1-frontend.md) |
| `PJRT_Client`, `device initialization`, `mesh device` | 2 | [stage2-pjrt.md](references/stage2-pjrt.md) |
| `VHLO`, `Failed to create VHLO module` | 2-3 | [stage3-stablehlo.md](references/stage3-stablehlo.md) |
| `Failed to convert from VHLO to SHLO` | 3 | [stage3-stablehlo.md](references/stage3-stablehlo.md) |
| `stablehlo pipeline`, `sharding`, `GSPMD`, `Shardy` | 3 | [stage3-stablehlo.md](references/stage3-stablehlo.md) |
| `Failed to convert from SHLO to TTIR` | 3 | [stage3-stablehlo.md](references/stage3-stablehlo.md) |
| `Failed to convert from TTIR to TTNN`, `legalize` | 4 | [stage4-ttir-ttnn.md](references/stage4-ttir-ttnn.md) |
| `Failed to generate flatbuffer`, `serialization` | 5 | [stage5-flatbuffer.md](references/stage5-flatbuffer.md) |
| `runtime`, `submit`, `execute`, `tt-metal`, `Argument count mismatch` | 6 | [stage6-runtime.md](references/stage6-runtime.md) |

> **Graph breaks (Stage 1):** for *excessive* graph breaks (a model
> generating more graphs than expected), the official tt-xla
> **`graph-break-analysis`** skill analyzes the cause and proposes fixes —
> reach for it rather than debugging graph splits by hand. See "Related
> Skills (official tt-xla)" below.

### Step 3: Check Stack Trace Source Files

| Source file path | Stage |
|------------------|-------|
| `torch/_dynamo/`, `torch_xla/` | 1 (Frontend) |
| `client_instance.cc`, `device_instance.cc` | 2 (PJRT) |
| `module_builder.cc`, `shlo_*.cc` | 3 (StableHLO) |
| `ttmlir/Dialect/TTIR/`, `ttmlir/Dialect/TTNN/` | 4 (tt-mlir) |
| `TTNNToFlatbuffer`, `flatbuffer` | 5 (Flatbuffer) |
| `tt/runtime/`, `tt-metal`, `loaded_executable_instance.cc` | 6 (Runtime) |

## Debug Environment Variables

```bash
# Enable debug logging (shows each compilation stage)
export TTXLA_LOGGER_LEVEL=DEBUG

# Enable verbose IR printing between passes
export TTXLA_LOGGER_LEVEL=VERBOSE

# Enable XLA debug info in StableHLO
export XLA_HLO_DEBUG=1

# Dump intermediate IR to directory
# In compile options: {"export_path": "/path/to/dump"}
```

## Compilation Pipeline

```
[Stage 0] Model Definition (tt-forge-models)
    │ torch.compile(model, backend="tt")
    ▼
[Stage 1] Frontend Tracing (TorchDynamo/Torch XLA) ← External, cannot modify
    │ FX Graph → StableHLO/VHLO
    ▼
[Stage 2] PJRT Interface (tt-xla)
    │ ClientInstance::compileMlirProgram()
    │ createVHLOModule() → convertFromVHLOToSHLO()
    ▼
[Stage 3] StableHLO → TTIR (tt-xla + tt-mlir)
    │ runFrontendSHLOPipeline() → runCompilerStableHLOPipeline()
    │ convertFromSHLOToTTIR()
    ▼
[Stage 4] TTIR → TTNN (tt-mlir)
    │ convertFromTTIRToTTNN()
    │ TTIRToTTNNBackendPipeline
    ▼
[Stage 5] Flatbuffer Generation (tt-mlir)
    │ createFlatbufferBinary()
    ▼
[Stage 6] Runtime Execution (tt-metal)
    │ tt::runtime::submit()
```

## Responsibility Boundaries

| Stage | Component | Can Modify? | Common Issues |
|-------|-----------|-------------|---------------|
| 0 | tt-forge-models | Yes | Model definition, input shapes |
| 1 | TorchDynamo/Torch XLA | No (External) | Graph breaks, unsupported ops |
| 2 | tt-xla (PJRT) | Yes | Device init, mesh config |
| 3 | tt-xla + tt-mlir | Yes | StableHLO conversion |
| 4 | tt-mlir | Yes | Unsupported ops, legalization |
| 5 | tt-mlir | Yes | Serialization errors |
| 6 | tt-metal | Yes | Runtime errors, memory |

## Stage Reference Files

- [Stage 0: Model Definition](references/stage0-models.md) - Model issues, input shapes, tt-forge-models
- [Stage 1: Frontend (External)](references/stage1-frontend.md) - TorchDynamo, graph breaks, workarounds
- [Stage 2: PJRT Interface](references/stage2-pjrt.md) - Device init, mesh configuration, distributed runtime
- [Stage 3: StableHLO→TTIR](references/stage3-stablehlo.md) - StableHLO pipeline, conversion
- [Stage 4: TTIR→TTNN](references/stage4-ttir-ttnn.md) - tt-mlir pipeline, op support
- [Stage 5: Flatbuffer](references/stage5-flatbuffer.md) - Binary generation, verification
- [Stage 6: Runtime](references/stage6-runtime.md) - Execution, tt-metal errors

## Additional References

- [Compile Options](references/compile-options.md) - Complete reference for all torch.compile options
- [ttrt Tool](references/tools-ttrt.md) - Flatbuffer inspection, execution, and debugging tool

## Related Skills (official tt-xla)

When working in the tt-xla repo, these official skills automate specific
debug/triage cases this workflow points at. Prefer them over manual
triage when the failure matches; they live in
`tenstorrent/tt-xla/.claude/skills/`.

- **`graph-break-analysis`** — analyzes and proposes fixes for excessive
  graph breaks in PyTorch/XLA compilation (**Stage 1**). Use when a model
  generates more graphs than expected, or the error mentions "graph
  break". Note the common misconception it corrects: different MLIR
  modules (e.g. VHLO versions) are *not* graph breaks.
- **`triage-dtype-bfloat16`** — triages one tt-forge-models *training*
  test failing with a bfloat16 dtype-mismatch `RuntimeError` (e.g. "mat1
  and mat2 must have the same dtype, but got Float and BFloat16", or
  "'<op>' not implemented for 'BFloat16'"). Attempts a minimal loader
  `dtype_override` fix, re-runs CPU + pytest, and updates the test YAML
  (`EXPECTED_PASSING` / `KNOWN_FAILURE_XFAIL`). Pairs with **Step 0**
  (known-failure YAML) above.
- **`triage-unpack-forward-output`** — triages one tt-forge-models
  *training* test stuck at `FAILED_FE_COMPILATION` with reason
  "tt-forge-models doesn't implement unpack_forward_output for this
  model" (**Stage 0**). Inspects the model's forward output, registers a
  handler or per-loader override, and updates the YAML.

These triage skills are narrow (one failure pattern each) and
tt-xla/tt-forge-models-specific; this `tt-forge-debug` skill remains the
general entry point for stack-wide diagnosis.

## Debug Feedback Loop

```
Identify → Read → Fix → Verify → Repeat
    │        │      │       │
    │        │      │       └─ Check new IR files appear
    │        │      └─ Model workaround (Stage 0-1) or proper fix (Stage 2-6)
    │        └─ Load stage-specific reference file
    └─ Match error to stage using tables above
```

If new error appears after fix, return to "Identify" step.

