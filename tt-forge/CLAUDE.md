# tt-forge CLAUDE.md

This file provides guidance to Claude Code when working with tt-forge.

## Overview

tt-forge is the DNN compiler for Tenstorrent hardware, using tt-xla, tt-onnx-fe as frontend to compile and run models on TT devices. It integrates TorchDynamo, Torch XLA (PJRT), tt-mlir, and tt-metal.

## Compilation Pipeline

```
[Stage 0] Model Definition (tt-forge-models)
    │ torch.compile(model, backend="tt")
    ▼
[Stage 1] Frontend Tracing (TorchDynamo/Torch XLA)
    │ FX Graph → StableHLO/VHLO
    ▼
[Stage 2] PJRT Interface (tt-xla)
    ▼
[Stage 3] StableHLO → TTIR (tt-xla + tt-mlir)
    ▼
[Stage 4] TTIR → TTNN (tt-mlir)
    ▼
[Stage 5] Flatbuffer Generation (tt-mlir)
    ▼
[Stage 6] Runtime Execution (tt-metal)
```

## Skills

| Skill | Description |
|-------|-------------|
| `tt-forge-bringup` | Bring up new models on tt-forge. Use when adding a new model or running a model on TT hardware for the first time. |
| `tt-forge-debug` | Debug compilation/execution errors across the tt-xla, tt-mlir, tt-metal stack. Use when errors occur during bringup or execution. |
| `tt-forge-test` | Run tests and validate PCC/atol accuracy. Use after code changes to verify correctness. |
| `tt-forge-review` | Review code changes for quality and consistency. Use after tests pass, before committing. |
| `tt-forge-perf` | Measure model performance (latency, throughput, bottlenecks). Use to identify optimization opportunities. |
| `tt-forge-optimize` | Implement performance optimizations in tt-mlir. Use after perf analysis identifies bottlenecks. |

## Typical Workflow

```
tt-forge-bringup (setup new model)
        │
        ▼ error?
tt-forge-debug (diagnose/fix)
        │
        ▼ success?
tt-forge-test (validate correctness)
        │
        ▼ optimize?
tt-forge-perf (measure bottlenecks)
        │
        ▼
tt-forge-optimize (implement)
        │
        ▼
tt-forge-test → tt-forge-perf (iterate)
        │
        ▼ done?
tt-forge-review (code review)
```

## Key Directories

| Path | Description |
|------|-------------|
| `tests/torch/models/` | Model test files |
| `third_party/tt_forge_models/` | Model definitions and loaders |
| `third_party/tt-mlir/src/tt-mlir/` | MLIR compiler (has its own CLAUDE.md) |
| `python_package/tt_torch/backend/` | Backend passes and decompositions |

## Important Principles

1. **No workarounds in Tenstorrent software** - Stage 2-6 issues require proper fixes, not hacks. Workarounds only acceptable in Stage 0 (model code) or Stage 1 (frontend).

2. **Read component CLAUDE.md before modifying** - When fixing issues in tt-mlir, read `third_party/tt-mlir/src/tt-mlir/CLAUDE.md` first.

3. **Check known failures first** - Before debugging, check test configs for known issues with documented reasons.

## Common Commands

```bash
# Run model test with debug logging
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv tests/torch/models/<model>/test_<model>.py

# Performance analysis
ttrt perf out.ttnn --save-artifacts

# Check known failures
grep -A5 "model_name" tests/runner/test_config/torch/test_config_inference_single_device.yaml
```
