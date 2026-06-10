---
name: tt-forge-bringup
description: Bring up or compile new models on tt-forge. Use when user says "bringup", "compile", "add model", or wants to run a new model on Tenstorrent hardware for the first time.
---

# TT-Forge Model Bringup Skill

## When to Use

A new model: not yet in the test suite, or running on TT hardware for the
first time. **Not for** debugging existing failures (use tt-forge-debug).

## Bringup Workflow

```
- [ ] Step 1: Check tt-forge-models for existing model
- [ ] Step 2: Create/update model in tt-forge-models if needed
- [ ] Step 3: Create test file with TorchModelTester pattern
- [ ] Step 4: First run with export_path
- [ ] Step 5: If error → tt-forge-debug Skill
- [ ] Step 6: Iterate until execution succeeds
```

## Step 1: Check tt-forge-models

```bash
ls third_party/tt_forge_models/ | grep -i "<model_name>"
```

| Result | Action |
|--------|--------|
| Found | Use existing `ModelLoader` |
| Not found | Create new model (Step 2) |

## Step 2: Create Model in tt-forge-models

Location: `third_party/tt_forge_models/<model_name>/pytorch/loader.py`

Implement a `ModelLoader(ForgeModel)` with a `_VARIANTS` map plus
`load_model()` and `load_inputs()`. Copy-paste skeleton:
[`references/model-templates.md`](references/model-templates.md#loaderpy-tt-forge-models).

## Step 3: Create Test File

Location: `tests/torch/models/<model_name>/test_<model_name>.py`

Subclass `TorchModelTester` (`_get_model` / `_get_input_activations` wired to
the loader) and add a `@pytest.mark.push` test. Copy-paste skeleton:
[`references/model-templates.md`](references/model-templates.md#test-file-torchmodeltester).

## Step 4: First Run

```bash
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv tests/torch/models/<model_name>/test_<model_name>.py
```

## Step 5: Handle Errors

If error occurs → **invoke tt-forge-debug Skill**

Do not debug here. Bringup skill focuses on setup, debug skill handles errors.

## Step 6: Success Report

```
**Model**: <model_name>
**Status**: BRINGUP COMPLETE
**Next**: tt-forge-test for PCC validation
```

## Integration

Position in the loop: **tt-forge-bringup** (this skill; setup) → on error
**tt-forge-debug** → on success **tt-forge-test** (PCC validation). Full
pipeline in `tt-forge/CLAUDE.md`.
