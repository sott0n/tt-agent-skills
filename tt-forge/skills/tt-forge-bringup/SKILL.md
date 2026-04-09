---
name: tt-forge-bringup
description: Bring up or compile new models on tt-forge. Use when user says "bringup", "compile", "add model", or wants to run a new model on Tenstorrent hardware for the first time.
---

# TT-Forge Model Bringup Skill

## When to Use

- Starting work on a new model
- Model doesn't exist in test suite yet
- First time running a model on TT hardware

**Not for**: Debugging existing failures (use tt-forge-debug)

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

Location: `third_party/tt_forge_models/<model_name>/pytorch/`

```python
# loader.py
from enum import StrEnum
from third_party.tt_forge_models import ForgeModel
from third_party.tt_forge_models.config import ModelConfig, ModelTask, ModelSource

class ModelVariant(StrEnum):
    DEFAULT = "default"

class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            name="org/model-name",
            task=ModelTask.CV_IMAGE_CLS,  # or NLP_TEXT_CLS, etc.
            source=ModelSource.HUGGINGFACE,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def load_model(self, **kwargs):
        # Load from HuggingFace/torchvision/etc.
        ...

    def load_inputs(self, batch_size=1, **kwargs):
        return (torch.randn(batch_size, 3, 224, 224, dtype=torch.bfloat16),)
```

## Step 3: Create Test File

Location: `tests/torch/models/<model_name>/test_<model_name>.py`

```python
import pytest
from infra import TorchModelTester, ComparisonConfig
from third_party.tt_forge_models.<model_name>.pytorch import ModelLoader, ModelVariant

class Tester(TorchModelTester):
    def __init__(self, variant, comparison_config=ComparisonConfig(), **kwargs):
        self._loader = ModelLoader(variant)
        super().__init__(comparison_config, **kwargs)

    def _get_model(self):
        return self._loader.load_model()

    def _get_input_activations(self):
        return self._loader.load_inputs()

@pytest.mark.push
def test_model():
    Tester(ModelVariant.DEFAULT).test()
```

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

```
tt-forge-bringup (setup)
        │
        ▼ error?
tt-forge-debug (diagnosis/fix)
        │
        ▼ success?
tt-forge-test (PCC validation)
```
