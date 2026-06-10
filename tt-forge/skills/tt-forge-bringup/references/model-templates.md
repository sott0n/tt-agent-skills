# Model & Test File Templates

Copy-paste skeletons for bringing up a new model. Used by **Step 2**
(create the loader) and **Step 3** (create the test file) of the
`tt-forge-bringup` skill.

## Contents

- [loader.py (tt-forge-models)](#loaderpy-tt-forge-models)
- [test file (TorchModelTester)](#test-file-torchmodeltester)

## loader.py (tt-forge-models)

Location: `third_party/tt_forge_models/<model_name>/pytorch/loader.py`

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

## test file (TorchModelTester)

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
