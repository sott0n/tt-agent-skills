# Stage 0: Model Definition

- **Modifiable**: Yes (model code, tt-forge-models)
- **Output**: Model ready for `torch.compile(backend="tt")`

## Contents

- [Basic Usage Pattern](#basic-usage-pattern)
- [Compile Options](#compile-options)
- [Common Issues and Fixes](#common-issues-and-fixes)
- [Model Structure Requirements](#model-structure-requirements)
- [Testing Your Model](#testing-your-model)
- [Known Failures and xfail Tracking](#known-failures-and-xfail-tracking)
- [tt-forge-models Library](#tt-forge-models-library)
- [Using tt-forge-models in Tests](#using-tt-forge-models-in-tests)
- [Creating New Models](#creating-new-models)

## Basic Usage Pattern

```python
import torch

# torch.compile with "tt" backend
model = MyModel().to(dtype=torch.bfloat16)
compiled = torch.compile(model, backend="tt", options={
    "export_path": "./debug_output",  # For debugging
})
output = compiled(input_x)
```

## Compile Options

Key options for debugging:

```python
options = {
    "export_path": "/path/to/dump",    # Dump intermediate IR
    "export_model_name": "my_model",   # Name prefix for IR files
    "optimization_level": 0,           # 0=none, 1+=optimizer passes
}
```

See [compile-options.md](compile-options.md) for complete reference.

## Common Issues and Fixes

### Issue: Unsupported dtype

**Symptom**: Error about unsupported data type

**Fix**: Use supported dtypes
```python
# Supported dtypes
torch.bfloat16  # Recommended for most models
torch.float32   # Supported but may be slower
torch.int32     # For integer operations

# Model should use bfloat16
model = MyModel().to(dtype=torch.bfloat16)
input_x = torch.randn(..., dtype=torch.bfloat16)
```

### Issue: Dynamic shapes

**Symptom**: `Dynamic dimensions not supported`

**Fix**: Use static shapes
```python
# BAD: Dynamic batch size
def forward(self, x):  # x can have any batch size
    return self.linear(x)

# GOOD: Fixed batch size or use padding
input_x = torch.randn(32, 32, dtype=torch.bfloat16)  # Fixed shape
```

### Issue: Model not on XLA device

**Symptom**: `Passing a non-XLA tensor to TT compile was likely not intended`

**Fix**: Ensure model and inputs are on XLA device for eager mode
```python
device = xm.xla_device()
model = model.to(device)
input_x = input_x.to(device)
```

### Issue: Parameters not registered properly

**Symptom**: Missing parameters or buffers in compiled graph

**Fix**: Use proper PyTorch module registration
```python
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # GOOD: Register as parameter/buffer
        self.weight = torch.nn.Parameter(torch.randn(32, 32))
        self.register_buffer('bias', torch.zeros(32))

        # BAD: Not registered
        # self.weight = torch.randn(32, 32)
```

## Model Structure Requirements

### Supported Operations

Most standard PyTorch operations are supported. Common supported ops:
- Linear layers (`nn.Linear`)
- Convolutions (`nn.Conv2d`, `nn.Conv3d`)
- Activations (`ReLU`, `GELU`, `SiLU`, etc.)
- Normalization (`LayerNorm`, `BatchNorm`)
- Attention (`scaled_dot_product_attention`)
- Pooling (`MaxPool2d`, `AvgPool2d`)
- Element-wise ops (`+`, `-`, `*`, `/`, `matmul`)

### Composite Operations

tt-xla recognizes certain patterns as composite ops for better optimization:
- RMSNorm
- Softmax
- GELU
- SiLU
- Attention patterns

Enable/disable with:
```python
options = {"tt_enable_composite_ops": True}
```

## Testing Your Model

### Minimal Test

```python
import torch
from torch_plugin_tt import TTPlugin
import torch_xla

# Register TT plugin
torch_xla.plugins.use_dynamic_plugins()
torch_xla.plugins.register_plugin("tt", TTPlugin())

model = MyModel()
compiled = torch.compile(model, backend="tt")

# Run with small input first
small_input = torch.randn(1, 32, dtype=torch.bfloat16)
try:
    output = compiled(small_input)
    print("Compilation successful!")
except Exception as e:
    print(f"Compilation failed: {e}")
```

### Debug with IR export

```python
options = {
    "export_path": "./debug_output",
    "export_model_name": "my_model",
}
compiled = torch.compile(model, backend="tt", options=options)
output = compiled(input_x)

# Check ./debug_output/irs/ for intermediate IR files
```

---

## Known Failures and xfail Tracking

**Before debugging a model failure, check if it's a known issue.**

### YAML Test Configuration (Primary Source)

`tests/runner/test_config/torch/test_config_inference_single_device.yaml` contains centralized status for all models:

```yaml
clip/pytorch-Large_Patch14-single_device-inference:
  status: EXPECTED_PASSING
  arch_overrides:
    p150:
      status: KNOWN_FAILURE_XFAIL
      reason: "AssertionError: PCC comparison failed. Calculated: pcc=0.983..."

perceiverio_vision/pytorch-Vision_Perceiver_Conv-single_device-inference:
  status: KNOWN_FAILURE_XFAIL
  reason: "Can't convert shape rank (https://github.com/tenstorrent/tt-xla/issues/3392)"

pi_0/pytorch-lerobot_pi0_libero_base-single_device-inference:
  arch_overrides:
    p150:
      status: NOT_SUPPORTED_SKIP
      reason: "Hangs - https://github.com/tenstorrent/tt-xla/issues/3922"
      bringup_status: FAILED_RUNTIME
```

### Status Values

| Status | Meaning |
|--------|---------|
| `EXPECTED_PASSING` | Should pass, tracked regression |
| `KNOWN_FAILURE_XFAIL` | Known failure, expected to fail |
| `NOT_SUPPORTED_SKIP` | Not supported on this architecture |

### BringupStatus Values

| BringupStatus | Stage | Meaning |
|---------------|-------|---------|
| `FAILED_FE_COMPILATION` | Stage 1 | TorchDynamo/Torch XLA failure |
| `FAILED_TTMLIR_COMPILATION` | Stage 3-5 | tt-mlir compilation failure |
| `FAILED_RUNTIME` | Stage 6 | Runtime execution failure |
| `INCORRECT_RESULT` | Stage 6 | Runs but produces wrong output |
| `PASSED` | - | All stages pass |

### Quick Lookup Commands

```bash
# Find status for a specific model
grep -A5 "resnet.*single_device" tests/runner/test_config/torch/test_config_inference_single_device.yaml

# Find all known failures
grep -B2 "KNOWN_FAILURE_XFAIL" tests/runner/test_config/torch/test_config_inference_single_device.yaml

# Find failures with GitHub issues
grep -E "reason:.*github.com" tests/runner/test_config/torch/test_config_inference_single_device.yaml
```

### xfail in Test Files

Individual test files may also have `@pytest.mark.xfail`:

```python
# tests/torch/models/whisper/test_whisper.py
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "RuntimeError: Not enough space to allocate 6710886400 B DRAM buffer "
        "across 12 banks - https://github.com/tenstorrent/tt-xla/issues/1886"
    )
)
def test_whisper_large():
    ...
```

### Helper Functions (tests/utils.py)

```python
from utils import (
    failed_fe_compilation,      # Stage 1 failure
    failed_ttmlir_compilation,  # Stage 3-5 failure
    failed_runtime,             # Stage 6 failure
    incorrect_result,           # Wrong output
    BringupStatus,              # Enum for status tracking
)
```

---

## tt-forge-models Library

tt-forge-models provides a standardized way to load and run 186+ pre-configured models. All models follow the `ForgeModel` abstract base class pattern.

### ForgeModel Base Class

```python
from abc import ABC, abstractmethod
from typing import Dict

class ForgeModel(ABC):
    """Abstract base class for all models in tt-forge-models."""

    # Map of variant enum -> ModelConfig
    _VARIANTS: Dict[StrEnum, ModelConfig] = {}

    # Default variant to use when none specified
    DEFAULT_VARIANT = None

    @abstractmethod
    def load_model(self, **kwargs):
        """Load and return the PyTorch model."""
        pass

    @abstractmethod
    def load_inputs(self, **kwargs):
        """Load and return sample input tensors."""
        pass

    @classmethod
    @abstractmethod
    def _get_model_info(cls, variant) -> ModelInfo:
        """Return metadata about the model variant."""
        pass

    def get_mesh_config(self, num_devices: int):
        """Return mesh configuration for multi-device execution."""
        return None, ()

    def load_shard_spec(self, model):
        """Return sharding specification for distributed execution."""
        return None
```

### Model Variants Pattern

Each model defines variants using an enum and `_VARIANTS` dictionary:

```python
from enum import StrEnum
from third_party.tt_forge_models import ForgeModel
from third_party.tt_forge_models.config import ModelConfig, ModelTask, ModelSource

class ModelVariant(StrEnum):
    RESNET_50_HF = "resnet_50_hf"
    RESNET_101_HF = "resnet_101_hf"

class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.RESNET_50_HF: ModelConfig(
            name="microsoft/resnet-50",
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGINGFACE,
        ),
        ModelVariant.RESNET_101_HF: ModelConfig(
            name="microsoft/resnet-101",
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGINGFACE,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.RESNET_50_HF
```

### Using tt-forge-models

```python
from third_party.tt_forge_models.resnet.pytorch import ModelLoader, ModelVariant

# Instantiate with a variant
loader = ModelLoader(ModelVariant.RESNET_50_HF)

# Load model and inputs
model = loader.load_model()
inputs = loader.load_inputs(batch_size=1)

# Compile and run
compiled = torch.compile(model, backend="tt")
output = compiled(*inputs)
```

### ModelConfig and ModelInfo

```python
from third_party.tt_forge_models.config import (
    ModelConfig,    # Configuration for loading a model variant
    ModelInfo,      # Metadata about a model (name, task, source, etc.)
    ModelTask,      # Enum: CV_IMAGE_CLS, NLP_TEXT_CLS, AUDIO_ASR, etc.
    ModelSource,    # Enum: HUGGINGFACE, TORCHVISION, TIMM, CUSTOM
    Framework,      # Enum: PYTORCH, JAX
)
```

---

## Using tt-forge-models in Tests

### TorchModelTester Pattern

Tests inherit from `TorchModelTester` and wrap a `ModelLoader`:

```python
from infra import TorchModelTester, ComparisonConfig, RunMode
from third_party.tt_forge_models.resnet.pytorch import ModelLoader

class ResnetTester(TorchModelTester):
    def __init__(self, variant_name, comparison_config=ComparisonConfig(),
                 run_mode=RunMode.INFERENCE, compiler_config=None):
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode, compiler_config)

    def _get_model(self):
        return self._model_loader.load_model()

    def _get_input_activations(self):
        return self._model_loader.load_inputs()
```

### Test Example

```python
import pytest
from third_party.tt_forge_models.resnet.pytorch import ModelVariant
from .tester import ResnetTester

@pytest.mark.push
@pytest.mark.single_device
def test_resnet_inference():
    tester = ResnetTester(ModelVariant.RESNET_50_HF)
    tester.test()
```

---

## Creating New Models

### Step 1: Create Model Directory

```
third_party/tt_forge_models/
└── my_model/
    └── pytorch/
        ├── __init__.py
        └── loader.py
```

### Step 2: Define Variants and ModelLoader

```python
# loader.py
from enum import StrEnum
from third_party.tt_forge_models import ForgeModel
from third_party.tt_forge_models.config import ModelConfig, ModelInfo, ModelTask, ModelSource

class ModelVariant(StrEnum):
    MY_MODEL_SMALL = "my_model_small"
    MY_MODEL_LARGE = "my_model_large"

class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.MY_MODEL_SMALL: ModelConfig(
            name="my-org/my-model-small",
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGINGFACE,
        ),
        ModelVariant.MY_MODEL_LARGE: ModelConfig(
            name="my-org/my-model-large",
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGINGFACE,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MY_MODEL_SMALL

    def load_model(self, *, dtype_override=None, **kwargs):
        config = self._VARIANTS[self.variant]
        # Load from HuggingFace, torchvision, etc.
        model = load_from_source(config.name, config.source)
        if dtype_override:
            model = model.to(dtype=dtype_override)
        return model

    def load_inputs(self, batch_size=1, dtype_override=None, **kwargs):
        # Return preprocessed sample inputs
        return (torch.randn(batch_size, 3, 224, 224, dtype=dtype_override),)

    @classmethod
    def _get_model_info(cls, variant) -> ModelInfo:
        config = cls._VARIANTS[variant]
        return ModelInfo(
            name=config.name,
            task=config.task,
            source=config.source,
        )
```

### Step 3: Export from __init__.py

```python
# __init__.py
from .loader import ModelLoader, ModelVariant
```

### Multi-Device Support (Optional)

For models that support multi-device execution:

```python
class ModelLoader(ForgeModel):
    # ... _VARIANTS, etc.

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, device_ids) for multi-device."""
        if num_devices == 8:
            return (2, 4), (0, 1, 2, 3, 4, 5, 6, 7)
        return None, ()

    def load_shard_spec(self, model):
        """Return sharding specification for model parameters."""
        return {
            "model.embed_tokens.weight": ShardSpec(...),
            "model.layers.*.attention.q_proj.weight": ShardSpec(...),
        }
```
