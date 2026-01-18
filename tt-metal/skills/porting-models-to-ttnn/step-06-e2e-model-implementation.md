# Step 6: End-to-End Model Implementation

## Objective

Compose all TTNN modules into a complete end-to-end model implementation.

## Prerequisites

- Completed Step 5 with all modules validated
- All module PCC results >= 0.99

## Tasks

### 6.1 Design Model Architecture

Plan how modules connect together:

```
Input Tokens
    ↓
[Embeddings] → word + position + token_type + LayerNorm
    ↓
[Encoder Layer 1] → Attention + FFN + Residuals
    ↓
[Encoder Layer 2] → Attention + FFN + Residuals
    ↓
    ...
    ↓
[Encoder Layer N] → Attention + FFN + Residuals
    ↓
[Pooler] (optional)
    ↓
Output
```

### 6.2 Create Main Model File

**ttnn_model.py:**
```python
import ttnn
from typing import Optional

from .ttnn_embeddings import embeddings
from .ttnn_attention import bert_self_attention_ttnn
from .ttnn_feedforward import feedforward_with_residual


def bert_encoder_layer(
    hidden_states,
    attention_mask,
    *,
    parameters,
    num_heads,
):
    """Single BERT encoder layer"""

    # Self-attention with residual
    attention_output = bert_self_attention_ttnn(
        hidden_states,
        attention_mask,
        parameters=parameters.attention,
        num_heads=num_heads,
    )

    # Attention output projection
    attention_output = ttnn.linear(
        attention_output,
        parameters.attention.output.dense.weight,
        bias=parameters.attention.output.dense.bias
    )

    # Residual + LayerNorm
    hidden_states = ttnn.add(attention_output, hidden_states)
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.attention.output.LayerNorm.weight,
        bias=parameters.attention.output.LayerNorm.bias
    )

    # Feed-forward with residual
    hidden_states = feedforward_with_residual(
        hidden_states,
        parameters=parameters,
    )

    return hidden_states


def bert_encoder(
    hidden_states,
    attention_mask,
    *,
    parameters,
    num_heads,
    num_layers,
):
    """Full BERT encoder (stack of layers)"""

    for layer_idx in range(num_layers):
        layer_params = getattr(parameters.encoder.layer, str(layer_idx))

        hidden_states = bert_encoder_layer(
            hidden_states,
            attention_mask,
            parameters=layer_params,
            num_heads=num_heads,
        )

    return hidden_states


def bert_model(
    input_ids,
    token_type_ids,
    attention_mask,
    *,
    parameters,
    config,
):
    """Complete BERT model"""

    # Embeddings
    hidden_states = embeddings(
        input_ids,
        token_type_ids,
        parameters=parameters.embeddings,
    )

    # Prepare attention mask
    # Expand from [batch, seq] to [batch, 1, 1, seq]
    if attention_mask is not None:
        extended_attention_mask = ttnn.reshape(attention_mask, (attention_mask.shape[0], 1, 1, attention_mask.shape[1]))
        # Convert 0/1 mask to -10000/0 for softmax
        extended_attention_mask = ttnn.multiply(
            ttnn.subtract(extended_attention_mask, 1.0),
            10000.0
        )
    else:
        extended_attention_mask = None

    # Encoder
    hidden_states = bert_encoder(
        hidden_states,
        extended_attention_mask,
        parameters=parameters,
        num_heads=config.num_attention_heads,
        num_layers=config.num_hidden_layers,
    )

    # Pooler (optional, for classification tasks)
    if hasattr(parameters, "pooler"):
        # Take [CLS] token output
        first_token = hidden_states[:, 0:1, :]
        pooled_output = ttnn.linear(
            first_token,
            parameters.pooler.dense.weight,
            bias=parameters.pooler.dense.bias
        )
        pooled_output = ttnn.tanh(pooled_output)
    else:
        pooled_output = None

    return hidden_states, pooled_output
```

### 6.3 Create Model Wrapper Class

For ease of use, create a wrapper class:

```python
class TtnnBertModel:
    """TTNN BERT Model wrapper"""

    def __init__(self, config, parameters, device):
        self.config = config
        self.parameters = parameters
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name, device):
        """Load from HuggingFace pretrained model"""
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        torch_model = AutoModel.from_pretrained(model_name).eval()

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            device=device,
        )

        return cls(config, parameters, device)

    def __call__(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        """Run inference"""

        # Convert inputs to TTNN
        ttnn_input_ids = ttnn.from_torch(
            input_ids,
            dtype=ttnn.uint32,
            device=self.device
        )

        if token_type_ids is not None:
            ttnn_token_type_ids = ttnn.from_torch(
                token_type_ids,
                dtype=ttnn.uint32,
                device=self.device
            )
        else:
            ttnn_token_type_ids = None

        if attention_mask is not None:
            ttnn_attention_mask = ttnn.from_torch(
                attention_mask.float(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device
            )
        else:
            ttnn_attention_mask = None

        # Run model
        hidden_states, pooled_output = bert_model(
            ttnn_input_ids,
            ttnn_token_type_ids,
            ttnn_attention_mask,
            parameters=self.parameters,
            config=self.config,
        )

        return hidden_states, pooled_output
```

### 6.4 Handle Device and Memory

For large models, manage memory across layers:

```python
def bert_encoder_memory_efficient(
    hidden_states,
    attention_mask,
    *,
    parameters,
    num_heads,
    num_layers,
    device,
):
    """Memory-efficient encoder with explicit memory management"""

    for layer_idx in range(num_layers):
        layer_params = getattr(parameters.encoder.layer, str(layer_idx))

        # Store previous hidden states for residual
        residual = hidden_states

        # Run attention
        attention_output = bert_self_attention_ttnn(
            hidden_states,
            attention_mask,
            parameters=layer_params.attention,
            num_heads=num_heads,
        )

        # Free hidden_states if not needed
        if layer_idx > 0:
            ttnn.deallocate(residual)

        # Continue with layer...
        hidden_states = attention_output

    return hidden_states
```

### 6.5 Add Configuration Support

Support model configurations:

```python
class TtnnModelConfig:
    """Configuration for TTNN model"""

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        max_position_embeddings: int = 512,
        vocab_size: int = 30522,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        # TTNN-specific options
        use_l1_cache: bool = False,
        use_sharding: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_l1_cache = use_l1_cache
        self.use_sharding = use_sharding

    @classmethod
    def from_huggingface(cls, hf_config):
        """Create from HuggingFace config"""
        return cls(
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            # ... map other fields
        )
```

### 6.6 Project Structure

Final project structure:

```
models/demos/<your_model>/
├── tt/
│   ├── __init__.py
│   ├── ttnn_model.py           # Main model (this file)
│   ├── ttnn_attention.py       # Attention module
│   ├── ttnn_feedforward.py     # FFN module
│   ├── ttnn_embeddings.py      # Embedding module
│   └── model_config.py         # Configuration
├── reference/
│   ├── torch_model.py          # Functional PyTorch reference
│   └── torch_modules.py        # Reference modules
├── tests/
│   ├── test_operators.py       # Step 3 tests
│   ├── test_modules.py         # Step 5 tests
│   └── test_model.py           # Step 7 tests
├── demo/
│   └── demo_inference.py       # Demo script
└── README.md
```

## Deliverables

1. **Complete TTNN model implementation**
2. **Model wrapper class** for easy usage
3. **Configuration support**
4. **Memory management** for large models

## Checklist

- [ ] All modules integrated into main model
- [ ] Model wrapper class created
- [ ] Configuration class implemented
- [ ] Memory management considered
- [ ] Model compiles without errors
- [ ] Demo script runs without errors

## Next Step

Once the model is fully implemented, proceed to:
→ **Step 7: End-to-End Model Testing** (`step-07-e2e-model-testing.md`)
