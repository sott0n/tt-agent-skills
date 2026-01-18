# Step 5: Per-Module Testing

## Objective

Test each TTNN module in isolation against its PyTorch reference to validate correctness before integrating into the full model.

## Prerequisites

- Completed Step 4 with all modules implemented
- Functional PyTorch reference modules from Step 4

## Tasks

### 5.1 Create Module Test Structure

```
tests/
└── ttnn/
    └── unit_tests/
        └── models/
            └── <your_model>/
                ├── test_embeddings.py
                ├── test_attention.py
                ├── test_feedforward.py
                ├── test_encoder_layer.py
                └── conftest.py
```

### 5.2 Write Module Tests

**test_attention.py:**
```python
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.demos.<your_model>.tt.ttnn_attention import bert_self_attention_ttnn
from models.demos.<your_model>.reference.torch_attention import bert_self_attention_torch
from ttnn.model_preprocessing import preprocess_model_parameters


class TestBertSelfAttention:

    @pytest.fixture
    def config(self):
        return {
            "hidden_size": 768,
            "num_attention_heads": 12,
            "batch_size": 1,
            "seq_len": 128,
        }

    @pytest.fixture
    def torch_model(self, config):
        """Create PyTorch reference model"""
        from transformers.models.bert.modeling_bert import BertSelfAttention, BertConfig

        bert_config = BertConfig(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
        )
        return BertSelfAttention(bert_config).eval()

    @pytest.fixture
    def parameters(self, torch_model, device):
        """Preprocess parameters for TTNN"""
        return preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            device=device,
        )

    def test_self_attention(self, device, config, torch_model, parameters):
        torch.manual_seed(0)

        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]

        # Create input
        torch_input = torch.randn(batch_size, seq_len, hidden_size)

        # PyTorch reference
        with torch.no_grad():
            torch_output = torch_model(torch_input)[0]

        # TTNN
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_output = bert_self_attention_ttnn(
            ttnn_input,
            attention_mask=None,
            parameters=parameters,
            num_heads=num_heads,
        )

        # Compare
        output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, output, 0.999)

    def test_self_attention_with_mask(self, device, config, torch_model, parameters):
        torch.manual_seed(0)

        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]

        # Create input with attention mask
        torch_input = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.zeros(batch_size, 1, 1, seq_len)
        attention_mask[:, :, :, seq_len//2:] = -10000.0  # Mask second half

        # PyTorch reference
        with torch.no_grad():
            torch_output = torch_model(torch_input, attention_mask=attention_mask)[0]

        # TTNN
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_output = bert_self_attention_ttnn(
            ttnn_input,
            attention_mask=ttnn_mask,
            parameters=parameters,
            num_heads=num_heads,
        )

        output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, output, 0.999)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [128, 256, 512])
    def test_various_shapes(self, device, config, torch_model, parameters, batch_size, seq_len):
        """Test with various input shapes"""
        torch.manual_seed(0)

        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]

        torch_input = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            torch_output = torch_model(torch_input)[0]

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_output = bert_self_attention_ttnn(
            ttnn_input,
            attention_mask=None,
            parameters=parameters,
            num_heads=num_heads,
        )

        output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, output, 0.999)
```

### 5.3 Test Feedforward Module

**test_feedforward.py:**
```python
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.demos.<your_model>.tt.ttnn_feedforward import feedforward, feedforward_with_residual


class TestFeedForward:

    @pytest.fixture
    def config(self):
        return {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "batch_size": 1,
            "seq_len": 128,
        }

    @pytest.fixture
    def torch_model(self, config):
        """Create PyTorch reference"""
        from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertConfig

        bert_config = BertConfig(
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
        )
        intermediate = BertIntermediate(bert_config).eval()
        output = BertOutput(bert_config).eval()
        return intermediate, output

    def test_feedforward(self, device, config, torch_model):
        torch.manual_seed(0)

        intermediate_module, output_module = torch_model
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        hidden_size = config["hidden_size"]

        # Create input
        torch_input = torch.randn(batch_size, seq_len, hidden_size)

        # PyTorch reference
        with torch.no_grad():
            intermediate_output = intermediate_module(torch_input)
            torch_output = output_module(intermediate_output, torch_input)  # with residual

        # Prepare parameters
        parameters = preprocess_model_parameters(...)  # Your preprocessing

        # TTNN
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_output = feedforward_with_residual(
            ttnn_input,
            parameters=parameters,
        )

        output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, output, 0.999)
```

### 5.4 Test Encoder Layer (Composed Module)

**test_encoder_layer.py:**
```python
class TestEncoderLayer:
    """Test a complete encoder layer (attention + FFN)"""

    def test_encoder_layer(self, device, config, torch_model, parameters):
        torch.manual_seed(0)

        # Create input
        torch_input = torch.randn(
            config["batch_size"],
            config["seq_len"],
            config["hidden_size"]
        )

        # PyTorch reference (full encoder layer)
        with torch.no_grad():
            torch_output = torch_model(torch_input)[0]

        # TTNN
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_output = encoder_layer_ttnn(
            ttnn_input,
            attention_mask=None,
            parameters=parameters,
        )

        output = ttnn.to_torch(ttnn_output)

        # Note: Composed modules may have slightly lower PCC
        assert_with_pcc(torch_output, output, 0.99)
```

### 5.5 Debug Module PCC Issues

When a module test fails, use incremental debugging:

```python
def debug_module_step_by_step(hidden_states, parameters, device, torch_model):
    """Debug by comparing intermediate outputs"""

    torch_input = ttnn.to_torch(hidden_states)

    # Step 1: Test attention query projection
    torch_q = torch_model.attention.self.query(torch_input)
    ttnn_q = ttnn.linear(hidden_states, parameters.attention.self.query.weight, ...)
    pcc_q = assert_with_pcc(torch_q, ttnn.to_torch(ttnn_q), 0.999, return_pcc=True)
    print(f"Query projection PCC: {pcc_q}")

    # Step 2: Test attention scores
    # ... continue step by step

    # This helps identify which specific operation is causing PCC degradation
```

### 5.6 Run Module Tests

```bash
# Run all module tests
pytest tests/ttnn/unit_tests/models/<your_model>/ -v

# Run specific module test
pytest tests/ttnn/unit_tests/models/<your_model>/test_attention.py -v

# Run with verbose output for debugging
pytest tests/ttnn/unit_tests/models/<your_model>/test_attention.py -v -s

# Run with coverage
pytest tests/ttnn/unit_tests/models/<your_model>/ -v --cov=models.demos.<your_model>
```

## Deliverables

1. **Module test files** for each TTNN module
2. **PCC results table** for each module
3. **Debug notes** for any modules with PCC issues

## Checklist

- [ ] Test file created for each module
- [ ] All modules pass PCC >= 0.99
- [ ] Tests cover various input shapes
- [ ] Tests include edge cases (masking, padding)
- [ ] Composed modules tested (encoder layer, decoder layer)

## Module PCC Results Template

| Module | Components | PCC | Status | Notes |
|--------|------------|-----|--------|-------|
| Embeddings | word + position + token_type | 0.999 | ✅ | |
| Self-Attention | Q,K,V + softmax + output | 0.998 | ✅ | |
| FFN | linear + gelu + linear | 0.999 | ✅ | |
| Encoder Layer | attention + FFN + residuals | 0.997 | ✅ | Accumulated error |
| LayerNorm | | 0.999 | ✅ | |

## Next Step

Once all modules pass PCC validation, proceed to:
→ **Step 6: End-to-End Model Implementation** (`step-06-e2e-model-implementation.md`)
