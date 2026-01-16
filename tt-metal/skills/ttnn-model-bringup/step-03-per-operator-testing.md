# Step 3: Per-Operator Testing

## Objective

Test each TTNN operator individually to validate correctness using PCC (Pearson Correlation Coefficient) comparison against PyTorch reference outputs.

## Prerequisites

- Completed Step 2 with operator mapping table
- Access to Tenstorrent device

## Tasks

### 3.1 Set Up Test Infrastructure

Create a test file structure:

```
tests/
└── ttnn/
    └── unit_tests/
        └── models/
            └── <your_model>/
                ├── test_operators.py
                └── conftest.py
```

**conftest.py:**
```python
import pytest
import ttnn

@pytest.fixture(scope="module")
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)
```

### 3.2 Write Unit Tests for Each Operator

Use the `assert_with_pcc` utility for validation:

```python
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

class TestOperators:

    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("seq_len", [128, 512])
    @pytest.mark.parametrize("hidden_size", [768])
    def test_linear(self, device, batch_size, seq_len, hidden_size):
        torch.manual_seed(0)

        # Create PyTorch reference
        in_features = hidden_size
        out_features = hidden_size * 4

        torch_input = torch.randn(batch_size, seq_len, in_features)
        torch_linear = torch.nn.Linear(in_features, out_features)
        torch_output = torch_linear(torch_input)

        # Convert to TTNN
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        # Prepare weights (transpose for TTNN)
        weight = torch_linear.weight.T.contiguous()
        ttnn_weight = ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_bias = ttnn.from_torch(
            torch_linear.bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        # Run TTNN operator
        ttnn_output = ttnn.linear(ttnn_input, ttnn_weight, bias=ttnn_bias)

        # Convert back and compare
        output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, output, 0.999)

    def test_layer_norm(self, device):
        torch.manual_seed(0)

        batch_size, seq_len, hidden_size = 1, 128, 768

        # PyTorch reference
        torch_input = torch.randn(batch_size, seq_len, hidden_size)
        torch_ln = torch.nn.LayerNorm(hidden_size)
        torch_output = torch_ln(torch_input)

        # TTNN
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_weight = ttnn.from_torch(
            torch_ln.weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_bias = ttnn.from_torch(
            torch_ln.bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_output = ttnn.layer_norm(
            ttnn_input,
            weight=ttnn_weight,
            bias=ttnn_bias
        )

        output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, output, 0.999)

    def test_gelu(self, device):
        torch.manual_seed(0)

        # PyTorch reference
        torch_input = torch.randn(1, 128, 768)
        torch_output = torch.nn.functional.gelu(torch_input)

        # TTNN
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_output = ttnn.gelu(ttnn_input)

        output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, output, 0.999)

    def test_softmax(self, device):
        torch.manual_seed(0)

        # PyTorch reference
        torch_input = torch.randn(1, 12, 128, 128)  # [batch, heads, seq, seq]
        torch_output = torch.nn.functional.softmax(torch_input, dim=-1)

        # TTNN
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_output = ttnn.softmax(ttnn_input, dim=-1)

        output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, output, 0.999)

    def test_matmul(self, device):
        torch.manual_seed(0)

        # PyTorch reference
        A = torch.randn(1, 12, 128, 64)
        B = torch.randn(1, 12, 64, 128)
        torch_output = torch.matmul(A, B)

        # TTNN
        ttnn_A = ttnn.from_torch(
            A,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_B = ttnn.from_torch(
            B,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        ttnn_output = ttnn.matmul(ttnn_A, ttnn_B)

        output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, output, 0.999)
```

### 3.3 Run Tests

```bash
# Run all operator tests
pytest tests/ttnn/unit_tests/models/<your_model>/test_operators.py -v

# Run specific test
pytest tests/ttnn/unit_tests/models/<your_model>/test_operators.py::TestOperators::test_linear -v

# Run with specific parameters
pytest tests/ttnn/unit_tests/models/<your_model>/test_operators.py -v -k "batch_size-1"
```

### 3.4 PCC Guidelines

| PCC Threshold | Interpretation |
|---------------|----------------|
| >= 0.9999 | Excellent - Nearly identical results |
| >= 0.999 | Good - Acceptable for most operators |
| >= 0.99 | Fair - May indicate precision differences |
| < 0.99 | Poor - Investigate numerical issues |

**Common causes of low PCC:**
- Data type precision (float32 vs bfloat16)
- Algorithm differences (e.g., different GELU approximations)
- Padding/masking issues
- Incorrect tensor layout or shape

### 3.5 Debug Low PCC

```python
def debug_pcc(torch_output, ttnn_output, name=""):
    """Debug helper for PCC issues"""
    import numpy as np
    from scipy.stats import pearsonr

    t1 = torch_output.flatten().float().numpy()
    t2 = ttnn_output.flatten().float().numpy()

    pcc, _ = pearsonr(t1, t2)

    print(f"\n{name} Debug Info:")
    print(f"  PCC: {pcc:.6f}")
    print(f"  Torch - min: {t1.min():.4f}, max: {t1.max():.4f}, mean: {t1.mean():.4f}")
    print(f"  TTNN  - min: {t2.min():.4f}, max: {t2.max():.4f}, mean: {t2.mean():.4f}")
    print(f"  Max diff: {np.abs(t1 - t2).max():.6f}")
    print(f"  Mean diff: {np.abs(t1 - t2).mean():.6f}")

    # Find location of max difference
    diff = np.abs(t1 - t2)
    max_idx = np.argmax(diff)
    print(f"  Max diff at idx {max_idx}: torch={t1[max_idx]:.4f}, ttnn={t2[max_idx]:.4f}")

    return pcc
```

### 3.6 Test with Model-Specific Shapes

Test operators with the exact shapes from your model:

```python
# Shapes extracted from Step 1 model analysis
MODEL_SHAPES = {
    "embedding_output": (1, 512, 768),
    "attention_scores": (1, 12, 512, 512),
    "ffn_intermediate": (1, 512, 3072),
}

@pytest.mark.parametrize("shape", list(MODEL_SHAPES.values()))
def test_with_model_shapes(device, shape):
    # Test operators with actual model shapes
    pass
```

## Deliverables

1. **Unit Tests** for each TTNN operator
2. **PCC Results Table** documenting achieved PCC for each operator
3. **Issue List** for operators with PCC < 0.999

## Checklist

- [ ] Test infrastructure set up
- [ ] Unit test written for each operator
- [ ] All tests pass with PCC >= 0.999
- [ ] Low PCC issues investigated and documented
- [ ] Tests use model-specific shapes

## PCC Results Template

| Operator | PyTorch | TTNN | PCC | Status |
|----------|---------|------|-----|--------|
| Linear | nn.Linear | ttnn.linear | 0.9998 | ✅ |
| LayerNorm | nn.LayerNorm | ttnn.layer_norm | 0.9997 | ✅ |
| GELU | F.gelu | ttnn.gelu | 0.9995 | ✅ |
| Softmax | F.softmax | ttnn.softmax | 0.9996 | ✅ |
| MatMul | torch.matmul | ttnn.matmul | 0.9999 | ✅ |

## Next Step

Once all operators pass PCC validation, proceed to:
→ **Step 4: Module Implementation** (`step-04-module-implementation.md`)
