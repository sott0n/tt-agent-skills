# Step 2: Operator Mapping

## Objective

Map all PyTorch operators identified in Step 1 to their TTNN equivalents.

## Prerequisites

- Completed Step 1 with a list of all PyTorch operators used in the model

## Tasks

### 2.1 Review Operator List from Step 1

Take the operator list from Step 1 and categorize them:

```python
# Example operator list from a transformer model
operators = [
    "Embedding",
    "LayerNorm",
    "Linear",
    "GELU",
    "Softmax",
    "Dropout",  # Note: Usually removed in inference
    "MatMul",
    "Add",
]
```

### 2.2 Map to TTNN Operators

Use the comprehensive mapping table:
→ **See: `ttnn-operator-mapping.md`**

Create a mapping document for your specific model:

```python
# Model-specific operator mapping
OPERATOR_MAPPING = {
    # Layer                    PyTorch Op          TTNN Op
    "embeddings":              ("nn.Embedding",    "ttnn.embedding"),
    "attention.query":         ("nn.Linear",       "ttnn.linear"),
    "attention.key":           ("nn.Linear",       "ttnn.linear"),
    "attention.value":         ("nn.Linear",       "ttnn.linear"),
    "attention.softmax":       ("Softmax",         "ttnn.softmax"),
    "attention.matmul":        ("matmul",          "ttnn.matmul"),
    "layer_norm":              ("nn.LayerNorm",    "ttnn.layer_norm"),
    "ffn.linear1":             ("nn.Linear",       "ttnn.linear"),
    "ffn.gelu":                ("GELU",            "ttnn.gelu"),
    "ffn.linear2":             ("nn.Linear",       "ttnn.linear"),
}
```

### 2.3 Identify Missing Operators

For operators not in the mapping table:

1. **Search Documentation**
   ```bash
   # Check TTNN API reference
   grep -r "<operator_name>" docs/source/ttnn/ttnn/api.rst
   ```

2. **Search Source Code**
   ```bash
   # Search Python bindings
   grep -r "def <operator_name>" ttnn/ttnn/

   # Search C++ operations
   ls ttnn-metal/ttnn/cpp/ttnn/operations/
   ```

3. **Check Existing Models**
   ```bash
   # See how other models handle similar operators
   grep -r "ttnn\." models/demos/
   ```

### 2.4 Handle Unsupported Operators

For operators not available in TTNN:

**Option A: Composite Operations**
```python
# Example: GELU can be approximated with other ops if not available
def gelu_approximation(x):
    return x * 0.5 * (1.0 + ttnn.tanh(0.7978845608 * (x + 0.044715 * ttnn.pow(x, 3))))
```

**Option B: PyTorch Fallback**
```python
def fallback_op(ttnn_tensor, torch_op):
    # Convert to torch, run op, convert back
    torch_tensor = ttnn.to_torch(ttnn_tensor)
    result = torch_op(torch_tensor)
    return ttnn.from_torch(result, device=ttnn_tensor.device())
```

**Option C: Request Implementation**
- File a GitHub issue at https://github.com/tenstorrent/tt-metal/issues
- Include: operator name, use case, expected behavior

### 2.5 Note API Differences

Document any differences between PyTorch and TTNN APIs:

```python
# Example: Linear layer differences
# PyTorch: nn.Linear(in_features, out_features, bias=True)
#   - Weight shape: (out_features, in_features)
#   - Output = input @ weight.T + bias

# TTNN: ttnn.linear(input, weight, bias=None)
#   - Weight should be pre-transposed for optimal performance
#   - May need to reshape weight: weight.T for compatibility
```

### 2.6 Document Shape Requirements

TTNN operators may have specific shape requirements:

```python
# TTNN shape requirements
SHAPE_REQUIREMENTS = {
    "ttnn.matmul": {
        "tile_layout": "Inner dimensions must be multiples of 32",
        "batch_dims": "Supports batched matmul with broadcasting",
    },
    "ttnn.conv2d": {
        "input_format": "NHWC format (unlike PyTorch NCHW)",
        "padding": "Height/width must be padded to multiples of 32",
    },
    "ttnn.layer_norm": {
        "normalized_shape": "Last dimension must be normalized",
    },
}
```

## Deliverables

1. **Operator Mapping Table** for your specific model
2. **List of Unsupported Operators** with fallback strategies
3. **API Difference Notes** for operators with different signatures
4. **Shape Requirement Notes** for operators with specific constraints

## Checklist

- [ ] All PyTorch operators mapped to TTNN equivalents
- [ ] Unsupported operators identified with fallback plans
- [ ] API differences documented
- [ ] Shape requirements noted
- [ ] Mapping table saved for reference during implementation

## Example: Complete Mapping for BERT

```python
BERT_OPERATOR_MAPPING = {
    # Embeddings
    "word_embeddings": ("nn.Embedding", "ttnn.embedding"),
    "position_embeddings": ("nn.Embedding", "ttnn.embedding"),
    "token_type_embeddings": ("nn.Embedding", "ttnn.embedding"),
    "LayerNorm": ("nn.LayerNorm", "ttnn.layer_norm"),

    # Attention
    "query": ("nn.Linear", "ttnn.linear"),
    "key": ("nn.Linear", "ttnn.linear"),
    "value": ("nn.Linear", "ttnn.linear"),
    "attention_scores": ("matmul", "ttnn.matmul"),
    "attention_probs": ("Softmax", "ttnn.softmax"),
    "context": ("matmul", "ttnn.matmul"),
    "output.dense": ("nn.Linear", "ttnn.linear"),

    # FFN
    "intermediate.dense": ("nn.Linear", "ttnn.linear"),
    "intermediate.activation": ("GELU", "ttnn.gelu"),
    "output.dense": ("nn.Linear", "ttnn.linear"),

    # Operations
    "add": ("+", "ttnn.add"),
    "dropout": ("nn.Dropout", "REMOVED"),  # No dropout in inference
}
```

## Next Step

Once all operators are mapped, proceed to:
→ **Step 3: Per-Operator Testing** (`step-03-per-operator-testing.md`)
