# Step 4: Module Implementation

## Objective

Implement TTNN modules by converting PyTorch modules to functional TTNN implementations.

## Prerequisites

- Completed Step 3 with all operators validated
- Understanding of model's module hierarchy from Step 1

## Tasks

### 4.1 Rewrite PyTorch Modules as Functional Code

The first step is to rewrite PyTorch modules using functional APIs (no `nn.Module` classes):

**Original PyTorch Module:**
```python
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)

        # Context
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return context
```

**Functional PyTorch Version:**
```python
def bert_self_attention_torch(
    hidden_states,
    attention_mask,
    *,
    parameters,
    num_heads,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads

    # Linear projections
    q = hidden_states @ parameters.query.weight.T + parameters.query.bias
    k = hidden_states @ parameters.key.weight.T + parameters.key.bias
    v = hidden_states @ parameters.value.weight.T + parameters.value.bias

    # Reshape for multi-head attention
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = torch.softmax(scores, dim=-1)

    # Context
    context = torch.matmul(probs, v)
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    return context
```

### 4.2 Convert to TTNN Implementation

**TTNN Module Version:**
```python
import ttnn
import math

def bert_self_attention_ttnn(
    hidden_states,
    attention_mask,
    *,
    parameters,
    num_heads,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads

    # Linear projections using ttnn.linear
    q = ttnn.linear(hidden_states, parameters.query.weight, bias=parameters.query.bias)
    k = ttnn.linear(hidden_states, parameters.key.weight, bias=parameters.key.bias)
    v = ttnn.linear(hidden_states, parameters.value.weight, bias=parameters.value.bias)

    # Reshape for multi-head attention
    q = ttnn.reshape(q, (batch_size, seq_len, num_heads, head_dim))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [batch, heads, seq, head_dim]

    k = ttnn.reshape(k, (batch_size, seq_len, num_heads, head_dim))
    k = ttnn.permute(k, (0, 2, 1, 3))

    v = ttnn.reshape(v, (batch_size, seq_len, num_heads, head_dim))
    v = ttnn.permute(v, (0, 2, 1, 3))

    # Attention scores
    k_t = ttnn.permute(k, (0, 1, 3, 2))  # Transpose last two dims
    scores = ttnn.matmul(q, k_t)
    scores = ttnn.multiply(scores, 1.0 / math.sqrt(head_dim))

    if attention_mask is not None:
        scores = ttnn.add(scores, attention_mask)

    probs = ttnn.softmax(scores, dim=-1)

    # Context
    context = ttnn.matmul(probs, v)
    context = ttnn.permute(context, (0, 2, 1, 3))  # [batch, seq, heads, head_dim]
    context = ttnn.reshape(context, (batch_size, seq_len, hidden_size))

    return context
```

### 4.3 Use `preprocess_model_parameters`

Convert PyTorch model weights to TTNN format:

```python
from ttnn.model_preprocessing import preprocess_model_parameters

def create_ttnn_parameters(torch_model, device):
    """Convert PyTorch model parameters to TTNN format"""

    def custom_preprocessor(model, name):
        """Custom preprocessing for specific layers"""
        parameters = {}

        if hasattr(model, "weight"):
            # Transpose weight for linear layers (TTNN expects different layout)
            weight = model.weight.T.contiguous()
            parameters["weight"] = ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device
            )

        if hasattr(model, "bias") and model.bias is not None:
            parameters["bias"] = ttnn.from_torch(
                model.bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device
            )

        return parameters

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )

    return parameters
```

### 4.4 Project Structure

Organize your TTNN model code:

```
models/
└── demos/
    └── <your_model>/
        ├── tt/
        │   ├── __init__.py
        │   ├── ttnn_<model>.py          # Main model
        │   ├── ttnn_attention.py         # Attention module
        │   ├── ttnn_feedforward.py       # FFN module
        │   └── ttnn_embeddings.py        # Embedding module
        ├── reference/
        │   └── torch_<model>.py          # Functional PyTorch reference
        └── tests/
            ├── test_attention.py
            ├── test_feedforward.py
            └── test_model.py
```

### 4.5 Example: Complete Module Implementation

**ttnn_feedforward.py:**
```python
import ttnn

def feedforward(
    hidden_states,
    *,
    parameters,
):
    """BERT Feed-Forward Network"""
    # Intermediate: hidden_size -> intermediate_size
    intermediate = ttnn.linear(
        hidden_states,
        parameters.intermediate.dense.weight,
        bias=parameters.intermediate.dense.bias
    )
    intermediate = ttnn.gelu(intermediate)

    # Output: intermediate_size -> hidden_size
    output = ttnn.linear(
        intermediate,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias
    )

    return output


def feedforward_with_residual(
    hidden_states,
    *,
    parameters,
):
    """FFN with residual connection and layer norm"""
    residual = hidden_states

    # FFN
    output = feedforward(hidden_states, parameters=parameters)

    # Residual + LayerNorm
    output = ttnn.add(output, residual)
    output = ttnn.layer_norm(
        output,
        weight=parameters.output.LayerNorm.weight,
        bias=parameters.output.LayerNorm.bias
    )

    return output
```

### 4.6 Handle Memory Management

For large models, manage memory explicitly:

```python
def memory_efficient_module(hidden_states, *, parameters, device):
    """Example with explicit memory management"""

    # Move input to L1 for faster access
    hidden_states = ttnn.to_memory_config(
        hidden_states,
        ttnn.L1_MEMORY_CONFIG
    )

    # Process
    output = ttnn.linear(hidden_states, parameters.weight)

    # Deallocate intermediate tensors
    ttnn.deallocate(hidden_states)

    # Move output back to DRAM for storage
    output = ttnn.to_memory_config(
        output,
        ttnn.DRAM_MEMORY_CONFIG
    )

    return output
```

## Deliverables

1. **Functional PyTorch modules** as intermediate step
2. **TTNN module implementations** for each module
3. **Parameter preprocessing code**
4. **Organized project structure**

## Checklist

- [ ] PyTorch modules rewritten as functional code
- [ ] TTNN modules implemented using ttnn operators
- [ ] Parameter preprocessing function created
- [ ] Project structure organized
- [ ] All modules compile without errors

## Next Step

Once all modules are implemented, proceed to:
→ **Step 5: Per-Module Testing** (`step-05-per-module-testing.md`)
