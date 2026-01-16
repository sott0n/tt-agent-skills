# Step 1: Reference Model Analysis

## Objective

Analyze the reference PyTorch model to understand its structure, operators, and data flow before converting to TTNN.

## Tasks

### 1.1 Obtain the Reference PyTorch Model

- Get the original PyTorch model from sources like HuggingFace, torchvision, or custom implementations
- Ensure you can run inference and get golden outputs
- Save sample inputs and outputs for later validation

```python
import torch

# Example: Load a model from HuggingFace
from transformers import AutoModel, AutoConfig

model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval()

# Generate sample input
sample_input = torch.randint(0, config.vocab_size, (1, 128))

# Get golden output
with torch.no_grad():
    golden_output = model(sample_input)

# Save for later validation
torch.save({"input": sample_input, "output": golden_output}, "golden_data.pt")
```

### 1.2 Generate Model Graph

Use `torchviz` to visualize the model computation graph:

```python
from torchviz import make_dot

output = model(sample_input)
dot = make_dot(output.last_hidden_state, params=dict(model.named_parameters()))
dot.render("model_graph", format="pdf")
```

Alternatively, use `torch.fx` for a more detailed trace:

```python
from torch.fx import symbolic_trace

# For models that support symbolic tracing
traced = symbolic_trace(model)
print(traced.graph)
```

### 1.3 Extract Model Summary

Use `torchinfo` (successor to torchsummary) to extract detailed information:

```python
from torchinfo import summary

model_stats = summary(
    model,
    input_data=sample_input,
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    verbose=2
)
```

Document:
- All operators used in the model
- Input/output shapes for each operator
- Parameter counts and sizes
- Memory requirements

### 1.4 Document All Modules and Operators

Create a comprehensive list:

```python
# List all unique module types
module_types = set()
for name, module in model.named_modules():
    module_types.add(type(module).__name__)

print("Module types used:")
for mt in sorted(module_types):
    print(f"  - {mt}")

# List all parameters with shapes
print("\nParameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

### 1.5 Identify Module Hierarchies

Map out the module structure:

```python
def print_module_tree(module, prefix=""):
    for name, child in module.named_children():
        print(f"{prefix}{name}: {type(child).__name__}")
        print_module_tree(child, prefix + "  ")

print_module_tree(model)
```

## Deliverables

After completing this step, you should have:

1. **Working reference model** that can run inference
2. **Model graph visualization** (PDF or image)
3. **Operator list** documenting all PyTorch operators used
4. **Shape information** for all intermediate tensors
5. **Module hierarchy** showing the model structure
6. **Golden data** (sample inputs and expected outputs)

## Checklist

- [ ] Reference model loads and runs successfully
- [ ] Model graph has been generated and reviewed
- [ ] All operators have been identified and listed
- [ ] Input/output shapes are documented for each layer
- [ ] Module hierarchy is understood
- [ ] Golden inputs/outputs are saved for validation

## Next Step

Once all deliverables are ready, proceed to:
→ **Step 2: Operator Mapping** (`step-02-operator-mapping.md`)
