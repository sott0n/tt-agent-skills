---
name: porting-models-to-ttnn
description: "Converts PyTorch models to TTNN for Tenstorrent hardware. Covers operator mapping, PCC validation, and module implementation. Use when porting PyTorch/HuggingFace models to TTNN, mapping torch operators to ttnn equivalents, debugging PCC failures, or implementing TTNN modules."
---

# TTNN Model Bringup

## Prerequisites

Before running TTNN programs, see the project CLAUDE.md for Build Commands, Python Environment, and Environment Variables.

## Execution Environment

TTNN models can be developed in two different environments. **Ask the user which environment they are using before starting model implementation.**

| Environment | Setup | Use Case |
|-------------|-------|----------|
| **Out-of-tree** | `pip install ttnn` | Quick prototyping, standalone projects, external repositories |
| **In-tree** | `./build_metal.sh` source build | Contributing to tt-metal/models, production models, full test integration |

### Out-of-tree (pip install)

```bash
# Install TTNN
pip install ttnn

# Set Python path (run from your project root)
export PYTHONPATH=$(pwd)

# CPU performance settings (Linux)
sudo apt-get install cpufrequtils
sudo cpupower frequency-set -g performance
```

- Standalone Python project
- No need to build tt-metal from source
- Suitable for rapid prototyping and experimentation
- Model code lives in user's own repository

### In-tree (source build)

```bash
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
./build_metal.sh
./create_venv.sh
source python_env/bin/activate
```

- Model code added to `models/` directory in tt-metal
- Full access to internal utilities and test infrastructure
- Required for contributing models to tt-metal repository
- See project CLAUDE.md for detailed build instructions

## Model-Type Specific Guides

For detailed guidance based on your model architecture:

| Model Type | Guide | Description |
|------------|-------|-------------|
| LLM | `llm-model-bringup.md` | Transformers, Llama, Falcon, BERT, GPT, etc. |
| CNN | `cnn-model-bringup.md` | ResNet, YOLO, VGG, image models |

**Choose the appropriate guide based on your model type**, then follow the step-by-step workflow below.

## Step-by-Step Workflow

Follow these steps in order. Complete each step before proceeding to the next.

| Step | Name | Description | Document |
|------|------|-------------|----------|
| 1 | Reference Model Analysis | Analyze PyTorch model structure, operators, and shapes | `step-01-reference-model-analysis.md` |
| 2 | Operator Mapping | Map PyTorch operators to TTNN equivalents | `step-02-operator-mapping.md` |
| 3 | Per-Operator Testing | Test each TTNN operator with PCC validation | `step-03-per-operator-testing.md` |
| 4 | Module Implementation | Implement TTNN modules from validated operators | `step-04-module-implementation.md` |
| 5 | Per-Module Testing | Test each module against PyTorch reference | `step-05-per-module-testing.md` |
| 6 | E2E Model Implementation | Compose modules into complete model | `step-06-e2e-model-implementation.md` |
| 7 | E2E Model Testing | Validate full model with comprehensive testing | `step-07-e2e-model-testing.md` |

### Progress Checklist

Copy this checklist to track your progress:

```
Model Bringup Progress:
- [ ] Environment: [ ] Out-of-tree (pip install) / [ ] In-tree (source build)
- [ ] Step 1: Reference model analyzed (operators listed, shapes documented)
- [ ] Step 2: Operators mapped to TTNN equivalents
- [ ] Step 3: Per-operator PCC validated (≥0.999)
- [ ] Step 4: Modules implemented in TTNN
- [ ] Step 5: Per-module PCC validated (≥0.99)
- [ ] Step 6: E2E model composed from modules
- [ ] Step 7: E2E model validated (accuracy + performance)
```

## Instructions for Claude

When helping with model bringup:

1. **Ask about execution environment first** - Out-of-tree (pip install) or In-tree (source build)?
2. **Ask which step the user is on** or determine from context
3. **Read the corresponding step file** to get detailed instructions
4. **Complete the step's checklist** before suggesting to move to the next step
5. **Output a step completion report** (see format below)
6. **Confirm with the user** before proceeding to the next step

### Step Completion Report

After completing each step, output a report in the following format:

```
## Step X Complete: [Step Name]

### Summary
- What was done in this step

### Outputs
- List of deliverables/artifacts created

### Issues (if any)
- Problems encountered and how they were resolved

### Ready for Next Step
- [ ] All checklist items completed
- Next: Step X+1 - [Next Step Name]
```

### Starting a New Model Bringup

```
User: I want to bring up [model name] on TTNN

Claude:
1. Ask about execution environment:
   "Which environment are you using?
   - Out-of-tree: pip install ttnn (standalone project)
   - In-tree: source build with ./build_metal.sh (adding to tt-metal/models)"
2. Determine model type (LLM or CNN)
3. Read the appropriate model-type guide:
   - LLM (transformers, language models): `llm-model-bringup.md`
   - CNN (image models): `cnn-model-bringup.md`
4. Read `step-01-reference-model-analysis.md`
5. Help user complete Step 1 tasks
6. When Step 1 checklist is complete, ask:
   "Step 1 is complete. Ready to proceed to Step 2: Operator Mapping?"
7. If yes, read `step-02-operator-mapping.md` and continue
```

### Resuming Model Bringup

```
User: I'm working on [model name], currently on operator testing

Claude:
1. Read `step-03-per-operator-testing.md`
2. Ask about current progress on the step
3. Help complete remaining tasks
4. Move to next step when ready
```

### Example Interactions

**Example 1: Starting bringup**
```
User: "I want to port BERT-base to TTNN"

Claude response:
1. Asks: "Which environment are you using?
   - Out-of-tree (pip install ttnn)
   - In-tree (source build, adding to tt-metal/models)"
2. User: "Out-of-tree, using pip install"
3. Identifies model type: LLM (transformer)
4. Reads llm-model-bringup.md for architecture specifics
5. Starts with Step 1: helps extract operators from BERT
   - Linear, LayerNorm, GELU, Softmax, matmul
6. Documents shapes: [batch, seq_len, hidden_dim]
```

**Example 2: Debugging PCC failure**
```
User: "My LayerNorm has PCC 0.95, expected 0.999"

Claude response:
1. Reads debugging-tools.md
2. Suggests enabling comparison mode
3. Checks: data type mismatch? eps value? input shape alignment?
4. Recommends testing with float32 to isolate numerical issues
```

**Example 3: Mapping unknown operator**
```
User: "How do I convert torch.einsum to TTNN?"

Claude response:
1. Reads ttnn-operator-mapping.md
2. Finds einsum → ttnn.einsum (direct mapping)
3. Notes: check equation string format compatibility
```

## Quick Reference

- **Key Concepts**: See [key-concepts.md](key-concepts.md) for tensor layouts, data types, memory configs, and PCC thresholds
- **Debugging Tools**: See [debugging-tools.md](debugging-tools.md) for comparison mode and PCC validation

## Files in This Skill

```
.claude/skills/porting-models-to-ttnn/
├── SKILL.md                              # This file (overview)
├── key-concepts.md                       # Tensor layouts, data types, memory configs, PCC
├── debugging-tools.md                    # Comparison mode, PCC validation
├── ttnn-operator-mapping.md              # PyTorch → TTNN operator mapping
│
├── # Model-Type Specific Guides
├── llm-model-bringup.md                  # LLM-specific bringup (Llama, BERT, etc.)
├── cnn-model-bringup.md                  # CNN-specific bringup (ResNet, YOLO, etc.)
│
├── # Step-by-Step Workflow
├── step-01-reference-model-analysis.md   # Step 1: Analyze reference model
├── step-02-operator-mapping.md           # Step 2: Map operators
├── step-03-per-operator-testing.md       # Step 3: Test operators
├── step-04-module-implementation.md      # Step 4: Implement modules
├── step-05-per-module-testing.md         # Step 5: Test modules
├── step-06-e2e-model-implementation.md   # Step 6: Implement full model
└── step-07-e2e-model-testing.md          # Step 7: Test full model
```

## Reference

### Documentation
- `models/docs/model_bring_up.md` - Official model bringup guide
- `docs/source/ttnn/ttnn/converting_torch_model_to_ttnn.rst` - Official conversion guide
- `docs/source/ttnn/ttnn/tensor.rst` - Tensor concepts, layouts, and data types
- `docs/source/ttnn/ttnn/api.rst` - Complete TTNN API reference

### Tech Reports
- `tech_reports/ttnn/TTNN-model-bringup.md` - Comprehensive model bringup guide
- `tech_reports/ttnn/comparison-mode.md` - Debugging with comparison mode
- `tech_reports/LLMs/llms.md` - LLMs in TT-NN
- `tech_reports/CNNs/cnn_optimizations.md` - CNN bringup and optimization

### Example Code
- `tests/ttnn/unit_tests/operations/` - Unit test examples for operators
- `models/demos/` - Example model implementations
- `models/tt_transformers/` - Reference implementation for transformer models (LLM)
- `models/demos/yolov4/` - YOLOv4 implementation example (CNN)
- `models/demos/resnet/` - ResNet implementation example (CNN)
