---
name: ttnn-model-bringup
description: "Converts PyTorch models to TTNN for Tenstorrent hardware. Covers operator mapping, PCC validation, and module implementation. Use when porting PyTorch/HuggingFace models to TTNN, mapping torch operators to ttnn equivalents, debugging PCC failures, or implementing TTNN modules."
---

# TTNN Model Bringup

This skill provides a systematic, step-by-step approach to converting PyTorch models to run on Tenstorrent hardware using the TTNN library.

## When to Use This Skill

- When starting to port a new PyTorch model to TTNN
- When mapping PyTorch operators to their TTNN equivalents
- When implementing and testing individual TTNN operators
- When building module blocks from converted operators
- When validating model correctness using PCC (Pearson Correlation Coefficient)
- When debugging accuracy issues in TTNN model implementations

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

## Instructions for Claude

When helping with model bringup:

1. **Ask which step the user is on** or determine from context
2. **Read the corresponding step file** to get detailed instructions
3. **Complete the step's checklist** before suggesting to move to the next step
4. **Confirm with the user** before proceeding to the next step

### Starting a New Model Bringup

```
User: I want to bring up [model name] on TTNN

Claude:
1. Determine model type (LLM or CNN)
2. Read the appropriate model-type guide:
   - LLM (transformers, language models): `llm-model-bringup.md`
   - CNN (image models): `cnn-model-bringup.md`
3. Read `step-01-reference-model-analysis.md`
4. Help user complete Step 1 tasks
5. When Step 1 checklist is complete, ask:
   "Step 1 is complete. Ready to proceed to Step 2: Operator Mapping?"
6. If yes, read `step-02-operator-mapping.md` and continue
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

## Quick Reference

- **Key Concepts**: See [key-concepts.md](key-concepts.md) for tensor layouts, data types, memory configs, and PCC thresholds
- **Debugging Tools**: See [debugging-tools.md](debugging-tools.md) for comparison mode and PCC validation

## Files in This Skill

```
.claude/skills/ttnn-model-bringup/
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
