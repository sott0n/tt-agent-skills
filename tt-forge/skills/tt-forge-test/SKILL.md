---
name: tt-forge-test
description: Run tests and validate correctness for tt-forge changes. Use after code modifications to verify PCC/atol accuracy.
---

# TT-Forge Test Skill

## When to Use

After code changes to verify correctness (PCC/atol), or as the validation
step in the optimization loop (after tt-forge-optimize).

## Test Commands

### PyTorch Model Tests

```bash
# Single model test
pytest -svv tests/torch/models/<model_name>/test_<model_name>.py

# With PCC threshold
pytest -svv tests/torch/models/<model_name>/test_<model_name>.py --pcc-threshold 0.99

# Memory tracking
pytest --log-memory tests/torch/models/<model_name>/test_<model_name>.py
```

### ttrt Golden Check

```bash
# Run with golden validation
ttrt run out.ttnn

# Custom tolerance
ttrt run out.ttnn --rtol 1e-3 --atol 1e-3

# Save golden tensors for analysis
ttrt run out.ttnn --save-golden-tensors --save-artifacts
```

### tt-mlir Tests

```bash
# Run specific tt-mlir test
cd third_party/tt-mlir/src/tt-mlir
cmake --build build --target check-ttmlir
```

## Output Format

Report results in this format:

```
**Test Result**: PASS / FAIL
**PCC**: 0.9999 (threshold: 0.99)
**atol**: 1e-5 (threshold: 1e-4)
**Failed Tests**: (if any)
  - test_name: error message
```

## Checklist

```
- [ ] Identify test file for the change
- [ ] Run test with appropriate thresholds
- [ ] Report PASS/FAIL with metrics
- [ ] If FAIL, provide error details
```

## Common Issues

| Symptom | Cause | Action |
|---------|-------|--------|
| PCC < threshold | Accuracy regression | Report to user, may need fix |
| Test timeout | Performance regression | Report time taken |
| Import error | Missing dependency | Check environment |
