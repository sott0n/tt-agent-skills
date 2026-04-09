---
name: tt-forge-optimize
description: Implement performance optimizations in tt-mlir. Use after tt-forge-perf identifies bottlenecks and user wants to improve performance.
---

# TT-Forge Optimization Skill

## When to Use

- User wants to improve performance
- tt-forge-perf identified bottlenecks
- Implementing fusion, tiling, or other optimizations

**Not for**: Measuring performance (use tt-forge-perf)

## Optimization Workflow

```
- [ ] Step 1: Review perf report (bottlenecks)
- [ ] Step 2: Propose optimization approach to user
- [ ] Step 3: Read tt-mlir CLAUDE.md
- [ ] Step 4: Implement optimization
- [ ] Step 5: Run tt-forge-test
- [ ] Step 6: Run tt-forge-perf (re-measure)
```

## Step 1: Review Bottlenecks

From tt-forge-perf report, identify:
- Which ops are slow
- Whether memory-bound or compute-bound
- Potential optimization opportunities

## Step 2: Propose Approach

**Always ask user before implementing.** Present options:

```
Based on perf report, matmul_42 is the bottleneck (35% of time).

Optimization options:
A) Fuse with following relu (reduces memory traffic)
B) Adjust tiling for better L1 utilization
C) Use different compute kernel

Which approach should I try?
```

## Step 3: Read tt-mlir CLAUDE.md

**Required before modifying tt-mlir:**

```bash
cat third_party/tt-mlir/src/tt-mlir/CLAUDE.md
```

Contains:
- Build commands
- Test commands
- Code style guidelines

## Step 4: Implementation Locations

### Op Fusion (tt-xla)

Location: `python_package/tt_torch/backend/`

```python
# decompositions.py - Add decomposition
# passes.py - Add fusion pass
```

### TTIR/TTNN Passes (tt-mlir)

Location: `third_party/tt-mlir/src/tt-mlir/`

```
lib/Dialect/TTIR/Transforms/  # TTIR passes
lib/Dialect/TTNN/Transforms/  # TTNN passes
```

### Compute Kernels (tt-metal)

Location: `tt-metal` repository (separate)

## Common Optimizations

| Bottleneck | Optimization | Location |
|------------|--------------|----------|
| Memory traffic | Op fusion | tt-xla passes |
| Compute efficiency | Tiling | tt-mlir TTNN |
| Small ops overhead | Composite ops | tt-xla |
| Layout conversion | Memory layout opt | tt-mlir |

## Step 5: Test Changes

After implementation:

```bash
# tt-mlir tests
cd third_party/tt-mlir/src/tt-mlir
cmake --build build --target check-ttmlir

# Model test
pytest -svv tests/torch/models/<model>/test_<model>.py
```

Use **tt-forge-test Skill** for validation.

## Step 6: Re-measure

Use **tt-forge-perf Skill** to compare before/after.

## Integration

```
tt-forge-perf (identify bottleneck)
        │
        ▼
tt-forge-optimize (this skill)
        │
        ├─ Propose approach → User approves
        │
        ▼
Implement changes
        │
        ▼
tt-forge-test (validate correctness)
        │
        ▼
tt-forge-perf (measure improvement)
        │
        ▼
User: continue or done?
```

## Output Format

After implementation:

```
**Optimization**: <description>
**Files Changed**:
- path/to/file1.cpp
- path/to/file2.py

**Approach**: <what was done>

**Next**: Run tt-forge-test to validate, then tt-forge-perf to measure.
```
