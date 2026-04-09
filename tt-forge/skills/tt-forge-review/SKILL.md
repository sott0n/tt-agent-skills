---
name: tt-forge-review
description: Review code changes for quality and consistency. Use after Test Skill passes, before committing changes.
---

# TT-Forge Code Review Skill

## When to Use

- After Test Skill reports PASS
- Before committing changes
- Part of optimization loop (after Test passes)

## Review Checklist

```
- [ ] Code correctness (logic, edge cases)
- [ ] Code style (matches existing patterns)
- [ ] No unnecessary changes (minimal diff)
- [ ] No debug artifacts (print statements, commented code)
- [ ] Documentation updated (if API changed)
```

## Review by Component

### tt-xla Changes

Check:
- PJRT interface consistency
- Error handling patterns
- Logging conventions (use TTXLA_LOG)

### tt-mlir Changes

Check:
- MLIR patterns (TableGen, passes)
- tt-mlir coding style (see `third_party/tt-mlir/src/tt-mlir/CLAUDE.md`)
- Test coverage for new ops/passes

### Model/Test Changes

Check:
- Uses `ModelLoader` when model exists in tt-forge-models
- Proper test markers (`@pytest.mark.push`, etc.)
- PCC/atol thresholds are reasonable

## Output Format

```
**Review Result**: APPROVED / CHANGES_REQUESTED

**Summary**: [1-2 sentence summary]

**Issues** (if any):
1. [file:line] Issue description
   Suggestion: ...

**Approved Changes**:
- [list of changes that look good]
```

## Review Criteria

| Aspect | Check |
|--------|-------|
| Correctness | Logic is sound, handles edge cases |
| Minimal | Only necessary changes, no scope creep |
| Consistent | Follows existing patterns in codebase |
| Clean | No debug code, no commented-out code |
| Tested | Changes covered by Test Skill |
