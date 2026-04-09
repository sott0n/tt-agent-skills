---
name: tt-forge-perf
description: Measure model performance on Tenstorrent hardware. Use when user asks about performance, latency, throughput, or wants to identify bottlenecks.
---

# TT-Forge Performance Measurement Skill

## When to Use

- Measure model execution time
- Identify performance bottlenecks
- Compare before/after optimization
- Profile op-level timing

**Not for**: Implementing optimizations (use tt-forge-optimize)

## Measurement Workflow

```
- [ ] Step 1: Run baseline measurement
- [ ] Step 2: Identify bottleneck ops
- [ ] Step 3: Analyze memory usage
- [ ] Step 4: Report findings to user
```

## Step 1: Run Baseline

### ttrt perf (Recommended)

```bash
# Performance analysis with artifacts
ttrt perf out.ttnn --save-artifacts

# Host-only metrics (faster, no device profiling)
ttrt perf out.ttnn --host-only

# Output:
# - ops_perf_results.csv: per-op timing
# - tracy_profile_log_host.tracy: Tracy GUI file
```

### pytest with timing

```bash
# Run test with timing
time pytest -svv tests/torch/models/<model>/test_<model>.py

# Memory tracking
pytest --log-memory tests/torch/models/<model>/test_<model>.py
```

## Step 2: Identify Bottlenecks

### Read ops_perf_results.csv

```bash
# Sort by execution time
cat ops_perf_results.csv | sort -t',' -k3 -rn | head -10
```

Key columns:
- `op_name`: Operation name
- `execution_time_ms`: Time in milliseconds
- `percentage`: Percentage of total time

### Common Bottleneck Patterns

| Pattern | Symptom |
|---------|---------|
| Single slow op | One op > 30% of total |
| Memory bound | High DRAM bandwidth usage |
| Compute bound | High FPU utilization |
| Many small ops | Overhead from op dispatch |

## Step 3: Memory Analysis

```bash
# Memory usage per operation
ttrt run out.ttnn --memory --save-artifacts

# Check for memory leaks
ttrt run out.ttnn --memory --check-memory-leak --save-artifacts
```

Memory report shows:
- DRAM/L1 allocation per op
- Bytes per bank
- Free/allocated memory

## Step 4: Report Format

Report to user in this format:

```
**Model**: <model_name>
**Total Time**: X.XX ms

**Top 5 Bottlenecks**:
| Rank | Op | Time (ms) | % Total |
|------|-----|-----------|---------|
| 1 | matmul_42 | 5.2 | 35% |
| 2 | conv2d_12 | 2.1 | 14% |
| ... | ... | ... | ... |

**Memory**:
- DRAM Peak: X MB
- L1 Peak: X KB

**Observations**:
- [Key finding 1]
- [Key finding 2]
```

## Environment Variables

```bash
# Enable perf tracing (requires perf-enabled build)
export TTMLIR_ENABLE_PERF_TRACE=1

# tt-metal debug logging
export TT_METAL_LOGGER_LEVEL=DEBUG
```

## Perf-Enabled Build

Performance profiling requires special build:

```bash
cmake -G Ninja -B build -DTT_RUNTIME_ENABLE_PERF_TRACE=ON
cmake --build build
```

## Integration

```
tt-forge-perf (measure)
        │
        ▼ report bottlenecks
User decision: optimize?
        │
        ▼ yes
tt-forge-optimize (implement)
        │
        ▼
tt-forge-test (validate)
        │
        ▼
tt-forge-perf (re-measure)
```

## Comparison Mode

When comparing before/after:

```
**Comparison**: <change description>

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Total Time | 15.2ms | 10.1ms | -33% |
| matmul_42 | 5.2ms | 2.1ms | -60% |
| Memory Peak | 128MB | 128MB | 0% |

**Result**: [IMPROVED / REGRESSED / NO CHANGE]
```
