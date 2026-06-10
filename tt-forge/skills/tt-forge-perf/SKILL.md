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

`ttrt perf` drives the **same tt-metal Tracy stack** as TTNN's
`python -m tracy` (it runs `capture-release` / `csvexport-release` and
`tracy.process_ops_logs` under the hood), so it emits the **identical**
`ops_perf_results.csv` schema — not a simplified one. Interpret it with
the front-end-agnostic **`analyzing-tt-profiles`** skill.

Key columns (full breakdown in `analyzing-tt-profiles/csv-columns.md`):
- `OP CODE` — operation name (e.g. `MatmulDeviceOperation`)
- `DEVICE KERNEL DURATION [ns]` — real device-side time (sum for total work)
- `HOST DURATION [ns]` — host dispatch time
- `CORE COUNT` — cores the op used (low + slow = under-parallelized)
- `MATH FIDELITY` — LoFi / HiFi2 / HiFi3 / HiFi4
- `ATTRIBUTES` — op-specific shape/config string
- ⚠ `OP TO OP LATENCY [ns]` — **inflated by profiler; not host overhead**

First-pass analysis is easiest with the `tt-perf-report` CLI:

```bash
pip install tt-perf-report
tt-perf-report ops_perf_results.csv --min-percentage 1.0
```

See `analyzing-tt-profiles/tt-perf-report.md` and `analysis-recipes.md`
for deeper slicing (core-count distribution, worst-N, per-core skew).

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

This writes `memory_results.json` per program (via the runtime
callback `save_memory_report`). Note this is a **different format** from
TTNN's `full_graph_capture` → `db.sqlite` — the SQLite SQL recipes in
`profiling-tt-metal` do **not** apply to tt-forge memory reports.

Memory report shows:
- DRAM/L1 allocation per op
- Bytes per bank
- Free/allocated memory

Granularity is controlled by `--memory-log-level`
(`none` / `program` / `operation` / `any`).

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

## Related Skills

- **`analyzing-tt-profiles`** (common) — how to interpret
  `ops_perf_results.csv` and NoC JSON: column semantics, the
  `tt-perf-report` CLI, Python aggregation recipes, and analysis
  pitfalls (OP-TO-OP trap, cold JIT, under-parallelization). The CSV
  `ttrt perf` produces is the same format analyzed there.
- `tt-forge-optimize` — implement optimizations once a bottleneck is
  identified.
- `recovering-tt-hardware` — recover a wedged device after a bad run.
