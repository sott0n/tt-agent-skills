# First-pass Analysis with `tt-perf-report`

## Objective

Use Tenstorrent's official
[`tt-perf-report`](https://github.com/tenstorrent/tt-perf-report) CLI
as the **first-pass analyzer** for `ops_perf_results_*.csv`. It
covers per-op aggregates, bound classification (memory / compute / both),
signpost-based phase filtering, and op-id range slicing — without
writing any Python.

Reach for `analysis-recipes.md` (Python recipes) only when
`tt-perf-report` doesn't cover the analysis you need (per-core skew,
worst-N with ATTRIBUTES, under-parallelized triage, cross-module merge).

## Installation

```bash
pip install tt-perf-report
# or, from source:
git clone https://github.com/tenstorrent/tt-perf-report.git
cd tt-perf-report && pip install -e .
```

Verify:

```bash
tt-perf-report --help
```

## Basic usage

```bash
tt-perf-report ops_perf_results_<run>_<ts>.csv
```

Prints a color-coded per-op table to the terminal: kernel time, host
time, DRAM/compute utilization, and a memory-bound / compute-bound /
both classification per op. By default it also prints "advice" —
suggestions for which ops to look at first.

## Common invocations

### Suppress advice (raw table only)

```bash
tt-perf-report ops_perf_results_*.csv --no-advice
```

### Hide noise (sub-1% ops)

```bash
tt-perf-report ops_perf_results_*.csv --min-percentage 1.0
```

### Per-phase analysis with signposts

If the workload was instrumented with Tracy signposts (e.g.
`extract_img_feat`, `bev_encoder`, `detr_decoder`), filter to one
phase:

```bash
tt-perf-report ops_perf_results_*.csv \
  --start-signpost detr_decoder \
  --end-signpost detr_decoder_end
```

List available signposts in a trace:

```bash
tt-perf-report ops_perf_results_*.csv --print-signposts
```

### Inspect a specific op-id range

```bash
tt-perf-report ops_perf_results_*.csv --id-range 5-10
```

Useful when a profile-aggregate flags op #7 as slow and you want to
see ops #5-10 in context.

### Export filtered results to CSV

```bash
tt-perf-report ops_perf_results_*.csv \
  --start-signpost detr_decoder --end-signpost detr_decoder_end \
  --csv detr_decoder_phase.csv
```

The exported CSV can then be fed to Python recipes (see
`analysis-recipes.md`) for further slicing.

### Disable color (for log capture / CI)

```bash
tt-perf-report ops_perf_results_*.csv --no-color > perf_report.txt
```

## How to use signposts

`tt-perf-report` reads Tracy signposts embedded by the model code.
A signpost is created with `tracy.signpost(name)`:

```python
# In Python model code
import tracy
with tracy.scoped_signpost("detr_decoder"):
    out = detr_decoder(...)
```

Without signposts, the full trace is analyzed (use `--ignore-signposts`
to make that explicit). With signposts, `--start-signpost`/`--end-signpost`
lets you analyze any named span.

For an existing model that lacks signposts, add them as a low-cost
instrumentation step — typically a single decorator at each phase
boundary.

## When to use `tt-perf-report` vs Python recipes

| Need | Tool |
|---|---|
| Per-op aggregate, totals, % share | `tt-perf-report` |
| Memory-bound / compute-bound classification | `tt-perf-report` |
| Phase isolation via signposts | `tt-perf-report` |
| Op-id range slice | `tt-perf-report` |
| Multi-machine trace merging | `tt-perf-report` |
| Hide / threshold-filter sub-X% ops | `tt-perf-report` |
| Distribution by `CORE COUNT` for one op | Recipe 2 (`analysis-recipes.md`) |
| Worst-N individual instances + `ATTRIBUTES` | Recipe 3 (`analysis-recipes.md`) |
| Cross-module CSV merging | Recipe 4 (`analysis-recipes.md`) |
| Under-parallelized triage (slow + low cores) | Recipe 6 (`analysis-recipes.md`) |
| Per-core skew (PER CORE MIN / MAX) | Recipe 7 (`analysis-recipes.md`) |
| Custom one-off slicing | Python (`analysis-recipes.md`) |

The recipes in `analysis-recipes.md` are upstream-contribution candidates (see the
note at the top of that file). As they land in `tt-perf-report`, the
"Tool" column above shifts left.

## Reading the output

A typical row looks like:

```
ID | OP CODE                  | Device µs | Host µs | Cores | DRAM % | Compute % | Bound
 7 | MatmulDeviceOperation    |    5558.2 |    12.3 |    8  |   2.1  |    7.4    | both?
```

Interpretation:
- `Cores 8` — running on 8 of 130 cores. Look at `ATTRIBUTES` in the
  raw CSV to find the `program_config` that caused this (see
  `csv-columns.md`).
- `Bound: both?` — neither DRAM nor compute is saturated, which means
  the op is doing far less work than it could. Confirms the
  under-parallelization hypothesis.

When `tt-perf-report` flags an op like this, drop down to Recipe 2/3
in `analysis-recipes.md` for the detailed per-instance breakdown.

## Pipeline integration

`tt-perf-report` is designed for headless pipelines:

```bash
# Generate profile, then report, then export for next stage
python -m tracy -r -n my_run -m pytest tests/test_model.py
tt-perf-report generated/profiler/reports/my_run/*/ops_perf_results_*.csv \
  --no-color --no-advice --csv perf.csv > perf_report.txt
```

Combine with the recipes in `analysis-recipes.md` if downstream analysis is needed.

## Checklist

- [ ] `tt-perf-report` installed and on `PATH`
- [ ] Confirmed input CSV is the file from `python -m tracy -r ...`
  (not the per-device markers CSV)
- [ ] If the model has signposts, used `--start-signpost` /
  `--end-signpost` for phase isolation
- [ ] Used `--min-percentage` to hide noise on large traces
- [ ] Confirmed `tt-perf-report` doesn't cover the analysis before
  dropping to `analysis-recipes.md` (Python recipes)
