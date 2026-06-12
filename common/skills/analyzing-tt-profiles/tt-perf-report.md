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

### Aggregate the Stacked report (`--group-by`)

```bash
tt-perf-report ops_perf_results_*.csv --group-by op       # by op type (default-ish)
tt-perf-report ops_perf_results_*.csv --group-by memory   # by input-0 layout/placement
tt-perf-report ops_perf_results_*.csv --group-by category # compute / data-movement / tensor-manip
```

`--group-by op` is the go-to for "where does device time go." `--group-by
memory` buckets by where input 0 lives (L1 vs DRAM vs sharded) — useful
when you suspect layout/placement (not compute) is the cost.

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
The API is a **plain function call** `tracy.signpost(header, message=None)`
(verified in `tools/tracy/__init__.py`) — there is **no** context-manager
form like `scoped_signpost`. It emits a `TT_SIGNPOST` marker via
`ttnn.tracy_message` and is a no-op cost-wise when tracy isn't capturing:

```python
# In Python model code — mark each phase / forward boundary
from tracy import signpost
signpost("detr_decoder")          # start-of-phase marker
out = detr_decoder(...)
# multi-forward profiling: signpost(f"forward_{i+1}") at the top of each iter,
# then slice the warm one with --start-signpost forward_2
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
| Bound classification **for matmul/conv** | `tt-perf-report` |
| Bound classification for **non-matmul** ops (BinaryNg/Reshape/…) | raw columns (`csv-columns.md`) |
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

Two sections print: a **per-op table** (one row per op, columns
`ID | Total % | Bound | OP Code | Device | Device Time | Op-to-Op Gap |
Cores | DRAM | DRAM % | FLOPs | FLOPs % | Math Fidelity`) and a
**Stacked report** (the `--group-by` aggregation: per op-type
`Device Time Sum`, `Op Count`, category, FLOPs stats).

For ranking where device time goes, read the **Stacked report's
`Device Time Sum`** — it's pure device kernel time.

### Interpretation gotchas (verified against the tool source)

1. **The per-op table's `Total %` = `Device Time` + `Op-to-Op Gap`.**
   It blends in the profiler-inflated gap, so it over-ranks ops that sat
   behind a host stall. For a *pure device-cost* ranking use the Stacked
   report `Device Time Sum`, not `Total %`.
2. **`Bound` is matmul-only** (plus `HOST` for `(torch)` fallbacks).
   `BinaryNg` / `ReshapeView` / `Permute` etc. get a blank `Bound` and no
   advice — classify them from raw columns (`csv-columns.md` →
   "Classifying a NON-matmul op").
3. **The "High Op-to-Op Gap → tracing could save X µs" advice is
   overstated.** It's computed from the inflated `Op-to-Op Gap`; under
   the profiler that gap is not real wall time. Treat it as "these ops
   had host stalls," not as a wall-time savings estimate.
4. **The `Cores` coloring (red <10, green =64) is Wormhole-centric.** On
   Blackhole (130 worker cores) an op on 130 cores is fully parallel but
   isn't colored green, and "64" isn't special. Judge against
   `AVAILABLE WORKER CORE COUNT`, not the color.
5. The detail-section footer may print the **whole-file** op count even
   under `--id-range` (display quirk); the Stacked table and time sums
   *do* respect the filter — trust those.

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
