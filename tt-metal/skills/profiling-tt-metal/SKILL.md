---
name: profiling-tt-metal
description: "Captures TTNN/TT-Metal profiles on Tenstorrent hardware: Performance Reports via `python -m tracy -p -r`, NoC Reports via `python -m tracy --collect-noc-traces` (experimental), and Memory Reports via `ttnn.graph.full_graph_capture` + `python -m ttnn.graph_report` → SQLite. Covers Tracy-enabled build setup, running `python -m tracy` (env vars, warm/cold path, per-module profiling), and the TTNN-specific Memory Report SQLite schema + SQL recipes. Use when you need to *produce* a profile from a TTNN workload. To *interpret* the resulting `ops_perf_results_*.csv` or NoC JSON, see the front-end-agnostic `analyzing-tt-profiles` skill. For worked end-to-end examples, see `tt-metal-perf-case-studies`."
---

# Profiling TT-Metal / TTNN Models (capture)

This skill covers **how to capture** a profile from a TTNN / TT-Metal
workload. The analysis of the resulting artifacts (`ops_perf_results_*.csv`
columns, `tt-perf-report`, Python recipes, NoC JSON parsing, pitfalls)
is **front-end agnostic** and lives in the **`analyzing-tt-profiles`**
common skill — the same CSV is produced whether you capture via
`python -m tracy` (here) or `ttrt perf` (tt-forge), so the
interpretation layer is shared.

| You want to… | Go to |
|---|---|
| Build with Tracy, run a TTNN workload, get the CSV/JSON | **this skill** |
| Interpret the CSV / NoC JSON, find the bottleneck | **`analyzing-tt-profiles`** |
| Capture from tt-forge instead of TTNN | `tt-forge-perf` |

## Report types this skill captures

| Report | Capture | Output |
|---|---|---|
| **Performance Reports** | `python -m tracy -p -r -m pytest ...` | `ops_perf_results_*.csv` |
| **NoC Reports** (experimental) | `python -m tracy --collect-noc-traces ...` | `noc_trace_*.json` |
| **Memory Reports** | `with ttnn.graph.full_graph_capture(...):` + `python -m ttnn.graph_report` | `db.sqlite` (17 tables) |
| **Tracy RISC-V mode** (future) | `python -m tracy -p ...` then open `.tracy` in GUI | `.tracy` binary (GUI handoff) |

For the optimization-category → report mapping (the 2D capability
matrix) and the mental models, see `analyzing-tt-profiles/SKILL.md`.

## Capture tracks

### Common foundation

| Topic | Document |
|-------|----------|
| Tracy-enabled build setup (one-time) | `tracy-build-setup.md` |
| Running `python -m tracy` (env vars, warm/cold, per-module) | `running-tracy.md` |

### Performance Reports

Capture with `running-tracy.md`, then analyze the CSV with the
`analyzing-tt-profiles` skill (`csv-columns.md`, `tt-perf-report.md`,
`analysis-recipes.md`).

### NoC Reports — experimental

Add `--collect-noc-traces` to the `python -m tracy` invocation (see
`running-tracy.md`). The JSON event schema and analysis recipes are in
`analyzing-tt-profiles/noc-reports.md`.

### Memory Reports — TTNN-specific

| Topic | Document |
|-------|----------|
| Capture (`full_graph_capture` → JSON → SQLite) and output structure | `memory-reports.md` |
| SQL recipes against `db.sqlite` | `memory-sqlite-recipes.md` |

Memory capture is the one report that is genuinely TTNN-specific: it
uses the `ttnn.graph.full_graph_capture()` Python context manager and
produces a `db.sqlite` with the ttnn graph-report schema. (tt-forge
captures memory differently — `ttrt run --memory` → `memory_results.json`
— so its memory analysis lives in `tt-forge-perf`, not here.)

### Tracy RISC-V mode — future

Capture the `.tracy` file (`python -m tracy -p ...` without `-r`) and
open in the Tracy GUI to see per-RISC-V dispatch timelines. Used for
Multi-CQ bubble debugging. GUI-driven; agent recommends the capture
but hands the analysis to a human.

## Quick capture cheatsheet

```bash
# Performance report (warm path, trace disabled)
python -m tracy -p -r -n <run_name> -m pytest <test_path>::<test_func> -svv

# NoC report (experimental) — keep forward_passes ≤ 1
python -m tracy --collect-noc-traces --op-support-count 10000 -p -r \
  -n <run_name> -m pytest <test_path>::<test_func> -svv
```

```python
# Memory report — step 1: capture in your script/test
import ttnn
with ttnn.graph.full_graph_capture("/tmp/report.json", slow_dispatch=True):
    out = model(inputs)
```
```bash
# Memory report — step 2: import to SQLite
python -m ttnn.graph_report /tmp/report.json /tmp/db_dir/
```

See `running-tracy.md` for env vars, warm/cold path, and per-module
profiling; `tracy-build-setup.md` for the one-time build; and
`memory-reports.md` for the full memory workflow.

## Capture-time pitfalls (TTNN)

- **Profiler + Metal Trace = fatal.** Disable trace before profiling
  (`ttnn.begin_trace_capture` / `execute_trace` crash with
  `Event Synchronization`). See `running-tracy.md`.
- **DRAM ring buffer overflow drops ops silently** on high-frequency
  workloads — raise `TT_PROFILER_OP_SUPPORT_COUNT` or profile per
  module.
- **Legacy `TTNN_CONFIG_OVERRIDES` memory path produces no SQLite** —
  use `full_graph_capture`. See `memory-reports.md`.
- **`full_graph_capture` autouse pytest fixture segfaults** — inline
  the context manager at a call site. See `memory-reports.md`.

For analysis-time pitfalls (OP-TO-OP trap, cold JIT, multi-chip,
under-parallelization), see `analyzing-tt-profiles/pitfalls.md`.

## Related skills

- `analyzing-tt-profiles` — **how to interpret** the captured CSV / NoC
  JSON (front-end agnostic).
- `tt-forge-perf` — capture from tt-forge (`ttrt perf`).
- `tt-metal-perf-case-studies` — worked end-to-end examples.
- `optimizing-ttnn-models` — how to fix a bottleneck once identified.
- Tech reports: `tech_reports/MetalProfiler/metal-profiler.md`
  (profiler internals); `ttnn/tutorials/ttnn_visualizer.md` (Memory
  Reports web UI workflow).
