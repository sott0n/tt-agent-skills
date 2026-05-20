---
name: profiling-tt-metal
description: "Profiles TTNN/TT-Metal model performance, memory, and NoC traffic on Tenstorrent hardware. Covers the four profiling report types (Performance Reports via `python -m tracy`, Memory Reports via `ttnn.graph.full_graph_capture` + `python -m ttnn.graph_report`, NoC Reports [experimental], Tracy RISC-V mode), with a 2D capability matrix mapping each report to the optimization category it informs (Tensix dataflow, per-op, data quantization, host-accelerator bubble). Includes Tracy build setup, CSV-column semantics, `tt-perf-report` CLI, Python aggregation recipes, Memory Report SQLite schema and SQL recipes, NoC event JSON parsing, and pitfalls. Use when picking which profile to capture, when interpreting `ops_perf_results_*.csv` / `db.sqlite` / `noc_trace_*.json`, or when planning what to tune next. For worked end-to-end examples, see `tt-metal-perf-case-studies`."
---

# Profiling TT-Metal / TTNN Models

The TT-Metal profiler stack produces **four kinds of report**, each
exposing a different layer of the stack. This skill helps an agent
pick the right report for a given optimization goal and turn the
output into actionable findings.

## Three mental models to keep in mind

### 1. Stack layer being optimized

| Layer | Bottleneck examples |
|---|---|
| **ttnn** (operator / API) | `program_config`, dtype, layout, sharding choice, op-level config |
| **tt-metal kernel** | dispatch bubbles between RISC-V cores, kernel implementation, circular-buffer sizing |
| **tt-fabric / distributed** | inter-chip data movement, NoC congestion (also intra-chip NoC) |

ttnn-layer fixes are cheapest and most agent-friendly. Kernel-layer
work usually needs a kernel engineer. Fabric work is mostly for
multi-chip.

### 2. Four categories of optimization opportunity

| Category | Levers |
|---|---|
| **Tensix dataflow** | sharding strategy, L1 vs DRAM placement, NoC routing |
| **Per-operator** | program_config, parallelization grid, kernel choice |
| **Data** | quantization (bf16 → bf8b), math fidelity (HiFi4 → HiFi2 → LoFi) |
| **Host-accelerator bubble** | Metal Trace, Multi-CQ, dispatch-core profiling |

### 3. Four profiling report types

| Report | Capture | Output | Agent-readable? |
|---|---|---|---|
| **Performance Reports** | `python -m tracy -p -r -m pytest ...` | `ops_perf_results_*.csv` | ✅ CSV + `tt-perf-report` |
| **Memory Reports** | `with ttnn.graph.full_graph_capture(...):` + `python -m ttnn.graph_report` | `db.sqlite` (17 tables) | ✅ SQLite queryable |
| **NoC Reports** (experimental) | `python -m tracy --collect-noc-traces ...` | JSON events + optional tt-npe | ✅ JSON readable |
| **Tracy RISC-V mode** (future track) | `python -m tracy -p ...` then open `.tracy` in GUI | `.tracy` binary | ❌ GUI handoff |

## 2D capability matrix — which report informs which optimization

| Optimization category \ Report | Tracy RISC-V `-v` (future) | Performance (`-p -r`) | Memory (SQLite) | NoC (`--collect-noc-traces`, experimental) |
|---|---|---|---|---|
| **Tensix dataflow** (NoC, L1/DRAM placement) | — | △ CB wait / reserve columns | ◎ buffer placement, L1 peak, shard topology | ◎ link congestion, per-link traffic |
| **Per-operator** (program_config, sharding) | — | ◎ `CORE COUNT`, kernel µs, `MATH FIDELITY` | ○ per-op tensor shape / memory_config | △ per-op NoC events |
| **Data** (quantization, math fidelity) | — | ◎ `MATH FIDELITY` column, kernel µs vs fidelity | ◎ tensor dtype, layout, size inventory | — |
| **Host-accelerator bubble** | ◎ RISC-V dispatch bubbles (Tracy GUI) | ○ `HOST DURATION` per op | — | — |

Legend: ◎ primary source · ○ secondary · △ partial · — not informative.

The matrix is the load-bearing decision tool. When a user says "I
want to optimize X", trace down the corresponding row to find the
report to capture.

## Workflow

The skill is organized in four **tracks**, one per report type. Files
are named by topic — there is **no global execution order**. Start
from the matrix above to pick a track, then read the files in that
track.

### Common foundation (used by all tracks)

| Topic | Document |
|-------|----------|
| Tracy-enabled build setup (one-time) | `tracy-build-setup.md` |
| Running `python -m tracy` | `running-tracy.md` |
| Interpreting `ops_perf_results_*.csv` columns | `csv-columns.md` |
| Pitfalls (cross-report) | `pitfalls.md` |

### Track A: Performance Reports

| Topic | Document |
|-------|----------|
| First-pass analysis with `tt-perf-report` CLI | `tt-perf-report.md` |
| Python recipes (analyses `tt-perf-report` doesn't cover) | `analysis-recipes.md` |

### Track B: Memory Reports

| Topic | Document |
|-------|----------|
| Capture (TTNN_CONFIG_OVERRIDES) and output structure | `memory-reports.md` |
| SQL recipes against `db.sqlite` | `memory-sqlite-recipes.md` |

### Track C: NoC Reports — **experimental**

| Topic | Document |
|-------|----------|
| Capture (`--collect-noc-traces`), JSON event schema, recipes | `noc-reports.md` |

The reporting path and JSON schema are still evolving. Use for NoC
traffic / link congestion / multicast efficiency analysis when other
tracks point at data movement rather than compute or memory layout
as the bottleneck.

### Track D: Tracy RISC-V mode — **future iteration**

Capture the `.tracy` file (`python -m tracy -p ...` without `-r`)
and open in the Tracy GUI to see per-RISC-V dispatch timelines.
Used for Multi-CQ bubble debugging. GUI-driven; agent recommends
the capture but hands the analysis to a human.

Cheatsheet: `quick-reference.md`. Worked end-to-end examples:
`tt-metal-perf-case-studies` skill. Known gaps / future recipes:
`BACKLOG.md`.

## When to use what

| Goal | Track | Start at |
|---|---|---|
| Find slow op + tune program_config | A | `tt-perf-report.md` |
| Slice slow op type by `CORE COUNT` | A | `analysis-recipes.md` Recipe 2 |
| Inspect worst-N matmul `ATTRIBUTES` | A | `analysis-recipes.md` Recipe 3 |
| Find peak L1 per op (OOM scan) | B | `memory-sqlite-recipes.md` Recipe 1 |
| Audit DRAM vs L1 placement | B | `memory-sqlite-recipes.md` Recipe 2 |
| Quantization opportunity scan (find big bf16 tensors) | B | `memory-sqlite-recipes.md` Recipe 3 + 5 |
| Sharding load-balance check (per-core) | B | `memory-sqlite-recipes.md` Recipe 6 |
| NoC congestion (multi-chip / heavy data movement) | C | `noc-reports.md` (experimental) |
| Multi-CQ dispatch bubble debug | D | (future, GUI) |

## Mental model for time-vs-memory profiling

- **Performance Reports** answer "where does time go?" — kernel µs
  per op, host dispatch, core utilization.
- **Memory Reports** answer "where does space go?" — buffer
  placement, tensor dtype/layout, peak L1, per-core load.
- They are independent captures. Performance uses `python -m tracy`
  (env vars + CLI flags). Memory uses a Python context manager
  (`with ttnn.graph.full_graph_capture(...)`) — no env vars, no
  tracy. They do not conflict, but separate runs remain cleaner for
  clarity.

## Mental model for interpreting Performance Reports CSV

Two distinct measurements to keep straight:

- **`DEVICE KERNEL DURATION [ns]`** — the time the op's kernel actually
  ran on cores. This is the real device-side cost. Sum it to get total
  device work.
- **`HOST DURATION [ns]`** — the time spent dispatching from host. On a
  warm path (post-JIT), this is usually 5-15 µs per op.
- **`OP TO OP LATENCY [ns]`** — *device-side cycle gap* between the end
  of op N and the start of op N+1. **Heavily inflated by the profiler
  instrumentation itself**; do not interpret as host overhead.

If `sum(DEVICE KERNEL DURATION)` is close to wall-clock, the model is
compute-bound and the lever is per-op tuning (program_config, sharding,
math fidelity → see `optimizing-ttnn-models` Step 1/3/6). If
`sum(HOST DURATION)` is comparable or larger, the lever is Metal Trace
or Multi-CQ (`optimizing-ttnn-models` Step 4/5). See
`csv-columns.md` for the full breakdown and
`pitfalls.md` for the OP-TO-OP trap.

## Related skills

- `tt-metal-perf-case-studies` — worked end-to-end examples that
  demonstrate this workflow (e.g. the UniAD DETR-decoder 8-core
  matmul case).
- `optimizing-ttnn-models` — once a bottleneck is identified, that
  skill covers how to fix it: data formats / sharding / Metal Trace /
  Multi-CQ / Conv2d tuning / multi-device.
- Tech reports: `tech_reports/MetalProfiler/metal-profiler.md`
  (profiler internals), `tech_reports/AdvancedPerformanceOptimizationsForModels/`
  (end-to-end example).
- `ttnn/tutorials/ttnn_visualizer.md` — the official Memory Reports
  workflow with the web UI (human-side).
