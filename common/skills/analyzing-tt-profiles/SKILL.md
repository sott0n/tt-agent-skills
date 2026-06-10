---
name: analyzing-tt-profiles
description: "Analyzes Tenstorrent device-profiler output — `ops_perf_results*.csv` (Performance Reports), NoC event JSON (`noc_trace_*.json`), and Memory reports — regardless of which front-end captured them. Both `python -m tracy` (TT-Metal/TTNN) and `ttrt perf` (tt-forge / tt-mlir runtime) drive the *same* tt-metal Tracy stack and emit the *same* CSV, so the interpretation layer is shared. Covers CSV-column semantics, the `tt-perf-report` CLI, Python aggregation recipes, NoC event JSON parsing, and cross-report pitfalls, with a 2D capability matrix mapping each report to the optimization category it informs. Use when interpreting a profile CSV / NoC JSON, deciding which profile to read for a goal, or planning what to tune next. For how to *capture* a profile, see the project capture skill: `profiling-tt-metal` (TTNN) or `tt-forge-perf` (tt-forge). For worked end-to-end examples, see `tt-metal-perf-case-studies`."
---

# Analyzing Tenstorrent Profiles

The Tenstorrent device profiler produces a small set of report
artifacts. **This skill is the analysis layer — it is front-end
agnostic.** Whether the profile was captured by `python -m tracy`
(TTNN, see `profiling-tt-metal`) or by `ttrt perf` (tt-forge /
tt-mlir runtime, see `tt-forge-perf`), both drive the same tt-metal
Tracy stack (`capture-release` / `csvexport-release` /
`tracy.process_ops_logs`) and emit the **same** `ops_perf_results*.csv`.
The columns, the traps, and the recipes below are identical across
projects.

## Capture vs analysis — read this first

| Concern | Where it lives |
|---|---|
| Build with profiling enabled, run the workload, produce the CSV/JSON | **Project capture skill** — `profiling-tt-metal` (TTNN) or `tt-forge-perf` (tt-forge) |
| Interpret `ops_perf_results*.csv`, NoC JSON, find the bottleneck | **This skill** |

The one report that genuinely **diverges** by project is the **Memory
report**:
- TTNN captures memory via `ttnn.graph.full_graph_capture()` →
  `db.sqlite` (17 tables) — analyzed with SQL recipes that live in
  `profiling-tt-metal` (`memory-reports.md`, `memory-sqlite-recipes.md`).
- tt-forge captures memory via `ttrt run --memory` →
  `memory_results.json` (different schema) — see `tt-forge-perf`.

Because the two memory formats are unrelated, memory analysis stays in
the project skills. Everything else (Performance CSV, NoC JSON) is
common and lives here.

## Three mental models to keep in mind

### 1. Stack layer being optimized

The profiler exposes different layers of the stack. The exact layer
names differ by project, but the analysis maps the same way:

| Layer (TTNN) | Layer (tt-forge) | Bottleneck examples |
|---|---|---|
| **ttnn** (operator / API) | **tt-mlir op / lowering** | dtype, layout, sharding choice, op-level config, `program_config` |
| **tt-metal kernel** | **tt-metal kernel** | dispatch bubbles between RISC-V cores, kernel implementation, circular-buffer sizing |
| **tt-fabric / distributed** | **tt-fabric / distributed** | inter-chip data movement, NoC congestion (also intra-chip NoC) |

Op-layer fixes are cheapest and most agent-friendly. Kernel-layer work
usually needs a kernel engineer. Fabric work is mostly for multi-chip.

### 2. Four categories of optimization opportunity

| Category | Levers |
|---|---|
| **Tensix dataflow** | sharding strategy, L1 vs DRAM placement, NoC routing |
| **Per-operator** | program_config, parallelization grid, kernel choice |
| **Data** | quantization (bf16 → bf8b), math fidelity (HiFi4 → HiFi2 → LoFi) |
| **Host-accelerator bubble** | Metal Trace, Multi-CQ, dispatch-core profiling |

### 3. Report types

| Report | Output | Agent-readable? |
|---|---|---|
| **Performance Reports** | `ops_perf_results*.csv` | ✅ CSV + `tt-perf-report` |
| **NoC Reports** (experimental) | JSON events + optional tt-npe | ✅ JSON readable |
| **Memory Reports** | TTNN: `db.sqlite` · tt-forge: `memory_results.json` | ✅ (project-specific — see capture skill) |
| **Tracy GUI / RISC-V mode** (future track) | `.tracy` binary | ❌ GUI handoff |

## 2D capability matrix — which report informs which optimization

| Optimization category \ Report | Performance (CSV) | NoC (JSON, experimental) | Memory (project-specific) | Tracy GUI (future) |
|---|---|---|---|---|
| **Tensix dataflow** (NoC, L1/DRAM placement) | △ CB wait / reserve columns | ◎ link congestion, per-link traffic | ◎ buffer placement, L1 peak, shard topology | — |
| **Per-operator** (program_config, sharding) | ◎ `CORE COUNT`, kernel µs, `MATH FIDELITY` | △ per-op NoC events | ○ per-op tensor shape / memory_config | — |
| **Data** (quantization, math fidelity) | ◎ `MATH FIDELITY` column, kernel µs vs fidelity | — | ◎ tensor dtype, layout, size inventory | — |
| **Host-accelerator bubble** | ○ `HOST DURATION` per op | — | — | ◎ RISC-V dispatch bubbles |

Legend: ◎ primary source · ○ secondary · △ partial · — not informative.

The matrix is the load-bearing decision tool. When a user says "I want
to optimize X", trace down the corresponding row to find the report to
read.

## Files in this skill

| Topic | Document |
|-------|----------|
| Interpreting `ops_perf_results*.csv` columns | `csv-columns.md` |
| First-pass analysis with `tt-perf-report` CLI | `tt-perf-report.md` |
| Python recipes (analyses `tt-perf-report` doesn't cover) | `analysis-recipes.md` |
| NoC event JSON schema + recipes (experimental) | `noc-reports.md` |
| Cross-report pitfalls | `pitfalls.md` |
| Cheatsheet | `quick-reference.md` |
| Known gaps / future recipes | `BACKLOG.md` |

## When to use what

| Goal | Start at |
|---|---|
| Find slow op + tune program_config | `tt-perf-report.md` |
| Slice slow op type by `CORE COUNT` | `analysis-recipes.md` Recipe 2 |
| Inspect worst-N matmul `ATTRIBUTES` | `analysis-recipes.md` Recipe 3 |
| Classify compute-bound vs host-bound | `csv-columns.md` "Wall time vs kernel sum" |
| NoC congestion (multi-chip / heavy data movement) | `noc-reports.md` (experimental) |
| Find peak L1 / quantization scan (memory) | project capture skill (TTNN: `profiling-tt-metal`; forge: `tt-forge-perf`) |

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
math fidelity). If `sum(HOST DURATION)` is comparable or larger, the
lever is Metal Trace or Multi-CQ. See `csv-columns.md` for the full
breakdown and `pitfalls.md` for the OP-TO-OP trap.

## Related skills

- `profiling-tt-metal` — **how to capture** a profile from TTNN
  (`python -m tracy`), Tracy build setup, and TTNN memory reports
  (`full_graph_capture` → SQLite).
- `tt-forge-perf` — **how to capture** a profile from tt-forge
  (`ttrt perf`, `ttrt run --memory`).
- `tt-metal-perf-case-studies` — worked end-to-end examples that
  demonstrate this analysis workflow (e.g. the UniAD DETR-decoder
  8-core matmul case).
- `optimizing-ttnn-models` — once a bottleneck is identified, how to
  fix it (data formats / sharding / Metal Trace / Multi-CQ / Conv2d).
