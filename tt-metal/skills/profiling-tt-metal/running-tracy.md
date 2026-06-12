# Running `python -m tracy`

## Objective

Capture an op-level profile of a TTNN workload and produce
`ops_perf_results_<run_name>_<timestamp>.csv` for downstream analysis.

## Basic invocation (Performance Reports — Track A)

```bash
python -m tracy -p -r -n <run_name> -m pytest <test_path>::<test_func> -svv
```

Flags (verified against `tools/tracy/__main__.py`):
- `-p` — "Only profile enabled zones." Reduces instrumentation
  overhead by limiting capture to zones explicitly enabled in code.
  Standard for warm-path performance reports.
- `-r` — "Generate ops report." Post-processes the `.tracy` capture
  to produce `ops_perf_results_<run>_<ts>.csv`. Without `-r`, only the
  raw `.tracy` file is written and `csvexport-release` must be run
  manually.
- `-n <run_name>` — "Custom name to be added to report name." Labels
  the output directory. Use a meaningful name (e.g. `uniad_decoder`,
  `resnet50_bs1_phaseB`). The output lands at
  `generated/profiler/reports/<run_name>/<timestamp>/`. **Optional**;
  if omitted, the directory uses the timestamp alone.
- `-m pytest ...` — "Profile a library module." Run pytest as the
  profiled module. Without `-m`, `python -m tracy` treats the next
  argument as a script path.
- `-v` — "More info is printed to stdout." Verbose mode, not related
  to RISC-V visualization. Optional debugging aid.

For the **NoC Reports** track, add `--collect-noc-traces` (future Track C
in this skill). For the **Tracy GUI / RISC-V mode** track (future Track D),
omit `-r` to keep the `.tracy` file for GUI inspection rather than
CSV post-processing. For **Memory Reports** (Track B), do *not* use
`python -m tracy` at all — see `memory-reports.md`.

## Profile a standalone script

```bash
python -m tracy -p -r -n my_bench path/to/bench_script.py
```

The script runs unchanged; markers are injected by the linked
Tracy-instrumented `.so`.

## Output location

```
generated/profiler/reports/<run_name>/<timestamp>/
├── ops_perf_results_<run_name>_<timestamp>.csv  ← main artifact
├── profile_log_device.csv                        ← raw device markers
└── tracy_profile_log_host.tracy                  ← Tracy GUI binary
```

## Combining with workload env vars

Workload env vars (`TT_METAL_*`, model-specific like `TT_UNIAD_*`) are
passed through unchanged:

```bash
TT_DCN_DEVICE=1 TT_UNIAD_TRACE_DISABLE=1 TT_UNIAD_WARM_ITERS=0 \
TT_METAL_DEVICE_PROFILER=1 \
  python -m tracy -p -r --op-support-count 8000 -n uniad_full -m pytest \
    models/experimental/uniad/tests/pcc/test_ttnn_uniad.py::test_uniad -svv
```

Key profiler-specific env vars:

| Env var | Effect |
|---|---|
| `TT_METAL_DEVICE_PROFILER=1` | Force device profiler on (usually auto when `python -m tracy` runs) |
| `--op-support-count <N>` (tracy **flag**, not env) | Sizes the per-program DRAM profiler buffer (`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`, **default 1000**). Set **N > total programs** = device-ops × forwards (e.g. 8000 for a ~5200-op run); too small → dropped markers → `-r` join crash. Changing it forces a one-time kernel recompile. ⚠️ The env var `TT_PROFILER_OP_SUPPORT_COUNT` is **read by nothing** — use the flag. |
| `TT_METAL_PROFILER_MID_RUN_DUMP=1` | Dump partial CSV mid-run; helpful when a test crashes. **Not** a substitute for `--op-support-count` — size the buffer instead. |

## Critical: disable Metal Trace while profiling

The profiler and `ttnn.begin_trace_capture` / `ttnn.execute_trace`
do not coexist — combining them produces a fatal
`Event Synchronization` error. Always profile with trace **off**:

- For UniAD: `TT_UNIAD_TRACE_DISABLE=1`
- For other models: whatever env var or code path turns off trace
- If the model has no flag: temporarily patch out `begin_trace_capture`
  or use a non-trace test variant.

## Cold vs warm path

The first forward pass triggers JIT compilation of ops. **Budget for it
when planning:** the first profiled run pays full JIT (can be minutes —
e.g. 312 s call vs 23 s once kernels are disk-cached). Two consequences:

- **Pre-warm for a clean warm capture.** Run the workload **once without
  the profiler** first (populates the on-disk kernel cache), *then* the
  profiled run is all-warm — cleaner than padding with extra warm iters
  (which only bloats the logs). If you skip the pre-warm, the first
  forward in the capture is cold and must be split off (below).
- For a useful profile, *the cold pass should not be the only one*.

Two patterns:

### Pattern A — single warm pass

```bash
python -m tracy -p -r -n model_warm -m pytest \
  models/.../test_model.py::test_model[warm_iter=1] -svv
```

(Whatever test parameter your project uses to add warm iterations.)

### Pattern B — capture the cold pass + ignore N ops

Cheaper if you can't easily configure warm iters. The first ~100 ops
in the CSV will have inflated `HOST DURATION` (JIT compile time).
Filter them out in analysis. See the `analyzing-tt-profiles` skill
(`analysis-recipes.md`).

## Splitting forwards in the CSV

When you run N forwards (cold + warm iters) the CSV holds **all of them
concatenated** — isolate the warm one or your totals double-count. There
is no boundary marker (`GLOBAL CALL COUNT` is a smooth per-op counter).
The boundary is the **single largest `OP TO OP LATENCY` gap** (the
inter-forward host stall: input deepcopy + re-upload), much larger than
any intra-forward gap on a warm run. Then slice with
`tt-perf-report <csv> --id-range <warmStartID>-` and confirm the per-op
counts sum to the warm op count. (`OP TO OP LATENCY` is for finding the
boundary only — under the profiler it's host-inflated, not a wall-time
metric; trust `DEVICE KERNEL DURATION`.) The cold forward has *more* ops
(one-time setup: conv-weight prep, ref points, lidar2img), so the warm
forward is the shorter tail.

## If `-r` crashes with `Device data missing: Op <N>`

Your `--op-support-count` was too small (default 1000), so the device
profiler dropped markers and the host↔device join can't match them. **Fix
it at the source: re-run with a larger `--op-support-count`** (see the env
table above) — don't try to salvage the truncated logs. To confirm a
clean capture afterward, check the CSV op-row count equals
`grep -c TT_DNN_DEVICE_OP generated/profiler/.logs/tracy_ops_data.csv`.

## Per-module profiling for large models

When the full model is too big (DRAM overflow drops 50%+ of ops),
profile each submodule separately by writing a per-module pytest:

```python
# models/experimental/<model>/tests/test_ttnn_<module>.py
def test_uniad_decoder(device, ...):
    model = TtDetrTransformerDecoder(...)
    out = model(inputs)
    # PCC check optional
```

Then:

```bash
python -m tracy -p -r -n <model>_<module> -m pytest \
  models/.../test_ttnn_<module>.py::test_<module> -svv
```

The captures land in separate directories and can be aggregated
together (see the `analyzing-tt-profiles` skill, `analysis-recipes.md`).

## Output sanity check

After a successful run, the last few lines of stdout look like:

```
INFO | tracy:generate_report:154 - Host side ops time report generated at ...
INFO | tracy.process_ops_logs:append_device_data:919 - Appending device data
INFO | tracy.process_ops_logs:generate_reports:1663 -
  OPs csv generated at: generated/profiler/reports/<run_name>/<ts>/ops_perf_results_<run_name>_<ts>.csv
```

If the run crashed early but Tracy still wrote a CSV, it will contain
only the ops up to the crash. Check `wc -l <csv>` — a "full" UniAD
forward is ~3000-4000 rows; a partial capture of 200-400 rows means
the test died early.

## Quick checks after a run

```bash
CSV=$(ls -t generated/profiler/reports/<run_name>/*/ops_perf_results_*.csv | head -1)
wc -l "$CSV"                       # op count
head -2 "$CSV" | tail -1           # first data row, sanity
awk -F, 'NR>1{s+=$17} END{print s/1e6, "ms total kernel"}' "$CSV"
```

(Column 17 is `DEVICE KERNEL DURATION [ns]` in the current header
order. Verify with `head -1 $CSV | tr ',' '\n' | nl | grep KERNEL`.)

## Checklist

- [ ] `tt-smi -r <device>` before each run
- [ ] Trace disabled in the workload
- [ ] `--op-support-count N` set with N > total programs (default 1000 drops ops on big models)
- [ ] (Optional) pre-warm once without the profiler for an all-warm capture
- [ ] Meaningful `-n <run_name>`
- [ ] **Verify completeness:** CSV op-row count == `grep -c TT_DNN_DEVICE_OP .logs/tracy_ops_data.csv` (short = dropped ops → raise `--op-support-count`)
- [ ] Multi-forward capture: split on the largest OP-TO-OP gap, slice with `tt-perf-report --id-range`
