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
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TT_PROFILER_OP_SUPPORT_COUNT=400 \
  python -m tracy -p -r -n uniad_full -m pytest \
    models/experimental/uniad/tests/pcc/test_ttnn_uniad.py::test_uniad -svv
```

Key profiler-specific env vars:

| Env var | Effect |
|---|---|
| `TT_METAL_DEVICE_PROFILER=1` | Force device profiler on (usually auto when `python -m tracy` runs) |
| `TT_METAL_PROFILER_MID_RUN_DUMP=1` | Dump partial CSV mid-run; helpful when test crashes |
| `TT_PROFILER_OP_SUPPORT_COUNT=<N>` | Raise device-side op buffer; default ~200, raise to 400+ for big models |

## Critical: disable Metal Trace while profiling

The profiler and `ttnn.begin_trace_capture` / `ttnn.execute_trace`
do not coexist — combining them produces a fatal
`Event Synchronization` error. Always profile with trace **off**:

- For UniAD: `TT_UNIAD_TRACE_DISABLE=1`
- For other models: whatever env var or code path turns off trace
- If the model has no flag: temporarily patch out `begin_trace_capture`
  or use a non-trace test variant.

## Cold vs warm path

The first forward pass triggers JIT compilation of ops. For a useful
profile, *the cold pass should not be the only one*. Two patterns:

### Pattern A — single warm pass

```bash
python -m tracy -p -r -n model_warm -m pytest \
  models/.../test_model.py::test_model[warm_iter=1] -svv
```

(Whatever test parameter your project uses to add warm iterations.)

### Pattern B — capture the cold pass + ignore N ops

Cheaper if you can't easily configure warm iters. The first ~100 ops
in the CSV will have inflated `HOST DURATION` (JIT compile time).
Filter them out in analysis. See `analysis-recipes.md`.

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
together (see `analysis-recipes.md`).

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
- [ ] Meaningful `-n <run_name>`
- [ ] If model is large: per-module profile or raise `TT_PROFILER_OP_SUPPORT_COUNT`
- [ ] Confirm CSV row count matches expected op count
