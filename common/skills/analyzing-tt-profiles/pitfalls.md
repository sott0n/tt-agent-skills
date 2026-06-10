# Pitfalls and Gotchas

## Objective

Avoid the common traps that produce misleading profile data or
silently truncate captures. These apply to **Performance Reports**
(`ops_perf_results*.csv`) regardless of how the profile was captured
(`python -m tracy` or `ttrt perf`).

For **Memory Report** pitfalls, see the project capture skill, since
the memory formats diverge: TTNN's `full_graph_capture` → SQLite
caveats live in `profiling-tt-metal` (`memory-reports.md`); tt-forge's
`ttrt run --memory` → `memory_results.json` caveats live in
`tt-forge-perf`.

## Pitfall 1: Profiler + Metal Trace = fatal

If `ttnn.begin_trace_capture` / `ttnn.execute_trace` is invoked while
the device profiler is active, the run crashes with an
`Event Synchronization` fatal error mid-run.

**Always disable trace before profiling.**

For models that auto-enable trace, find the toggle:

```python
# UniAD example
import os
if os.environ.get("TT_UNIAD_TRACE_DISABLE") == "1":
    use_trace = False
```

In the test harness:

```bash
TT_UNIAD_TRACE_DISABLE=1 python -m tracy -r -n <run> -m pytest ...
```

If no env toggle exists, patch out the trace call temporarily or write
a non-trace test variant.

## Pitfall 2: `OP TO OP LATENCY` looks like host overhead — it isn't

```
OP TO OP LATENCY [ns] = DEVICE FW START CYCLE[N+1] - DEVICE FW END CYCLE[N]
```

is the *device-side* gap between two ops, measured by device markers.
It is **inflated by the profiler's own ring-buffer writes**.

Common wrong conclusion: "the model spends 80% in OP TO OP latency, so
it's host-bound, let's add Metal Trace."

Reality:
- Real warm host dispatch: 5-15 µs/op (visible in `HOST DURATION`).
- The OP TO OP latency you see in profile is profiler overhead, not
  observable on a non-profiler build.

**Use `HOST DURATION` for dispatch cost. Use `DEVICE KERNEL DURATION`
for device work. Ignore `OP TO OP LATENCY` for bottleneck classification.**

See `tt_metal/impl/profiler/profiler_analysis.cpp:514-531` for the
exact formula.

## Pitfall 3: DRAM ring buffer overflow drops ops silently

The device profiler writes marker events to a fixed-size DRAM ring
buffer. For models with thousands of high-frequency ops (Conv2d-heavy
backbones, GridSample chunked ops), the buffer overflows and many ops
are not recorded.

Symptoms:
- Op count in CSV is much lower than expected
- Total kernel time is suspiciously small (e.g. 25 ms for a model
  with 1.5 s warm wall)
- A specific op type (often `Conv2dDeviceOperation`, `HaloDevice...`,
  `Pool2D`) appears at counts far below what's expected

Mitigations:

1. **Raise the buffer**:
   ```bash
   TT_PROFILER_OP_SUPPORT_COUNT=400 python -m tracy ...
   ```
   Default is around 200; bump to 400-800 for large models. Setting it
   too high uses more device DRAM — be aware of L1/DRAM budget.

2. **Mid-run dumps**:
   ```bash
   TT_METAL_PROFILER_MID_RUN_DUMP=1 python -m tracy ...
   ```
   Flushes buffer mid-run rather than only at end.

3. **Per-module profiling**: split the workload. Run each submodule's
   pytest separately and aggregate the CSVs (see `analysis-recipes.md`).

## Pitfall 4: First forward pass is cold (JIT compile)

The first time each unique op (shape × dtype × program_config
combination) is dispatched, host-side time includes JIT compile (often
100s of ms). This dominates the warm-cache behavior you actually care
about.

Avoid by running ≥1 warm iteration before measurement. Or filter cold
rows in analysis:

```python
warm = [r for r in rows if int(r["HOST DURATION [ns]"]) < 50_000]
```

(50 µs cutoff is generous; warm dispatch is usually < 30 µs.)

Verify by plotting `HOST DURATION` vs `GLOBAL CALL COUNT` and looking
for the cliff where it drops to single-digit µs.

## Pitfall 5: Branch / build version mismatch

If the build was made from one commit but you're running tests from
another, op APIs can mismatch. Examples seen in practice:

- `ttnn.grid_sample` removed `compute_kernel_config` parameter →
  `TypeError: incompatible function arguments` mid-run
- Kernel-side `compile_time_args.h` index out of range → JIT compile
  fails for an op that uses Tracy macros

Symptoms during profiling:
- Run aborts partway through with a Python `TypeError`
- A subset of CSV is still produced (everything before the crash)
- JIT compile error referencing `static assertion failed: Index out of range`

Fix:
- Match build to checked-out branch: rebuild from the branch HEAD, or
  check out a commit compatible with the existing build.
- For one-off API mismatches, patch the offending call in the model
  code to the new API.

## Pitfall 6: Reading totals without sanity checking op count

A common mistake: trust the totals from the CSV without checking the
op count. Examples:

- Full UniAD forward: ~3000-4000 ops. If CSV has 400, you only
  captured ~10%.
- Per-module BEV encoder: hundreds to ~1000 ops depending on layer count.

**Always print the op count first** and compare to your expectation
before drawing conclusions. If the count is too low, the cause is
usually pitfall 3 (DRAM overflow) or pitfall 5 (mid-run crash).

## Pitfall 7: Confusing `DEVICE FW DURATION` with `DEVICE KERNEL DURATION`

Two distinct numbers, both real:

- `DEVICE KERNEL DURATION [ns]` — kernel-only time
- `DEVICE FW DURATION [ns]` — kernel + firmware wrap (setup, teardown)

`DEVICE FW DURATION` is typically `DEVICE KERNEL DURATION + 1-2 µs`.
For bottleneck analysis, sum `DEVICE KERNEL DURATION` only — that's
the actual work. `DEVICE FW DURATION` is useful when investigating FW
overhead specifically.

## Pitfall 8: Mixing devices in a multi-chip profile

On multi-chip configs (n300, T3K, QuietBox, Galaxy), each device
records markers independently. The CSV may include multiple devices —
check `DEVICE ID` column. Summing across devices counts kernel time
twice for parallel work.

Filter to a single device for serial analysis:

```bash
awk -F, 'NR==1 || $4==0' ops_perf_results_*.csv > device0.csv
```

## Pitfall 9: `CORE COUNT == 1` for many ops is normal

A handful of ops (especially `TilizeWithValPaddingDeviceOperation`
during host-tensor uploads) legitimately run on 1 core because they
process a small alignment region. Don't chase those.

The ones to chase are *compute-heavy* ops (Matmul, Conv2d) running on
small core counts.

## Pitfall 10: Ignoring `ATTRIBUTES` when comparing ops

Two `MatmulDeviceOperation` rows with identical shape can have very
different `program_config` settings. Same shape, different config →
different `CORE COUNT`, different runtime.

Always inspect `ATTRIBUTES` before claiming "these are the same op."

```bash
# Show ATTRIBUTES of all matmuls
awk -F, '$1=="MatmulDeviceOperation" {print $5}' ops_perf_results_*.csv | sort -u | head
```

## Quick mental model

| Symptom | Likely cause |
|---|---|
| "Model is host-bound per OP TO OP LATENCY" | Pitfall 2 — read HOST DURATION instead |
| "Total kernel time is way too small" | Pitfall 3 (overflow) or 5 (mid-run crash) |
| "First few HOST DURATION values are huge" | Pitfall 4 — cold JIT, filter or warm up |
| "TypeError mid-profile" | Pitfall 5 — branch/build mismatch |
| "8-core matmul with 5 ms kernel" | Real bottleneck — wrong program_config (not a profiler bug) |

## Checklist before trusting a profile

- [ ] Trace was disabled during the run
- [ ] Op count matches expectation (no silent overflow / crash)
- [ ] Cold JIT rows filtered or warm path measured
- [ ] `OP TO OP LATENCY` not used for host overhead estimation
- [ ] `CORE COUNT` examined alongside duration
- [ ] `ATTRIBUTES` inspected for outlier ops
- [ ] Single device filtered if multi-chip
