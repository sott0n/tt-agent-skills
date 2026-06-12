# Interpreting `ops_perf_results_*.csv` Columns

## Objective

Know which columns to trust, which are inflated by instrumentation, and
how to read per-op shape / layout / dtype / memory and the perf-model
roofline directly from columns (no `ATTRIBUTES` parsing needed for shapes).

## Full column list (v2.1 format — ~120 columns)

Don't index by number — header order shifts across releases and the
modern CSV is wide (~120 cols). Match by name. The columns cluster into
groups:

```
IDENTITY / CONFIG
  OP CODE, OP TYPE, GLOBAL CALL COUNT, DEVICE ID, DEVICE ARCH,
  ATTRIBUTES, MATH FIDELITY, CORE COUNT, AVAILABLE WORKER CORE COUNT,
  SUB DEVICE ID, PARALLELIZATION STRATEGY

HOST TIMING
  HOST START TS, HOST END TS, HOST DURATION [ns]

DEVICE TIMING  (the load-bearing block)
  DEVICE FW START/END CYCLE, OP TO OP LATENCY [ns] (INFLATED — see below),
  DEVICE FW DURATION [ns], DEVICE KERNEL DURATION [ns]  <- trust this,
  DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns]  <- per-core skew,
  DEVICE {BRISC,NCRISC,TRISC0,TRISC1,TRISC2,ERISC} KERNEL DURATION [ns],
  DEVICE COMPUTE CB WAIT FRONT / RESERVE BACK [ns]  <- data-starvation,
  DISPATCH TOTAL CQ CMD OP TIME / GO SEND WAIT TIME [ns]

TENSOR SHAPES / FORMATS  (first-class columns — DON'T parse ATTRIBUTES for these)
  INPUT_<n>_{W,Z,Y,X}_PAD[LOGICAL]   for n = 0..5   (padded[logical] dims)
  INPUT_<n>_LAYOUT     (TILE / ROW_MAJOR)
  INPUT_<n>_DATATYPE   (BFLOAT16 / BFLOAT8_B / ...)
  INPUT_<n>_MEMORY     (DEV_<k>_DRAM_INTERLEAVED / DEV_<k>_L1_* / ...)
  OUTPUT_<n>_*         same fields, n = 0..1

KERNEL IDENTITY / CACHE
  COMPUTE/DATA MOVEMENT KERNEL SOURCE + HASH, PROGRAM HASH,
  PROGRAM CACHE HIT  <- False = this op compiled on this call (cold)

PERF-MODEL ROOFLINE  (per-op, not just matmul)
  PM IDEAL [ns], PM COMPUTE [ns], PM BANDWIDTH [ns],
  PM FPU UTIL (%)   <- ~0 means NOT compute-bound,
  NOC UTIL (%), DRAM BW UTIL (%), MULTICAST NOC UTIL (%),
  ETH BW UTIL (%), NPE CONG IMPACT (%)
```

**⚠️ Which roofline columns are populated depends on the capture.** In a
plain `python -m tracy -p -r` run, only `PM IDEAL` and `PM FPU UTIL (%)`
are filled; `DRAM BW UTIL (%)`, `NOC UTIL (%)`, and the `CB WAIT/RESERVE`
columns are **empty** — they're computed by **tt-npe**, which only runs
with `--collect-noc-traces` + `--analyze-noc-traces` (see
`noc-reports.md`). To confirm a *memory/NoC-bandwidth* bound you usually
need a NoC capture, not the perf CSV alone.

## Trust matrix

| Column | Trust? | Notes |
|---|---|---|
| `DEVICE KERNEL DURATION [ns]` | **High** | Sum it for total device work. This is the load-bearing column. |
| `HOST DURATION [ns]` | High | Warm path: 5-15 µs/op typical. Cold path (first ~100 ops): inflated by JIT compile. |
| `CORE COUNT` | High | Critical for spotting under-parallelized ops. |
| `MATH FIDELITY` | High | Reveals fidelity choice per op. |
| `ATTRIBUTES` | High | Long string; parse for shape / dtype / program_config. |
| `DEVICE FW DURATION [ns]` | Medium | Usually = kernel + 1-2 µs FW wrap. Use kernel duration instead. |
| `OP TO OP LATENCY [ns]` | **Low** | Device-side cycle gap. Inflated by profiler-side bookkeeping (see below). |
| `DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG` | High | Per-core skew indicator: if MAX ≫ MIN, work is unbalanced across cores. |
| Per-RISC durations (BRISC/NCRISC/TRISC0-2/ERISC) | Medium | Mostly diagnostic; usually dominated by the active engine for the op. |
| `DEVICE COMPUTE CB WAIT FRONT [ns]` | Medium | Time waiting for CB input. Long waits = upstream slow / CB under-sized. |
| `DEVICE COMPUTE CB RESERVE BACK [ns]` | Medium | Time waiting for CB output space. Long waits = downstream slow / CB under-sized. |

## The `OP TO OP LATENCY` trap

`OP TO OP LATENCY [ns]` is computed as:

```
LATENCY[N] = DEVICE FW START CYCLE[N+1] - DEVICE FW END CYCLE[N]
```

i.e. the device-side cycle gap between op N ending and op N+1 starting.

In a profiler-instrumented build, this gap **also includes the time the
profiler spends writing markers to the DRAM ring buffer**, which is
substantial. On a non-profiler build, the gap would be much smaller.

**Do not interpret OP-TO-OP latency as "host dispatch overhead".** That
mistake leads to wrong conclusions like "model is 80% host-bound, add
Metal Trace" when in fact the model is compute-bound with bad parallelization.

To estimate real host overhead:
- Use `HOST DURATION` directly.
- Or measure wall time with profiler **off** and subtract
  `sum(DEVICE KERNEL DURATION)`. The remainder is real host + DRAM
  transfer time.

See `tt_metal/impl/profiler/profiler_analysis.cpp:514-531` for the
exact OP-TO-OP formula.

## Reading `ATTRIBUTES`

`ATTRIBUTES` is a long semicolon-delimited string in the form:

```
{'attr1': 'value1'; 'attr2': 'value2'; ...; 'compute_kernel_config': 'ComputeKernelConfig(math_fidelity=HiFi2;...)'}
```

For matmul, useful fields to grep for:

- `'bcast_batch'` — batching mode
- `'compute_kernel_config'` — fidelity + flags (`packer_l1_acc`, `fp32_dest_acc_en`)

**Shapes are no longer in `ATTRIBUTES`** — read the `INPUT_<n>_*_PAD[LOGICAL]`
/ `INPUT_<n>_LAYOUT` / `_DATATYPE` / `_MEMORY` columns directly.

## Classifying a NON-matmul op (what `tt-perf-report` won't do)

`tt-perf-report`'s `Bound` column is **matmul-only** (plus `HOST` for
`(torch)` fallbacks). For the ops that often dominate a model —
`BinaryNg`, `ReshapeView`, `Permute`, `Tilize/Untilize` — it gives no
bound and no advice. Classify them yourself from columns:

| Read | Means |
|---|---|
| `PM FPU UTIL (%)` ≈ 0 | **Not compute-bound** — the FPU is idle; cost is data movement, not math. |
| `INPUT_<n>_MEMORY` = `*_DRAM_INTERLEAVED` | Operand lives in DRAM → every touch is a DRAM round-trip. Moving it to L1 / fusing to avoid the round-trip is the lever. |
| large `INPUT/OUTPUT_*_PAD[LOGICAL]` (e.g. 10000×256) | Big tensor → memory-bandwidth-bound; cost scales with bytes moved. |
| `INPUT_<n>_LAYOUT` = `ROW_MAJOR` on a compute op | Forces a tilize somewhere; layout churn. |
| `PROGRAM CACHE HIT` = `False` on a warm iter | This instance recompiled — you're measuring cold, re-check the forward split. |
| `DEVICE KERNEL DURATION PER CORE MAX ≫ MIN` | Per-core skew (bad sharding), not raw op cost. |

So: a `BinaryNg` with `PM FPU UTIL ≈ 0`, `DRAM_INTERLEAVED` inputs, and a
large shape is **DRAM-bandwidth-bound** — the fix is layout/placement
(L1, sharding, fusion), not a faster compute kernel. To *quantify* the
bandwidth saturation you need `DRAM BW UTIL (%)`, which is only populated
in a NoC capture (see `noc-reports.md`).

Quick filter for "all matmuls with HiFi2":

```bash
awk -F, '$1=="MatmulDeviceOperation" && $7 ~ /HiFi2/' ops_perf_results_*.csv | wc -l
```
(`$7` = `MATH FIDELITY` in v2.1; verify with `head -1 csv | tr ',' '\n' | nl | grep FIDELITY`.)

## How to spot bottleneck patterns from columns

### Pattern 1: Under-parallelized matmul

- `OP CODE == MatmulDeviceOperation`
- `CORE COUNT` is small (8 or 16 on a 130-core Blackhole)
- `DEVICE KERNEL DURATION` is large (1000+ µs)
- `DEVICE KERNEL DURATION PER CORE MAX / MIN` ratio close to 1
  (cores are equally loaded — they're just too few)

→ Fix: pick a different `program_config` or reshape input so the
heuristic picks a wider parallelization. See the
`tt-metal-perf-case-studies` skill (`uniad-detr-decoder-matmul.md`)
for a worked example.

### Pattern 2: Cold dispatch (JIT compile)

- First ~100-300 rows by `GLOBAL CALL COUNT` have `HOST DURATION` in
  the 100s of ms
- Subsequent calls of the same `OP CODE` with same `ATTRIBUTES` drop
  to single-digit µs

→ Filter out cold rows in analysis (skip the first N rows or run with
warm iterations).

### Pattern 3: Unbalanced sharded op

- `DEVICE KERNEL DURATION PER CORE MAX / MIN` > 1.5
- `CORE COUNT` is high but `MAX` core is the bottleneck

→ Re-examine sharding; one core has more work than the rest.

### Pattern 4: Layout glue overhead

- High count of `TilizeWithValPadding`, `UntilizeWithUnpadding`,
  `ReshapeView`, `Permute` ops
- Their summed `DEVICE KERNEL DURATION` is non-trivial (e.g. 10%+)

→ Audit explicit `ttnn.to_layout` / reshape calls; many can be
eliminated by using sharded matmul / fused ops.

## Wall time vs kernel sum

```
wall_time   ≈ sum(DEVICE KERNEL DURATION) + sum(HOST DURATION) + DRAM IO + overhead
device_work = sum(DEVICE KERNEL DURATION)
host_work   = sum(HOST DURATION)
```

If `device_work / wall_time` is close to 1 → compute-bound → tune ops.
If `host_work / wall_time` is significant → host-bound → trace/multi-CQ.

`OP TO OP LATENCY` never enters this equation — ignore it for the
top-line bottleneck classification.

## Checklist

- [ ] Confirmed which column index has `DEVICE KERNEL DURATION` (header order can shift across releases)
- [ ] Aware that `OP TO OP LATENCY` is inflated and not host overhead
- [ ] Filtered cold rows if profile contained the first forward pass
- [ ] Looked at `CORE COUNT` alongside duration for every "slow" op
- [ ] Parsed `ATTRIBUTES` to find the actual config of outlier ops
