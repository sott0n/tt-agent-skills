# Interpreting `ops_perf_results_*.csv` Columns

## Objective

Know which columns to trust, which are inflated by instrumentation,
and how to read shape / config info from `ATTRIBUTES`.

## Full column list (current order)

```
1.  OP CODE                              # e.g. MatmulDeviceOperation
2.  OP TYPE                              # tt_dnn_device / tt_metal_l1_to_l1 / ...
3.  GLOBAL CALL COUNT                    # monotonically increasing op index
4.  DEVICE ID
5.  ATTRIBUTES                           # op-specific config (long string)
6.  MATH FIDELITY                        # LoFi / HiFi2 / HiFi3 / HiFi4
7.  CORE COUNT                           # cores the op was dispatched on
8.  PARALLELIZATION STRATEGY             # e.g. "Width", "1D" — not always populated
9.  HOST START TS                        # ns since profiler init
10. HOST END TS
11. HOST DURATION [ns]                   # = HOST END - HOST START
12. DEVICE FW START CYCLE
13. DEVICE FW END CYCLE
14. OP TO OP LATENCY [ns]                # device-side gap N→N+1; INFLATED by profiler
15. OP TO OP LATENCY BR/NRISC START [ns]
16. DEVICE FW DURATION [ns]
17. DEVICE KERNEL DURATION [ns]          # the real device-side work for this op
18. DEVICE KERNEL DURATION DM START [ns]
19. DEVICE KERNEL DURATION PER CORE MIN [ns]
20. DEVICE KERNEL DURATION PER CORE MAX [ns]
21. DEVICE KERNEL DURATION PER CORE AVG [ns]
22. DEVICE KERNEL FIRST TO LAST START [ns]
23. DEVICE BRISC KERNEL DURATION [ns]
24. DEVICE NCRISC KERNEL DURATION [ns]
25. DEVICE TRISC0 KERNEL DURATION [ns]
26. DEVICE TRISC1 KERNEL DURATION [ns]
27. DEVICE TRISC2 KERNEL DURATION [ns]
28. DEVICE ERISC KERNEL DURATION [ns]
29. DEVICE COMPUTE CB WAIT FRONT [ns]
30. DEVICE COMPUTE CB RESERVE BACK [ns]
... (more per-RISC and CB-event columns follow)
```

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
- shape information embedded in tensor specs

Quick filter for "all matmuls with HiFi2":

```bash
awk -F, '$1=="MatmulDeviceOperation" && $5 ~ /HiFi2/' ops_perf_results_*.csv | wc -l
```

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
