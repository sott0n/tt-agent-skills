# Quick Reference — Analyzing TT Profiles

This cheatsheet is the **analysis** layer. For how to *capture* a
profile, see the project capture skill:
- TTNN: `profiling-tt-metal` (`python -m tracy`, `full_graph_capture`)
- tt-forge: `tt-forge-perf` (`ttrt perf`, `ttrt run --memory`)

Both front-ends drive the same tt-metal Tracy stack and emit the same
`ops_perf_results*.csv`, so everything below applies regardless of who
captured the profile.

## Choosing a report

| Goal | Report | Capture (see project skill) |
|---|---|---|
| Per-op kernel time, CORE COUNT, host overhead | Performance CSV | TTNN: `python -m tracy -p -r` · forge: `ttrt perf` |
| NoC traffic, link congestion | NoC JSON (experimental) | TTNN: `python -m tracy --collect-noc-traces` |
| Buffer placement, peak L1, tensor dtype/layout | Memory | TTNN: `full_graph_capture` → SQLite · forge: `ttrt run --memory` → JSON |
| RISC-V dispatch bubbles, Multi-CQ tuning | Tracy GUI (future) | open the `.tracy` file in the Tracy GUI |

## Trustworthy CSV columns

| Column | Meaning | Trust? |
|---|---|---|
| `DEVICE KERNEL DURATION [ns]` | Time op kernel actually ran on cores | **Yes** |
| `HOST DURATION [ns]` | Host dispatch time | Yes |
| `CORE COUNT` | Cores the op used | Yes |
| `MATH FIDELITY` | LoFi / HiFi2 / HiFi3 / HiFi4 | Yes |
| `ATTRIBUTES` | op-specific shape/config string | Yes |
| `OP TO OP LATENCY [ns]` | Device-side cycle gap N→N+1 | **No** — inflated by profiler |
| `DEVICE FW DURATION [ns]` | FW wrap around kernel | Approx (often ~kernel + 1-2 µs) |

Full breakdown: `csv-columns.md`.

## Quick CSV aggregation (tt-perf-report CLI — preferred)

```bash
pip install tt-perf-report
tt-perf-report ops_perf_results_*.csv                       # full table
tt-perf-report ops_perf_results_*.csv --min-percentage 1.0  # hide noise
tt-perf-report ops_perf_results_*.csv --print-signposts     # list phases
tt-perf-report ops_perf_results_*.csv \
  --start-signpost detr_decoder --end-signpost detr_decoder_end
tt-perf-report ops_perf_results_*.csv --csv perf.csv --no-color --no-advice
```

See `tt-perf-report.md` for full flag reference.

## Quick CSV aggregation (Python — fallback)

```python
import csv
from collections import defaultdict

stats = defaultdict(lambda: {"cnt":0, "k":0, "h":0, "core":0, "kmax":0})
with open("ops_perf_results_*.csv") as fp:
    for row in csv.DictReader(fp):
        op = row["OP CODE"]
        k = int(row["DEVICE KERNEL DURATION [ns]"] or 0)
        h = int(row["HOST DURATION [ns]"] or 0)
        c = int(row["CORE COUNT"] or 0)
        s = stats[op]
        s["cnt"]+=1; s["k"]+=k; s["h"]+=h; s["core"]+=c
        s["kmax"] = max(s["kmax"], k)

for op, s in sorted(stats.items(), key=lambda x:-x[1]["k"])[:15]:
    print(f"{op:<40} {s['cnt']:>5} {s['k']/1e6:>8.2f}ms  avg={s['k']/s['cnt']/1e3:.0f}µs  "
          f"max={s['kmax']/1e3:.0f}µs  cores={s['core']/s['cnt']:.1f}")
```

More recipes (core-count distribution, worst-N, cross-module merge,
per-core skew): `analysis-recipes.md`.

## NoC JSON quick analysis (experimental)

NoC traces land at `generated/profiler/.logs/noc_trace_dev*.json`.

```python
# NoC0/NoC1 balance + total bytes
import json, glob
from collections import Counter
LOGS = "generated/profiler/.logs"
total = 0; noc = Counter()
for p in glob.glob(f"{LOGS}/noc_trace_dev*.json"):
    for ev in json.load(open(p)):
        if "num_bytes" in ev:
            total += ev["num_bytes"]; noc[ev["noc"]] += ev["num_bytes"]
print(f"total NoC: {total/1e6:.1f} MB")
for n, b in noc.most_common():
    print(f"  {n}: {b/1e6:>8.1f} MB ({100*b/total:.1f}%)")
```

Schema + 6 recipes + pitfalls: `noc-reports.md`.

**Gotchas**: plain write is `WRITE_` (not `WRITE`); `op_name` is empty
unless the model emits Tracy op-zone markers — group by `(sx, sy)` or
`zone` instead.

## Common analysis pitfalls (one-liners)

- `OP TO OP LATENCY` is **not** host overhead. It is device-side cycle gap
  inflated by profiler. Use `HOST DURATION` for dispatch cost.
- A 5 ms matmul on 8 cores is almost always a config bug, not real compute.
  Always look at `CORE COUNT` alongside duration.
- "Total kernel time way too small" → DRAM ring buffer overflow dropped
  ops (`TT_PROFILER_OP_SUPPORT_COUNT`) or the run crashed mid-capture.
  Always check op count vs expectation first.
- First ~100-300 ops are cold (JIT compile) — filter or run a warm pass.
- Multi-chip CSV mixes devices — filter `DEVICE ID` before summing.

Full list: `pitfalls.md`. Capture-time pitfalls (Metal Trace conflict,
memory-capture quirks) live in the project capture skill.

## Output locations (artifact layout)

```
# Performance Reports
generated/profiler/reports/.../ops_perf_results_*.csv  ← main artifact
generated/profiler/.logs/profile_log_device.csv        ← raw device markers
tracy_profile_log_host.tracy                            ← open in Tracy GUI

# NoC Reports (experimental)
generated/profiler/.logs/noc_trace_dev<N>_*.json
generated/profiler/.logs/topology.json                 # multi-chip routing
generated/profiler/.logs/cluster_coordinates.json
```

(`ttrt perf` writes the same files; with `--save-artifacts` they are
also copied into the ttrt artifacts dir.)

## Hardware reset

```bash
tt-smi -ls                  # list devices
tt-smi -r 0                 # reset device 0
tt-smi -r 0,1,2,3           # reset all 4 (QuietBox / multi-chip)
```

If a run wedges the device, see the `recovering-tt-hardware` skill.

## Where to look next

- Bottleneck = compute-bound (kernel ≈ wall) → `optimizing-ttnn-models`
  (Conv2d / matmul `program_config` tuning) or `tt-forge-optimize`.
- Bottleneck = host-bound (host ≫ kernel) → Metal Trace / Multi-CQ.
- Outlier op uses too few cores → see `tt-metal-perf-case-studies`
  (`uniad-detr-decoder-matmul.md`).
