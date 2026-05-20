# Python Recipes for Op-Level Analysis

## Objective

Turn one or more `ops_perf_results_*.csv` files into actionable
bottleneck summaries: per-op aggregates, distribution by core count,
worst offenders, cross-module aggregates.

The CSV is small (typically 100s of KB to a few MB) — `pandas` or
plain `csv` is enough. No need for a database.

## Relationship to `tt-perf-report`

The recipes below cover analyses that the official
[`tt-perf-report`](https://github.com/tenstorrent/tt-perf-report) CLI
does **not** yet support — per-op core-count distribution, worst-N
with `ATTRIBUTES`, cross-module CSV merging, under-parallelized op
triage, per-core skew. If `tt-perf-report` is enough for your task
(per-op aggregates with signpost filtering, bound classification),
prefer it — its CLI accepts the same `ops_perf_results_*.csv` files
this skill produces.

**These recipes are upstream contribution candidates.** Each of them
is generic (not UniAD- or model-specific) and would benefit other
model teams. When you reach for one of them, treat it as a signal
that `tt-perf-report` has a feature gap worth filing:

| Recipe | Possible `tt-perf-report` flag |
|---|---|
| Recipe 2: distribution by core count | `--core-distribution Matmul` |
| Recipe 3: worst-N individual ops with `ATTRIBUTES` | `--worst-n 10 Matmul` |
| Recipe 4: cross-module merge | `--merge a.csv b.csv c.csv` |
| Recipe 6: under-parallelized triage | `--under-parallelized --core-budget 130 --min-duration 500us` |
| Recipe 7: per-core skew | `--per-core-skew` |

When one of these lands upstream, delete the corresponding recipe
from this file and replace it with a `tt-perf-report --flag` example.
Goal: this file shrinks over time to "things tt-perf-report still
can't do."

## Recipe 1: per-op aggregate (one CSV)

```python
import csv
from collections import defaultdict

CSV = "generated/profiler/reports/<run>/<ts>/ops_perf_results_<run>_<ts>.csv"

stats = defaultdict(lambda: {"cnt":0, "k":0, "h":0, "core":0, "kmax":0})
with open(CSV) as fp:
    for row in csv.DictReader(fp):
        op = row["OP CODE"]
        try: k = int(row["DEVICE KERNEL DURATION [ns]"] or 0)
        except: k = 0
        try: h = int(row["HOST DURATION [ns]"] or 0)
        except: h = 0
        try: c = int(row["CORE COUNT"] or 0)
        except: c = 0
        s = stats[op]
        s["cnt"]+=1; s["k"]+=k; s["h"]+=h; s["core"]+=c
        s["kmax"] = max(s["kmax"], k)

rows = sorted(stats.items(), key=lambda x: -x[1]["k"])
total_k = sum(s["k"] for _,s in rows)
total_h = sum(s["h"] for _,s in rows)
total_n = sum(s["cnt"] for _,s in rows)

print(f"total: {total_n} ops, {total_k/1e6:.2f} ms kernel, {total_h/1e6:.2f} ms host")
print(f"{'op':<44} {'cnt':>5} {'ker ms':>9} {'avg µs':>8} {'max µs':>9} {'cores':>6} {'%':>5}")
print("-"*92)
for op, s in rows[:20]:
    avg = s["k"]/s["cnt"]/1e3
    pct = 100*s["k"]/total_k
    print(f"{op:<44} {s['cnt']:>5} {s['k']/1e6:>9.2f} {avg:>8.1f} "
          f"{s['kmax']/1e3:>9.1f} {s['core']/s['cnt']:>6.1f} {pct:>5.1f}")
```

## Recipe 2: distribution by core count (for one op)

When a single op type (e.g. `MatmulDeviceOperation`) is the bottleneck,
slice it by `CORE COUNT` to find under-parallelized outliers:

```python
import csv
from collections import defaultdict

CSV = "..."
TARGET = "MatmulDeviceOperation"

by_core = defaultdict(lambda: {"cnt":0, "k":0, "kmax":0})
with open(CSV) as fp:
    for row in csv.DictReader(fp):
        if row["OP CODE"] != TARGET: continue
        c = int(row["CORE COUNT"] or 0)
        k = int(row["DEVICE KERNEL DURATION [ns]"] or 0)
        b = by_core[c]
        b["cnt"]+=1; b["k"]+=k; b["kmax"] = max(b["kmax"], k)

print(f"{'cores':>6} {'count':>6} {'total ms':>9} {'avg µs':>8} {'max µs':>8}")
print("-"*45)
for c in sorted(by_core):
    b = by_core[c]
    print(f"{c:>6} {b['cnt']:>6} {b['k']/1e6:>9.2f} {b['k']/b['cnt']/1e3:>8.0f} "
          f"{b['kmax']/1e3:>8.0f}")
```

Expected output pattern (UniAD DETR decoder, Phase A pre-fix):

```
 cores  count  total ms   avg µs   max µs
     8     24     83.44     3477     5558
    16      6     19.13     3188     3202
    29     48      1.43       30      174
    79      6      0.13       21       22
   100      6      0.35       58       59
```

The 8-core rows are 100× slower per op than the 79-core rows of the
same op type. That is the bottleneck.

## Recipe 3: worst N individual op instances

```python
import csv

CSV = "..."
TARGET = "MatmulDeviceOperation"

worst = []
with open(CSV) as fp:
    for row in csv.DictReader(fp):
        if row["OP CODE"] != TARGET: continue
        k = int(row["DEVICE KERNEL DURATION [ns]"] or 0)
        worst.append((k, row.get("CORE COUNT","?"), row.get("ATTRIBUTES","")[:100]))

worst.sort(reverse=True)
for k, c, attr in worst[:10]:
    print(f"{k/1e3:>8.0f} µs  ({c:>3} cores)  {attr}")
```

## Recipe 4: cross-module aggregate

When you ran per-module profiles separately, sum them to get a
model-wide picture:

```python
import csv
from collections import defaultdict
from pathlib import Path

modules = [
    "generated/profiler/reports/uniad_decoder/2026_05_11_06_40_25/ops_perf_results_uniad_decoder_2026_05_11_06_40_25.csv",
    "generated/profiler/reports/uniad_planning_head/2026_05_11_06_44_31/ops_perf_results_uniad_planning_head_2026_05_11_06_44_31.csv",
    # ... etc
]

total = defaultdict(lambda: {"cnt":0,"k":0,"h":0,"core":0,"kmax":0})
for path in modules:
    with open(path) as fp:
        for row in csv.DictReader(fp):
            op = row["OP CODE"]
            k = int(row["DEVICE KERNEL DURATION [ns]"] or 0)
            h = int(row["HOST DURATION [ns]"] or 0)
            c = int(row["CORE COUNT"] or 0)
            t = total[op]
            t["cnt"]+=1; t["k"]+=k; t["h"]+=h; t["core"]+=c
            t["kmax"] = max(t["kmax"], k)

# print as in Recipe 1
```

## Recipe 5: filter out cold (JIT) rows

If your profile includes the first forward pass, the first ~100-300 ops
have inflated `HOST DURATION`. Drop them:

```python
import csv

CSV = "..."
SKIP_FIRST_N = 200    # tune by inspecting GLOBAL CALL COUNT vs HOST DURATION

with open(CSV) as fp:
    reader = csv.DictReader(fp)
    rows = list(reader)

warm = [r for r in rows if int(r["GLOBAL CALL COUNT"]) > SKIP_FIRST_N]
# ... aggregate warm only
```

Or better, identify the JIT cliff visually:

```python
hosts = sorted([(int(r["GLOBAL CALL COUNT"]), int(r["HOST DURATION [ns]"]))
                 for r in rows])
# look for the index where HOST DURATION drops to <50000 ns
```

## Recipe 6: spot under-parallelized ops across all op types

A useful first-pass triage — find every op whose kernel is slow *and*
core count is low:

```python
import csv

CSV = "..."
CORES_BUDGET = 130   # Blackhole p150b; adjust per device

slow_underused = []
with open(CSV) as fp:
    for row in csv.DictReader(fp):
        k = int(row["DEVICE KERNEL DURATION [ns]"] or 0)
        c = int(row["CORE COUNT"] or 0)
        if k > 500_000 and c < CORES_BUDGET / 2:    # >500µs and using <half the cores
            slow_underused.append((k, c, row["OP CODE"], row.get("ATTRIBUTES","")[:80]))

slow_underused.sort(reverse=True)
print(f"Found {len(slow_underused)} slow + under-parallelized ops")
for k, c, op, attr in slow_underused[:15]:
    print(f"{k/1e3:>7.0f} µs ({c:>3} cores) {op:<30} {attr}")
```

## Recipe 7: per-core skew (load imbalance)

```python
import csv

CSV = "..."

skewed = []
with open(CSV) as fp:
    for row in csv.DictReader(fp):
        try:
            mn = int(row["DEVICE KERNEL DURATION PER CORE MIN [ns]"] or 0)
            mx = int(row["DEVICE KERNEL DURATION PER CORE MAX [ns]"] or 0)
        except: continue
        if mn > 0 and mx / mn > 1.5:
            skewed.append((mx/mn, row["OP CODE"], mx, mn, row.get("CORE COUNT","?")))

skewed.sort(reverse=True)
for ratio, op, mx, mn, c in skewed[:10]:
    print(f"skew={ratio:.2f}x  op={op:<30} max={mx/1e3:.0f}µs min={mn/1e3:.0f}µs ({c} cores)")
```

High skew = some cores have much more work; sharding may be uneven.

## Tips

- **Always print totals first**: `sum(DEVICE KERNEL DURATION)`,
  `sum(HOST DURATION)`, `op count`. If totals don't match expectations
  (e.g. full UniAD forward should be ~1000 ms kernel; if you see
  25 ms, you only captured a fraction).
- **Average µs/op is more useful than total ms** when comparing
  parallelization. Two ops with the same total kernel time can have
  very different averages and tail lengths.
- **Track `kmax`** (the slowest single op of a kind). The mean can
  hide a single tail-latency outlier.
- **Save the aggregation script as a project utility** so you don't
  re-type it every time. e.g.
  `models/<model>/tools/profile_summary.py`.

## Pandas equivalent (one-liner)

```python
import pandas as pd
df = pd.read_csv("ops_perf_results_*.csv")
df.groupby("OP CODE").agg(
    cnt=("OP CODE","count"),
    kernel_ms=("DEVICE KERNEL DURATION [ns]", lambda x: x.sum()/1e6),
    avg_us=("DEVICE KERNEL DURATION [ns]", lambda x: x.mean()/1e3),
    max_us=("DEVICE KERNEL DURATION [ns]", lambda x: x.max()/1e3),
    avg_core=("CORE COUNT","mean"),
).sort_values("kernel_ms", ascending=False).head(20)
```

## Checklist

- [ ] Total kernel + host + op count printed first
- [ ] Top ops by kernel time identified
- [ ] For top op type: distribution by `CORE COUNT` examined
- [ ] Worst N individual instances inspected with `ATTRIBUTES`
- [ ] Cold rows filtered out if profile included first forward
