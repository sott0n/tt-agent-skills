# Quick Reference — TT-Metal Profiling

## Choosing a report

| Goal | Report | Capture |
|---|---|---|
| Per-op kernel time, CORE COUNT, host overhead | Performance | `python -m tracy -p -r ...` |
| Buffer placement, peak L1, tensor dtype/layout | Memory | `with ttnn.graph.full_graph_capture(...):` + `python -m ttnn.graph_report` |
| NoC traffic, link congestion | NoC (experimental) | `python -m tracy --collect-noc-traces -p -r ...` |
| RISC-V dispatch bubbles, Multi-CQ tuning | Tracy GUI (future) | `python -m tracy -p ...` then open `.tracy` in GUI |

Performance and Memory captures are independent — Memory uses a
Python context manager (no env var, no tracy), so they don't conflict
even in the same process. Separate runs remain cleaner.

## Track A: Performance Reports — one-line invocations

```bash
# Standard performance report
python -m tracy -p -r -n <run_name> -m pytest <test_path>::<test_func> -svv

# Profile a standalone script
python -m tracy -p -r -n <run_name> path/to/script.py

# With env required by the workload (trace disabled)
TT_DCN_DEVICE=1 TT_UNIAD_TRACE_DISABLE=1 \
  python -m tracy -p -r -n <run_name> -m pytest <test_path>::<test_func> -svv
```

## Track B: Memory Reports — two-step workflow

The `TTNN_CONFIG_OVERRIDES` path is deprecated (it produces no SQLite).
Capture is via `ttnn.graph.full_graph_capture()` in Python code, then
import to SQLite separately:

```python
# Step 1: in your model script or test, wrap with full_graph_capture
import ttnn
with ttnn.graph.full_graph_capture("/tmp/report.json", slow_dispatch=True):
    # ... your model forward pass / ttnn ops ...
    out = ttnn.linear(a, w)
```

```bash
# Step 2: import JSON to SQLite
python -m ttnn.graph_report /tmp/report.json /tmp/db_dir/
# Output: /tmp/db_dir/db.sqlite + cluster_descriptor.yaml + mesh_coord.yaml

DB=/tmp/db_dir/db.sqlite

# Top 10 L1 consumers (empty if workload is DRAM-only)
sqlite3 "$DB" "
  SELECT o.name, SUM(b.max_size_per_bank) AS l1
  FROM operations o JOIN buffers b ON b.operation_id=o.operation_id
  WHERE b.buffer_type=1 GROUP BY o.operation_id
  ORDER BY l1 DESC LIMIT 10;"

# Buffer placement audit (0=DRAM 1=L1 3=L1_SMALL 4=TRACE)
sqlite3 "$DB" "
  SELECT buffer_type, COUNT(*), SUM(max_size_per_bank)
  FROM buffers GROUP BY buffer_type;"

# Tensor dtype distribution (handle Python-repr / C++-repr mixing)
sqlite3 "$DB" "
  SELECT REPLACE(REPLACE(dtype,'DataType.',''),'DataType::','') AS d,
         COUNT(*)
  FROM tensors GROUP BY d ORDER BY 2 DESC;"
```

`db.sqlite` schema reference + 8 recipes: `memory-sqlite-recipes.md`.
Full capture workflow: `memory-reports.md`.

## Track C: NoC Reports — one-line invocations (experimental)

```bash
# Capture — keep forward_passes ≤ 1 to limit DRAM ring buffer pressure
tt-smi -r 0
python -m tracy --collect-noc-traces --op-support-count 10000 -p -r \
  -n <run_name> -m pytest <test_path>::<test_func> -svv

# A final AssertionError in process_ops_logs.py is non-fatal — the JSON
# traces are still written. The error stops only the tracy report
# enrichment step.

# Output: generated/profiler/.logs/noc_trace_dev*.json + topology.json
LOGS=generated/profiler/.logs

# NoC0/NoC1 balance + total bytes
python3 -c "
import json, glob
from collections import Counter
LOGS = '$LOGS'
total = 0; noc = Counter()
for p in glob.glob(f'{LOGS}/noc_trace_dev*.json'):
    for ev in json.load(open(p)):
        if 'num_bytes' in ev:
            total += ev['num_bytes']; noc[ev['noc']] += ev['num_bytes']
print(f'total NoC: {total/1e6:.1f} MB')
for n, b in noc.most_common():
    print(f'  {n}: {b/1e6:>8.1f} MB ({100*b/total:.1f}%)')
"

# Hot source cores (op_name is usually empty, so group by core)
python3 -c "
import json, glob
from collections import Counter
src = Counter()
for p in glob.glob('$LOGS/noc_trace_dev*.json'):
    for ev in json.load(open(p)):
        if 'num_bytes' in ev: src[(ev['sx'], ev['sy'])] += ev['num_bytes']
for (x,y), b in src.most_common(10):
    print(f'src ({x:>2},{y:>2}): {b/1e6:>6.2f} MB')
"

# Bytes by kernel phase (BRISC-KERNEL / NCRISC-KERNEL — usable even
# when op_name is empty)
python3 -c "
import json, glob
from collections import Counter
z = Counter()
for p in glob.glob('$LOGS/noc_trace_dev*.json'):
    cur = None
    for ev in json.load(open(p)):
        if ev.get('zone_phase') == 'ZONE_START': cur = ev['zone']
        elif ev.get('zone_phase') == 'ZONE_END': cur = None
        elif 'num_bytes' in ev and cur: z[cur] += ev['num_bytes']
for k, b in z.most_common(): print(f'{k:<28} {b/1e6:>8.2f} MB')
"
```

Schema + 6 recipes + pitfalls: `noc-reports.md`.

**Gotchas**: plain write is `WRITE_` (not `WRITE`); `op_name` is empty
unless the model emits Tracy op-zone markers; `--op-support-count`
may not prevent overflow on large workloads.

## Prerequisite env

```bash
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=blackhole  # or wormhole_b0 / grayskull
export PATH=$TT_METAL_HOME/build_Release/tools/profiler/bin:$PATH
source python_env/bin/activate

tt-smi -r 0   # reset device before each profile run
```

## Output locations

```
# Performance Reports (Track A)
generated/profiler/reports/<run_name>/<timestamp>/
├── ops_perf_results_<run_name>_<timestamp>.csv  ← main artifact
├── profile_log_device.csv
└── tracy_profile_log_host.tracy                 ← open in Tracy GUI

# Memory Reports (Track B) — wherever you pass to `python -m ttnn.graph_report`
<db_dir>/
├── db.sqlite                                     ← main artifact (17 tables)
├── cluster_descriptor.yaml
└── physical_chip_mesh_coordinate_mapping_*.yaml

# NoC Reports (Track C, experimental) — per-op JSON files
generated/profiler/.logs/
├── noc_trace_dev<N>_<op>_ID<id>[_traceID...].json
├── topology.json                                  # multi-chip routing
└── cluster_coordinates.json
```

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

## Common pitfalls (one-liners)

- Profiler + Metal Trace = fatal. Disable trace before profiling (Track A).
- Memory Reports legacy `TTNN_CONFIG_OVERRIDES` path produces no SQLite —
  use `ttnn.graph.full_graph_capture()` + `python -m ttnn.graph_report`.
- Memory Reports: code outside the `with` block is not captured.
- Memory Reports: do not wrap every pytest in `full_graph_capture` via
  autouse fixture — observed interpreter segfault. Inline at a call site.
- Memory Reports: `dtype` stored as both `DataType.X` and `DataType::X` —
  use `LIKE '%X'` or `REPLACE` to normalize.
- Profiler DRAM buffer overflows on high-frequency ops (Conv2d, GridSample);
  raise `TT_PROFILER_OP_SUPPORT_COUNT` or split per-module captures.
- A 5 ms matmul on 8 cores is almost always a config bug, not real compute.
  Always look at `CORE COUNT` alongside duration.
- `OP TO OP LATENCY` is **not** host overhead. It is device-side cycle gap
  inflated by profiler. Use `HOST DURATION` for dispatch cost.
- NoC Reports overflow the DRAM ring buffer fast — keep `forward_passes ≤ 1`
  and raise `TT_PROFILER_OP_SUPPORT_COUNT` for larger models. Track C is
  experimental; schema may shift between releases.

## Per-module profile pattern

For a large model, profile each submodule separately to avoid DRAM overflow:

```python
# write a per-module pytest, e.g. test_ttnn_<module>.py::test_<module>
python -m tracy -r -n <model>_<module> -m pytest \
    models/.../tests/test_ttnn_<module>.py::test_<module> -svv
```

The CSVs land in `generated/profiler/reports/<run_name>/<ts>/`. Aggregate
them with the Python recipe above.

## Reset before / after

```bash
tt-smi -ls                  # list devices
tt-smi -r 0                 # reset device 0 (Blackhole p150b)
tt-smi -r 0,1,2,3           # reset all 4 devices (QuietBox / multi-chip)
```

## Where to look next

- Bottleneck = compute-bound (kernel ≈ wall) → `optimizing-ttnn-models`
  Step 6 (Conv2d) or matmul `program_config` tuning.
- Bottleneck = host-bound (host ≫ kernel) → `optimizing-ttnn-models`
  Step 4 (Metal Trace) or Step 5 (Multi-CQ).
- Outlier op uses too few cores → see `tt-metal-perf-case-studies`
  skill, `uniad-detr-decoder-matmul.md`.
