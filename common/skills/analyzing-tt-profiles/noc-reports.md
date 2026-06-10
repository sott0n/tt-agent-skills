# NoC Reports (`--collect-noc-traces`)

> **⚠ Experimental.** The NoC reporting path, the JSON event schema, and
> the optional `tt-npe` analyzer are still evolving. Field names and
> output directory conventions may change between releases. Treat the
> recipes below as guidance, not a stable contract — verify against the
> current `tt_metal/impl/profiler/profiler.cpp` if anything looks off.

> **Capture entry point.** NoC traces are currently captured only via the
> TTNN Tracy path (`python -m tracy --collect-noc-traces`, see the
> `profiling-tt-metal` skill). `ttrt perf` does not yet expose a
> NoC-trace flag. The JSON schema and analysis recipes below apply to the
> output regardless of who produced it.

## Objective

Capture per-RISC NoC events (reads, writes, multicast, semaphores) on
every Tensix core during a run, as JSON files. Use this when you
suspect NoC traffic — link congestion, multicast inefficiency, or
disproportionate data movement — is the bottleneck.

This is the only report type in the skill that exposes **the routing
layer** (intra-chip NoC, and on multi-chip configurations the
fabric). Performance Reports (Track A) and Memory Reports (Track B)
cannot answer "which NoC link is saturated."

## Capture

```bash
tt-smi -r 0
python -m tracy --collect-noc-traces -p -r -n <run_name> \
  -m pytest <test_path>::<test_func> -svv
```

Under the hood, the `--collect-noc-traces` flag sets:

| Env var | Effect |
|---|---|
| `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` | Enables NoC event recording in the device profiler |
| `TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=<dir>` | Where the per-op JSON files are written |

You can set these env vars directly if running outside `python -m tracy`.

**Strongly recommended**: keep `forward_passes ≤ 1` for NoC capture
runs. Each NoC event consumes a marker slot in the device DRAM ring
buffer; multi-pass runs overflow quickly. Raise the buffer with
`TT_PROFILER_OP_SUPPORT_COUNT=10000` (default 1000) for large models,
but expect long post-processing time.

## Output

Two report directories are produced:

```
# 1. Per-op NoC event traces (the main artifact)
generated/profiler/.logs/
├── noc_trace_dev<N>_<op_name>_ID<runtime_id>[_traceID<...>].json
├── ...
├── topology.json              # cluster routing (multi-chip)
└── cluster_coordinates.json   # device coord mapping

# 2. Standard performance CSV (also produced because `-r` is passed)
generated/profiler/reports/<run_name>/<ts>/
├── ops_perf_results_<run_name>_<ts>.csv
└── ...
```

The NoC trace dir defaults to `generated/profiler/.logs/`. Override
with the env var above if needed.

## JSON event schema

Each `noc_trace_dev<N>_*.json` is a JSON array of event objects.
Two flavors, distinguished by whether `zone_phase` is present:

### Zone marker events (kernel phase boundaries)

```json
{
  "run_host_id": 42,
  "op_name": "",                          // see "op_name caveat" below
  "proc": "BRISC",                        // BRISC / NCRISC / TRISC0..2
  "src_device_id": 0,
  "zone": "BRISC-KERNEL",                 // or NCRISC-KERNEL, READER-KERNEL, etc.
  "zone_phase": "ZONE_START",             // or ZONE_END
  "sx": 1, "sy": 2,
  "timestamp": 5681077486161              // device cycles
}
```

### Local NoC events (the data-movement payload)

```json
{
  "run_host_id": 42,
  "op_name": "",
  "proc": "NCRISC",
  "noc": "NOC_0",                         // NOC_0 / NOC_1
  "vc": -1,                                // virtual channel; -1 = "any / unset"
  "src_device_id": 0,
  "sx": 1, "sy": 2,                        // source core
  "num_bytes": 2048,
  "type": "WRITE_",                        // see full enum below
  "timestamp": 5681077486370,
  "dx": 0, "dy": 11,                       // destination core (unicast)
  // OR for WRITE_MULTICAST / SEMAPHORE_SET_MULTICAST:
  "mcast_start_x": 1, "mcast_start_y": 0,
  "mcast_end_x":  10, "mcast_end_y": 2,
  // Optional (trailer, only some types):
  "src_addr": 123456, "dst_addr": 654321,
  "posted": true, "noc_status_counter": 17
}
```

### `op_name` caveat (frequently empty)

`op_name` populates only when the model code is instrumented with
**Tracy operation-zone markers** (`tracy.scoped_signpost` etc., or
C++ `OpZoneScoped`). Many model implementations (including the
UniAD modules) do not emit these — in which case **every event has
`op_name: ""`**.

Do not write recipes that group by `op_name` unless you have verified
the workload emits zone markers. Use `(sx, sy)` core coordinates,
`zone` (kernel phase: `BRISC-KERNEL` / `NCRISC-KERNEL`), or
`run_host_id` as group keys instead.

### Full `type` enum

The `type` field is the string name of
`EMD::NocEventType` from `tt_metal/tools/profiler/event_metadata.hpp`.
Current values:

```
UNDEF
# READ family
READ                READ_SET_STATE          READ_SET_TRID
READ_WITH_STATE     READ_WITH_STATE_AND_TRID
READ_BARRIER_START  READ_BARRIER_END        READ_BARRIER_WITH_TRID
READ_DRAM_SHARDED_SET_STATE      READ_DRAM_SHARDED_WITH_STATE
# WRITE family — note the trailing underscore on plain WRITE_
WRITE_              WRITE_SET_TRID          WRITE_WITH_TRID
WRITE_INLINE        WRITE_MULTICAST
WRITE_SET_STATE     WRITE_WITH_STATE
WRITE_WITH_TRID_SET_STATE        WRITE_WITH_TRID_WITH_STATE
WRITE_BARRIER_START WRITE_BARRIER_END       WRITE_BARRIER_WITH_TRID
WRITE_FLUSH         WRITE_FLUSH_WITH_TRID
FULL_BARRIER        ATOMIC_BARRIER
# Semaphore family
SEMAPHORE_INC       SEMAPHORE_WAIT          SEMAPHORE_SET
SEMAPHORE_SET_REMOTE                        SEMAPHORE_SET_MULTICAST
SEMAPHORE_INC_MULTICAST
# Fabric (inter-chip) events from FABRIC_UNICAST_WRITE onward
FABRIC_*
```

**Key naming gotcha**: plain write is `WRITE_` (trailing underscore),
not `WRITE`. Recipes that filter `type == "WRITE"` match nothing.
Use `LIKE 'WRITE%'` or check `type.startswith("WRITE")`.

Only `READ*`, `WRITE_*`, `WRITE_MULTICAST`, and the `*_WITH_TRID`
variants carry `num_bytes`. BARRIER / SEMAPHORE / FLUSH events have
no payload (they are sync primitives).

Fabric (inter-chip) events have additional metadata and a different
serialization path; see `profiler.cpp:856+` if you need them.

## When to use NoC Reports vs other tracks

| Symptom | Track |
|---|---|
| "op kernel is slow, low CORE COUNT" | Track A (Perf) |
| "L1 OOM, sharding too eager" | Track B (Memory) |
| "kernel is fast in isolation but slow in pipeline — suspect data wait" | **Track C (NoC)** |
| "is NoC0 / NoC1 balanced?" | **Track C (NoC)** |
| "multicast efficiency unclear" | **Track C (NoC)** |
| "where is data actually flowing (which cores → which cores)?" | **Track C (NoC)** |
| "multi-chip workload — fabric link congested?" | **Track C (NoC)** |

Note: identifying *which op* a NoC pattern belongs to is hard without
op-zone markers. If you need per-op NoC breakdown, instrument the
model first (see "op_name caveat" in the schema section). For
per-kernel-phase breakdown, Recipe B works out of the box.

## Analysis options

### Option 1: Direct JSON parsing (agent-friendly)

The JSON files are small enough to read directly. The recipes below
have been verified on an actual capture (planning_head, 4 M events,
841 MB total NoC bytes). They use group keys that work even when
`op_name` is empty.

```python
import json, glob
from collections import Counter
LOGS = "generated/profiler/.logs"
files = sorted(glob.glob(f"{LOGS}/noc_trace_dev*.json"))
```

#### Recipe A: totals + NoC0/NoC1 imbalance

```python
total_bytes = 0
noc_bytes = Counter()
type_bytes = Counter()
for p in files:
    for ev in json.load(open(p)):
        if "num_bytes" not in ev:
            continue
        total_bytes += ev["num_bytes"]
        noc_bytes[ev["noc"]] += ev["num_bytes"]
        type_bytes[ev["type"]] += ev["num_bytes"]

print(f"Total NoC bytes: {total_bytes/1e6:.1f} MB")
for n, b in noc_bytes.most_common():
    print(f"  {n}: {b/1e6:>8.1f} MB  ({100*b/total_bytes:.1f}%)")
# Imbalance: if NOC_0 > 70% or NOC_1 > 70%, one NoC is underused.
# Dual-NoC kernels can use the idle plane to double effective bandwidth.
```

#### Recipe B: bytes by kernel phase (zone)

Use this in place of "bytes by op_name" — works even when `op_name`
is empty.

```python
zone_bytes = Counter()
for p in files:
    cur_zone = None
    for ev in json.load(open(p)):
        if ev.get("zone_phase") == "ZONE_START":
            cur_zone = ev["zone"]
        elif ev.get("zone_phase") == "ZONE_END":
            cur_zone = None
        elif "num_bytes" in ev and cur_zone:
            zone_bytes[cur_zone] += ev["num_bytes"]

for z, b in zone_bytes.most_common():
    print(f"  {z:<28} {b/1e6:>8.2f} MB")
# Typical: NCRISC-KERNEL is the writer kernel (large bytes).
# BRISC-KERNEL is the reader.
```

#### Recipe C: hot source cores (emitters)

```python
src = Counter()
for p in files:
    for ev in json.load(open(p)):
        if "num_bytes" in ev:
            src[(ev["sx"], ev["sy"])] += ev["num_bytes"]

for (x, y), b in src.most_common(10):
    print(f"  src=({x:>2},{y:>2})  {b/1e6:>8.2f} MB")
# A cluster of source cores on a single row (e.g. all sy=2) reveals
# the sharding axis. If 7 cores in y=2 each emit ~15 MB, the workload
# is sharded across 7 cores in row y=2.
```

#### Recipe D: hot destination cores (receivers)

```python
dst = Counter()
for p in files:
    for ev in json.load(open(p)):
        if "num_bytes" in ev and "dx" in ev:
            dst[(ev["dx"], ev["dy"])] += ev["num_bytes"]

for (x, y), b in dst.most_common(10):
    print(f"  dst=({x:>2},{y:>2})  {b/1e6:>8.2f} MB")
# Bytes concentrated on column boundaries (x=0 or x=9 on Blackhole)
# typically means DRAM controllers are the destination — i.e. many
# cores writing back to DRAM. Cross-reference cluster_coordinates.json
# to map device-physical to logical worker cores.
```

#### Recipe E: transfer-type breakdown (sync vs data)

```python
ev_type = Counter()
for p in files:
    for ev in json.load(open(p)):
        if "type" in ev:
            ev_type[ev["type"]] += 1

# Group by family
families = {"READ":0, "WRITE":0, "SEMAPHORE":0, "BARRIER":0, "OTHER":0}
for t, n in ev_type.items():
    if t.startswith("READ") and "BARRIER" not in t: families["READ"] += n
    elif t.startswith("WRITE") and "BARRIER" not in t and "FLUSH" not in t: families["WRITE"] += n
    elif "SEMAPHORE" in t: families["SEMAPHORE"] += n
    elif "BARRIER" in t or "FLUSH" in t: families["BARRIER"] += n
    else: families["OTHER"] += n

for f, n in families.items():
    print(f"  {f:<12} {n:>10,}")
# High SEMAPHORE / BARRIER counts vs data events = compute & data
# movement are serialized; consider double-buffering or async patterns.
```

#### Recipe F: multicast usage

```python
mcast = [ev for p in files for ev in json.load(open(p)) if ev.get("type") == "WRITE_MULTICAST"]
print(f"WRITE_MULTICAST events: {len(mcast)}")
mcast_bytes = sum(ev["num_bytes"] for ev in mcast)
print(f"WRITE_MULTICAST total:  {mcast_bytes/1e6:.2f} MB")
# If multicast bytes are << 1% of total NoC traffic but many cores read
# from the same source, multicast is under-utilized. Sharded reads
# from a shared weight tensor are the prime multicast candidate.
```

### Option 2: `tt-npe` (Tenstorrent NoC Performance Estimator)

If `tt-npe` is installed (separate Tenstorrent project, not vendored
in this repo), `python -m tracy --collect-noc-traces` will additionally
import and run `npe_analyze_noc_trace_dir` after capture:

```
tools/tracy/process_ops_logs.py:1671
    from npe_analyze_noc_trace_dir import analyze_noc_traces_in_dir
```

Output: `npe_viz/*.npeviz.zst` files (binary, visualized in tt-npe's
own GUI). When `tt-npe` is missing, the import silently fails and only
the JSON traces are written — that is the agent-friendly path.

For analysis with tt-npe, install per its upstream docs and pass the
`.logs/` directory.

## Pitfalls specific to NoC Reports

- **`op_name` is empty unless the model emits Tracy op-zone markers.**
  Many TTNN model implementations do not. Group by `(sx, sy)`, `zone`,
  or `run_host_id` instead. (See "op_name caveat" in the schema
  section.)
- **`type == "WRITE"` matches nothing.** Plain write is `WRITE_`
  (trailing underscore). Use `LIKE 'WRITE%'` or
  `type.startswith("WRITE")`.
- **DRAM ring buffer overflow is acute and `--op-support-count` may
  not be enough.** Observed message during a planning_head capture:
  `Profiler DRAM buffers were full, markers were dropped! device 0,
  worker core 1,2, Risc BRISC, bufferEndIndex = 120000` even with
  `--op-support-count 10000`. Workloads with high BRISC NoC activity
  can saturate before any reasonable buffer size. Mitigations: split
  the workload (per-module captures), use a single forward pass,
  accept partial coverage and verify event counts.
- **Post-processing CSV enrichment can `AssertionError`.** Seen:
  `process_ops_logs.py:516 AssertionError: Device data missing: Op
  267264 not present in cpp_device_perf_report.csv for device 0
  (trace_id=None)`. The JSON traces are still written and usable; the
  failure is in the tracy report generator's final step, not the
  capture itself. Don't be fooled by the stack trace at the end of
  the run — check `generated/profiler/.logs/noc_trace_dev*.json` for
  the actual data.
- **Post-processing time grows non-linearly** with event count.
  Plan for minutes, not seconds, on a full-pipeline NoC capture.
- **`type` enum is wide and grows over releases.** The schema section
  above lists the values from `event_metadata.hpp` at the time of
  writing; verify against the current header if a recipe filter
  stops matching.
- **Coordinates are NoC0-translated.** The `dx`/`dy` and `mcast_*`
  fields are projected to NoC0 space even when `noc` is `NOC_1`.
  Cross-reference `cluster_coordinates.json` for logical coordinates.
- **Multi-chip mixes intra-chip and fabric events.** Fabric events
  (`FABRIC_*` types) use a different schema (see `profiler.cpp:856+`);
  the recipes above ignore them.

## Cross-references

- `tt_metal/impl/profiler/profiler.cpp:780-1044` — event JSON
  serialization (source of truth for the schema)
- `tt_metal/tools/profiler/event_metadata.hpp` — `NocEventType` enum
- `tt_metal/tools/profiler/noc_event_profiler.hpp` — kernel-side
  recording
- `tools/tracy/__main__.py:128-237` — `--collect-noc-traces` flag
  wiring (env vars)
- `tools/tracy/process_ops_logs.py:1671` — optional tt-npe import

## Checklist

- [ ] Run with `forward_passes ≤ 1` to control DRAM overflow
- [ ] `TT_PROFILER_OP_SUPPORT_COUNT` or `--op-support-count` raised
  (note: even 10000 may not be enough for large models — accept
  dropped markers if the warning appears)
- [ ] `generated/profiler/.logs/noc_trace_dev*.json` files present
  and non-empty (count and size them; tiny files mean events were
  dropped)
- [ ] `topology.json` and `cluster_coordinates.json` present
- [ ] If the run ended with `AssertionError` in `process_ops_logs.py`,
  ignore it — check the JSON traces directly
- [ ] Decide group key: `op_name` only if instrumented, else `zone`
  or `(sx, sy)`
- [ ] When writing recipes, treat the type enum as version-dependent
  (`WRITE_` not `WRITE`)
