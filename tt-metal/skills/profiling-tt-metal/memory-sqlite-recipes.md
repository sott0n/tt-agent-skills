# SQL Recipes for Memory Reports

## Objective

Query the `db.sqlite` produced by `memory-reports.md` to extract actionable
memory-layout findings: peak L1 per op, buffer placement audit,
quantization candidates, large-tensor inventory, DAG hotspots.

Every recipe below is a single SQL statement against the schema in
`ttnn/ttnn/database.py`. No special tooling required — plain
`sqlite3 <db.sqlite>` or Python `sqlite3`.

## Schema reference

The 17 tables (from `ttnn/ttnn/database.py` and the importer in
`ttnn/ttnn/graph_report.py`):

| Table | Key columns | Purpose |
|---|---|---|
| `devices` | `device_id`, `worker_l1_size`, `l1_num_banks`, `total_l1_for_tensors` | Per-device topology |
| `operations` | `operation_id UNIQUE`, `name`, `duration` (float, Python-side seconds) | Op metadata |
| `operation_arguments` | `operation_id`, `name`, `value` | Per-op config strings (parallel to `ATTRIBUTES` in Perf CSV) |
| `stack_traces` | `operation_id`, `stack_trace` | Python stack trace per op |
| `tensors` | `tensor_id UNIQUE`, `shape`, `dtype`, `layout`, `memory_config`, `device_id`, `address`, `buffer_type` | Tensor metadata |
| `device_tensors` | `tensor_id`, `device_id`, `address` | Multi-device tensor placement |
| `input_tensors` | `operation_id`, `input_index`, `tensor_id` | Op→input edges |
| `output_tensors` | `operation_id`, `output_index`, `tensor_id` | Op→output edges |
| `buffers` | `operation_id`, `device_id`, `address`, `max_size_per_bank`, `buffer_type`, `buffer_layout` | Per-op buffer allocations |
| `buffer_pages` | `operation_id`, `device_id`, `address`, `core_y`, `core_x`, `bank_id`, `page_index`, `page_address`, `page_size`, `buffer_type` | Per-core, per-bank pages — **populated only when ops produce sharded L1 buffers** |
| `nodes` | `operation_id`, `unique_id`, `node_operation_id`, `name` | DAG nodes |
| `edges` | `operation_id`, `source_unique_id`, `sink_unique_id`, `source_output_index`, `sink_input_index`, `key` | DAG edges |
| `captured_graph` | `operation_id`, `captured_graph` (JSON) | Raw DAG JSON |
| `local_tensor_comparison_records` | `tensor_id UNIQUE`, `golden_tensor_id`, `matches`, `desired_pcc`, `actual_pcc` | Per-op PCC (from `enable_comparison_mode`) |
| `global_tensor_comparison_records` | same as above | Cross-run PCC |
| `errors` | `operation_id`, `operation_name`, `error_type`, `error_message`, `stack_trace`, `timestamp` | Run-time errors |
| `report_metadata` | `key UNIQUE`, `value` | Schema version + capture metadata (added by importer in `graph_report.py`) |

**`buffer_type` integer mapping** (from
`tt_metal/api/tt-metalium/buffer_types.hpp`):

| Value | Type |
|---|---|
| 0 | `DRAM` |
| 1 | `L1` |
| 2 | `SYSTEM_MEMORY` |
| 3 | `L1_SMALL` |
| 4 | `TRACE` |

Indexes are created on `buffers.address`, `buffers.operation_id`,
`buffers.device_id`, `buffers.max_size_per_bank`, `buffers.buffer_type`,
`input_tensors.tensor_id`, `output_tensors.tensor_id`.

## Common gotchas

Three quirks the recipes below have to handle:

1. **`dtype` is stored in two formats** — Python repr `DataType.BFLOAT16`
   and C++ repr `DataType::BFLOAT16` both appear in the same `tensors`
   table (one from `from_torch`, one from on-device intermediate
   tensors). Filter against both forms: `dtype IN ('DataType.BFLOAT16',
   'DataType::BFLOAT16')` or use `LIKE '%BFLOAT16'`. Plain
   `dtype = 'DataType::BFLOAT16'` will miss roughly half the rows.

2. **Recipe 1 returns empty when no L1 buffer was allocated.** `from_torch`
   defaults to DRAM, so a script that only creates tensors via
   `from_torch` and runs eltwise ops will have `buffer_type` = 0 (DRAM)
   for everything. To populate L1 buffers, the model must use a
   sharded `memory_config` somewhere. If Recipe 1 returns nothing,
   that is the explanation — not a missing flag.

3. **`buffer_pages` is empty unless an op produces a sharded L1 buffer
   with per-core page tracking.** With the default
   `full_graph_capture(slow_dispatch=True)` invocation, this table is
   populated only for ops that actually shard. If you need per-core
   layout for a non-sharded workload, the question itself is
   ill-posed — the buffer lives on a single bank.

## Recipe 1: Peak L1 per op (find OOM candidates)

```sql
SELECT
  o.operation_id,
  o.name,
  SUM(b.max_size_per_bank) AS l1_total_per_bank
FROM operations o
JOIN buffers b ON b.operation_id = o.operation_id
WHERE b.buffer_type = 1     -- L1
GROUP BY o.operation_id
ORDER BY l1_total_per_bank DESC
LIMIT 20;
```

Multiply `l1_total_per_bank` by `devices.l1_num_banks` to get
total L1 across the device. Watch for ops approaching `worker_l1_size`.

## Recipe 2: Buffer placement audit (DRAM vs L1 vs L1_SMALL vs TRACE)

```sql
SELECT
  CASE buffer_type
    WHEN 0 THEN 'DRAM'
    WHEN 1 THEN 'L1'
    WHEN 3 THEN 'L1_SMALL'
    WHEN 4 THEN 'TRACE'
    ELSE 'OTHER'
  END AS placement,
  COUNT(*) AS num_buffers,
  SUM(max_size_per_bank) AS total_size_per_bank,
  AVG(max_size_per_bank) AS avg_size_per_bank
FROM buffers
GROUP BY buffer_type
ORDER BY total_size_per_bank DESC;
```

Use to verify that bulk weights live in DRAM and activations in L1
(or vice versa per design).

## Recipe 3: Tensor dtype distribution (quantization scan)

```sql
-- Normalize Python vs C++ repr ('DataType.X' vs 'DataType::X')
-- so each dtype reports a single count
SELECT
  REPLACE(REPLACE(dtype, 'DataType.', ''), 'DataType::', '') AS normalized_dtype,
  COUNT(*) AS num_tensors,
  COUNT(DISTINCT tensor_id) AS unique_tensors
FROM tensors
GROUP BY normalized_dtype
ORDER BY num_tensors DESC;
```

Large `bfloat16` count is a quantization opportunity (bf16 → bfloat8_b).
Cross-check with Recipe 5 to identify which large tensors are still
bf16.

**Note** (Common gotcha #1): without the `REPLACE` normalization, the
same dtype shows up as two rows. Make sure to use one form or the
other in downstream filters.

## Recipe 4: Per-op input/output tensor shapes

```sql
SELECT
  o.operation_id, o.name,
  it.input_index AS idx, 'in' AS kind,
  t.shape, t.dtype, t.layout, t.memory_config
FROM operations o
JOIN input_tensors it ON it.operation_id = o.operation_id
JOIN tensors t ON t.tensor_id = it.tensor_id
WHERE o.operation_id = ?       -- bind specific op id
UNION ALL
SELECT
  o.operation_id, o.name,
  ot.output_index AS idx, 'out' AS kind,
  t.shape, t.dtype, t.layout, t.memory_config
FROM operations o
JOIN output_tensors ot ON ot.operation_id = o.operation_id
JOIN tensors t ON t.tensor_id = ot.tensor_id
WHERE o.operation_id = ?
ORDER BY kind DESC, idx;
```

Drop-in replacement for grepping `ATTRIBUTES` in Perf Reports — gives
structured shape/layout/memory_config directly.

## Recipe 5: Large tensor inventory (> 1 MB)

Tensor size requires parsing `shape` (text) — easier in Python. Pure
SQL approximation: find tensors with non-trivial address space.

```sql
SELECT
  tensor_id, shape, dtype, layout, memory_config,
  CASE buffer_type
    WHEN 0 THEN 'DRAM' WHEN 1 THEN 'L1'
    WHEN 3 THEN 'L1_SMALL' WHEN 4 THEN 'TRACE'
    ELSE 'OTHER' END AS placement
FROM tensors
WHERE dtype LIKE '%BFLOAT16'  -- matches both 'DataType.BFLOAT16' and 'DataType::BFLOAT16'
ORDER BY tensor_id;
```

(See Common gotcha #1 — naive `dtype = 'DataType::BFLOAT16'` will miss
half the rows.)

For exact byte size, parse `shape` and `dtype` in Python (see
"Python helper" below).

## Recipe 6: Per-core load (page count per core)

```sql
SELECT
  core_y, core_x,
  COUNT(*) AS num_pages,
  SUM(page_size) AS bytes_resident
FROM buffer_pages
WHERE buffer_type = 1   -- L1
GROUP BY core_y, core_x
ORDER BY bytes_resident DESC;
```

Reveals sharding imbalance: if one core holds 2× the bytes of others,
shard topology is uneven.

**Caveat (Common gotcha #3)**: returns empty if no L1-sharded buffer
was allocated during the captured region. To use, run with a workload
that uses `memory_config=ttnn.create_sharded_memory_config(...)`. For
non-sharded workloads this question is ill-posed.

## Recipe 7: Top-N slowest ops (Python-side duration)

```sql
SELECT operation_id, name, duration
FROM operations
ORDER BY duration DESC
LIMIT 20;
```

Cross-check against Performance Reports CSV. Note: `operations.duration`
is Python-side wall, includes host overhead. Real kernel time is in
the Perf Reports CSV.

## Recipe 8: DAG edges incoming to a specific op

```sql
SELECT
  e.source_unique_id, ns.name AS source_name,
  e.sink_unique_id,   nk.name AS sink_name,
  e.source_output_index, e.sink_input_index
FROM edges e
JOIN nodes ns ON ns.unique_id = e.source_unique_id AND ns.operation_id = e.operation_id
JOIN nodes nk ON nk.unique_id = e.sink_unique_id   AND nk.operation_id = e.operation_id
WHERE e.operation_id = ?     -- bind specific op id
ORDER BY e.sink_input_index;
```

Populated when `full_graph_capture(slow_dispatch=True)` is used —
that enables per-op sub-graph capture, which `python -m ttnn.graph_report`
serializes into `nodes` / `edges`.

## Python helper (preferred for non-trivial joins)

```python
import sqlite3
import re

# Path is whatever you passed as <db_dir> to `python -m ttnn.graph_report`
DB = "/tmp/db_dir/db.sqlite"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Top L1 consumers
rows = conn.execute("""
    SELECT o.name, SUM(b.max_size_per_bank) AS l1
    FROM operations o JOIN buffers b ON b.operation_id = o.operation_id
    WHERE b.buffer_type = 1
    GROUP BY o.operation_id
    ORDER BY l1 DESC LIMIT 10
""").fetchall()

for r in rows:
    print(f"{r['name']:<40} {r['l1']/1024:>10.1f} KB / bank")

# Compute exact tensor size in bytes.
# Handle BOTH dtype formats: 'DataType.BFLOAT16' (Python repr) and
# 'DataType::BFLOAT16' (C++ repr) both appear in the tensors table.
def normalize_dtype(s):
    return s.replace("DataType.", "").replace("DataType::", "")

DTYPE_BYTES = {
    "BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1, "BFLOAT4_B": 0.5,
    "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1,
}

def shape_to_elem(shape_text):
    nums = re.findall(r"\d+", shape_text)
    n = 1
    for x in nums:
        n *= int(x)
    return n

# Large bf16 tensors (quantization candidates) — match both dtype formats
for r in conn.execute("SELECT * FROM tensors WHERE dtype LIKE '%BFLOAT16'"):
    elems = shape_to_elem(r["shape"])
    bytes_ = elems * DTYPE_BYTES[normalize_dtype(r["dtype"])]
    if bytes_ > 1_048_576:   # > 1 MB
        print(f"{r['tensor_id']:>6} {r['shape']:<30} {bytes_/1e6:>6.2f} MB  {r['memory_config']}")
```

## Notes on accuracy

- `operations.duration` is wall time including host dispatch. Use
  Performance Reports CSV `DEVICE KERNEL DURATION` for actual device
  work.
- `buffers.max_size_per_bank` is per-bank — multiply by
  `devices.l1_num_banks` to get total device-wide allocation.
- `buffer_pages` is populated only with
  `enable_detailed_buffer_report=true`. With the recommended config
  (`memory-reports.md`), this is enabled.
- DAG tables (`nodes`/`edges`/`captured_graph`) require
  `enable_graph_report=true`, off by default in the recommended config.

## tt-perf-report parity status

None of these recipes are covered by the current `tt-perf-report`
CLI — that tool consumes Perf Reports CSV, not Memory SQLite.

Memory Reports analysis is a logical extension that could be added
to `tt-perf-report` (or a sibling `tt-memory-report` tool). The
recipes above are upstream contribution candidates in the same spirit
as `analysis-recipes.md`:

| Recipe | Possible flag in a future `tt-memory-report` CLI |
|---|---|
| 1: Peak L1 per op | `--peak-l1 --top 20` |
| 2: Placement audit | `--placement-audit` |
| 3: Dtype distribution | `--dtype-summary` |
| 5: Large tensor inventory | `--large-tensors --min-mb 1` |
| 6: Per-core load | `--shard-balance` |

## Checklist

- [ ] `db.sqlite` exists and has populated `operations` + `buffers`
- [ ] Confirmed `buffer_type` integer mapping for the queries you write
- [ ] Used Python helper for size calculations (SQL alone can't parse
  shape strings)
- [ ] Cross-referenced findings with Performance Reports CSV when
  comparing duration
