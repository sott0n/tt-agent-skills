# Memory Reports (Graph Capture → SQLite)

## Objective

Capture a TTNN memory report — a SQLite database describing every
operation, tensor, buffer, and per-core/per-bank page allocation
during a model run. Use this when you need to reason about memory
layout, sharding topology, peak L1 usage, or quantization
opportunities.

The output is **fully agent-queryable** via `sqlite3` (see
`memory-sqlite-recipes.md` for ready-to-use SQL recipes).
It is **not** a profile of execution time — for that, capture a
Performance Report (`running-tracy.md`) and analyze it with the
`analyzing-tt-profiles` skill (`tt-perf-report.md`).

## ⚠ Workflow change

The older `TTNN_CONFIG_OVERRIDES`-driven path (still shown in
`ttnn/tutorials/ttnn_visualizer.md`) **no longer writes a SQLite
during execution**. The current workflow is **two steps**:

1. **Capture** the graph to JSON via the `ttnn.graph.full_graph_capture()`
   context manager wrapping your model code
2. **Import** the JSON to SQLite via `python -m ttnn.graph_report`

This is documented at the top of `ttnn/ttnn/graph_report.py`:

> "This module completely decouples graph capture (C++) from
> visualization (SQLite). No database operations happen during model
> execution - everything is offline."

If you set the legacy config flags (`enable_logging`,
`enable_detailed_buffer_report`, `enable_graph_report` in
`TTNN_CONFIG_OVERRIDES`) the run will not error, but no SQLite will
be written.

## When to use Memory Reports vs Performance Reports

| Goal | Use |
|---|---|
| Find peak L1 per op, OOM risk | **Memory Reports** |
| Audit buffer placement (DRAM / L1 / L1_SMALL / TRACE) | **Memory Reports** |
| Inspect tensor dtype / layout / memory_config per op | **Memory Reports** |
| Scan for quantization candidates (bf16 → bf8b) | **Memory Reports** |
| Capture the computation DAG | **Memory Reports** |
| Per-op kernel time, host time, CORE COUNT | Performance Reports |
| Op-to-op latency analysis | Performance Reports |
| `tt-perf-report` bound classification | Performance Reports |

Performance and Memory captures are **independent runs**. They can
happen in either order. The Performance Reports path uses
`python -m tracy ...`; the Memory Reports path needs neither tracy
nor TTNN env vars.

## Step 1 — Capture to JSON

### Pattern A: Standalone script (easiest)

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)
try:
    with ttnn.graph.full_graph_capture("/tmp/report.json", slow_dispatch=True):
        # ... your model runs here ...
        a = ttnn.from_torch(torch.randn(1, 1, 32, 768),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        w = ttnn.from_torch(torch.randn(768, 256),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.linear(a, w)
finally:
    ttnn.close_device(device)
```

`full_graph_capture(report_path, slow_dispatch=True)` on entry:
- enables Python stack traces
- enables detailed buffer tracing
- when `slow_dispatch=True`, temporarily disables `enable_fast_runtime_mode`
  so per-operation captured sub-graphs are populated

On exit it writes `report.json` and restores the previous settings.

### Pattern B: Pytest test (invasive — edit the test or model code)

To capture a graph from a pytest test, wrap the relevant call site
inside the test or the model:

```python
# In your test or model code, wrap the forward pass:
with ttnn.graph.full_graph_capture("/tmp/report.json", slow_dispatch=True):
    output = model(inputs)
```

**Do not** use a `@pytest.fixture(autouse=True)` wrapper that calls
`full_graph_capture` for every test — that pattern has caused
interpreter segfaults during fixture teardown in practice. Inline the
context manager at a specific call site instead.

### Capture verification

```bash
ls -la /tmp/report.json
# Expect: a small (<1 MB for a few ops, 10-100s of MB for a full model) JSON file
```

If the JSON is missing, `full_graph_capture` was not actually
entered. Check that the `with` block contains the code that runs the
ops you care about.

## Step 2 — Import JSON to SQLite

```bash
python -m ttnn.graph_report /tmp/report.json /tmp/db_dir/
```

Output:

```
/tmp/db_dir/
├── db.sqlite                                   ← main artifact (17 tables)
├── cluster_descriptor.yaml                     ← device cluster topology
└── physical_chip_mesh_coordinate_mapping_1_of_1.yaml
```

The importer reports counts to stdout:

```
- 2 devices
- 10 operations
- 28 tensors
- 21 device tensor entries
- 102 buffers
- 227 edges
- 10 stack traces captured
```

Optional flags:

```bash
# Custom DB filename
python -m ttnn.graph_report report.json db_dir/ --db-name custom.sqlite

# Also generate SVG visualizations of each op's sub-graph
python -m ttnn.graph_report report.json db_dir/ --svg

# Import an entire directory of JSON reports (multi-run aggregation)
python -m ttnn.graph_report reports_dir/ db_dir/
```

## Sanity check after import

```bash
DB=/tmp/db_dir/db.sqlite
python3 -c "
import sqlite3
c = sqlite3.connect('$DB')
for t in ['operations','tensors','buffers','buffer_pages','edges','devices']:
    n = c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'  {t:<20} {n}')
"
```

Expected:
- `operations` ≈ number of ttnn ops dispatched inside the `with` block
- `tensors` typically > operations (each op has 1+ input + 1 output)
- `buffers` populated; `buffer_pages` may be 0 (see Recipe 6 caveat)
- `devices` = host + each accelerator opened

If `operations` is 0, the model code was outside the `with` block.

## ttnn-visualizer (web UI handoff)

Same `db.sqlite` directory is what `ttnn-visualizer` consumes:

```bash
pip install ttnn-visualizer
ttnn-visualizer    # opens http://localhost:8000
```

Upload the `/tmp/db_dir/` directory under "Memory reports". For
agent-side analysis, query the SQLite directly — see
`memory-sqlite-recipes.md`.

## Pitfalls

- **Legacy `TTNN_CONFIG_OVERRIDES` path silently produces nothing.**
  Setting `enable_logging`, `enable_detailed_buffer_report`,
  `enable_graph_report` will appear to work (the test PASSES) but no
  SQLite is written. Use the two-step `full_graph_capture` path.
- **Code outside the `with` block is not captured.** Ops dispatched
  before `full_graph_capture` is entered, or after it exits, do not
  appear in the JSON. Wrap the smallest meaningful region.
- **Pytest autouse fixture for `full_graph_capture` segfaults.** Do
  not wrap *all* tests via `conftest.py`. Inline the context manager
  at a specific call site.
- **JSON can be hundreds of MB on large models.** The C++ side writes
  a single file; plan disk space. The SQLite is typically smaller
  (relational normalization).
- **`buffer_pages` is empty when no L1 buffer with per-core sharding
  was allocated.** Default `from_torch` puts tensors in DRAM. See
  `memory-sqlite-recipes.md` Recipe 6 caveat.

## Cross-references

- `ttnn/ttnn/graph_report.py` — JSON-to-SQLite importer (top docstring
  is the source of truth for the new workflow)
- `ttnn/ttnn/graph.py:full_graph_capture` — capture context manager
- `ttnn/ttnn/database.py` — SQLite schema (`CREATE TABLE` statements)
- `ttnn/tutorials/ttnn_visualizer.md` — official tutorial (currently
  describes the deprecated config-flag path)

## Checklist

- [ ] Model code wrapped in `with ttnn.graph.full_graph_capture(...):`
- [ ] No `TTNN_CONFIG_OVERRIDES` set (legacy path, will not write SQLite)
- [ ] Tracy is not also running in the same process
- [ ] JSON file created and non-empty (`ls -la` to verify)
- [ ] `python -m ttnn.graph_report` succeeded with non-zero counts
- [ ] Sanity check `operations` row count matches expectation
- [ ] Drop to `memory-sqlite-recipes.md` for analysis
