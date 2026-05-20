# profiling-tt-metal — Backlog

Forward-looking improvements organized by Track and theme. Each item
is sized small enough to be a single PR / commit. Promote items off
this list when implemented; keep the list honest about what's stale.

Use this as the entry point when the question is "what's still missing
from this skill?".

## Conventions

- **High / Medium / Low** = subjective ROI vs effort. High = high
  value, low effort; Low = either expensive or narrow use case.
- File names in `backticks` point at the file inside this skill
  that should host the new content.
- "Done when:" gives the merge criterion so future readers can decide
  whether to retire the item.

---

## Track C (NoC Reports) — additional recipes

Ranked by priority (see "What more NoC analysis is possible" in the
session that produced these). All sit inside `noc-reports.md`.

### Priority: High

- **Recipe G: Hop distance histogram.** For each event with `dx`/`dy`,
  compute `|dx-sx| + |dy-sy|` and bucket. Reveals how local the
  workload's traffic actually is. Done when: histogram printed,
  `< 2-hop` / `2-5 hop` / `> 5-hop` percentages reported.
- **Recipe H: Time-domain (achieved bandwidth per NoC).** Sum
  `num_bytes` per NoC, divide by `(max(timestamp) - min(timestamp)) *
  cycle_period`. Compare to Blackhole NoC0/NoC1 theoretical ~300 GB/s.
  Critical for compute-bound vs NoC-bound classification. Done when:
  GB/s reported per `noc`, with theoretical-peak comparison.
- **Recipe I: Sankey-style top flows.** Group bytes by `(sx,sy) →
  (dx,dy)` pair, print top 20. Reveals dominant data paths in text
  form (no visualization needed). Done when: top-20 table with
  `bytes`, `% of total`, hop-distance column.

### Priority: Medium

- **Recipe J: Sync-to-data event ratio per zone.** For each zone,
  `count(SEMAPHORE_* ∪ BARRIER_*) / count(READ ∪ WRITE_)`. High ratio
  = over-synchronized. Done when: per-zone ratios printed with
  flag-threshold heuristic.
- **Recipe K: Transfer size histogram.** Bucket `num_bytes` by
  power-of-two, separately for READ and WRITE_. Highlights coalescing
  opportunities (many `< 256 B` transfers). Done when: histogram
  printed with "median / p95 / max" rollup.
- **Recipe L: Before/after capture diff helper.** Take two
  `noc_trace_dev*.json` directories, compute per-metric delta (total
  bytes, NoC0/1 split, hot cores). Used to verify that a tuning change
  actually moved NoC behavior. Done when: helper script in `noc-reports.md`
  with example output.

### Priority: Low

- **Recipe M: VC utilization heatmap.** Histogram `vc` field per NoC
  per source core. Mostly useful for kernel engineers — narrow audience.
  Done when: top VC-saturated `(sx, sy, noc)` triples listed.
- **Recipe N: Address stride pattern.** Detect arithmetic stride in
  per-core sorted `src_addr` / `dst_addr`. Useful for diagnosing
  bank-conflict patterns. Done when: stride table or "no stride
  detected" diagnostic.

### Priority: Documentation

- **Document fabric (FABRIC_*) event schema.** Currently
  `noc-reports.md` says "see `profiler.cpp:856+`" and skips fabric
  events. Multi-chip workload analyses need this. Done when: schema
  table for the FABRIC_* family + 1 recipe.

---

## Track D (Tracy RISC-V mode) — not yet implemented

Currently a placeholder in `SKILL.md`. Promote to its own file when
addressed.

- **Create `tracy-riscv-mode.md`.** Document the `-v` flag capture,
  what the `.tracy` GUI shows (per-RISC dispatch timeline, bubble
  visualization), Multi-CQ tuning patterns, and the human-handoff
  workflow (agent recommends the capture; analysis is GUI-driven).
  Done when: file exists with capture command, screenshot description,
  and pointer to `optimizing-ttnn-models` Step 5 (Multi-CQ).
- **Remove "future" markers from `SKILL.md`.** Once Track D has its
  own file, drop the `(future)` annotations in the workflow table
  and the 2D capability matrix.

---

## Track A (Performance Reports) — upstream contributions

The Python recipes in `analysis-recipes.md` are documented as
contribution candidates to `tt-perf-report`. The note in that file
already enumerates the mapping; track here when each lands upstream.

- [ ] PR: `tt-perf-report --core-distribution <op-type>` (Recipe 2)
- [ ] PR: `tt-perf-report --worst-n N --filter <op-type>` (Recipe 3)
- [ ] PR: `tt-perf-report --merge a.csv b.csv ...` (Recipe 4)
- [ ] PR: `tt-perf-report --under-parallelized --core-budget 130 --min-duration 500us` (Recipe 6)
- [ ] PR: `tt-perf-report --per-core-skew` (Recipe 7)

Done when: the flag lands in `tt-perf-report` upstream → delete the
corresponding recipe from `analysis-recipes.md` and replace with the
`tt-perf-report --flag` example.

---

## Track B (Memory Reports) — robustness

- **Track API drift.** `memory-reports.md` was rewritten in mid-2026
  when the ttnn capture path changed from `TTNN_CONFIG_OVERRIDES` to
  `ttnn.graph.full_graph_capture()`. The new flow is also evolving
  (Python repr / C++ repr dtype mixing, fragile autouse fixture).
  When `ttnn/ttnn/graph_report.py` or `ttnn/ttnn/graph.py` materially
  change, re-validate Recipes 1-8.
- **Recipe for capturing full UniAD pipeline.** Currently the
  `full_graph_capture` examples are standalone scripts. UniAD's pytest
  test crashes the Python interpreter when wrapped via autouse fixture.
  Investigate whether wrapping inside the test body (not autouse) is
  stable. Done when: a verified pattern for capturing a full pytest
  test (without code modification of fixtures) exists.
- **Document `buffer_pages` populate condition more precisely.**
  Currently the skill says "L1-sharded buffer required". Investigate
  whether DRAM_INTERLEAVED also populates the table on some configs.

---

## Cross-skill / evaluation work

- **Add planning_head case study to `tt-metal-perf-case-studies`.**
  The UniAD planning_head profile (Track A) revealed the same
  8-core matmul / bcast_batch pattern that the DETR-decoder case
  study documents, plus a NoC0/NoC1 77/23 imbalance. Write
  `uniad-planning-head-multipath.md` showing Track A + Track C
  combined diagnosis. Done when: new case study file + index entry
  in `tt-metal-perf-case-studies/SKILL.md`.
- **Transferability evaluation on a non-UniAD model.** The current
  skill was built end-to-end against UniAD only. Re-run the 3-Track
  workflow against an LLM (Llama-style) or vision model (ResNet50)
  and patch any UniAD-specific assumptions found. Done when: a new
  case study demonstrating the skill on a different model architecture.
- **Audit recipes for `ttnn` API regressions.** Memory Reports broke
  when the underlying capture API changed. Periodically (or on a
  major ttnn version bump) run each Recipe in `memory-sqlite-recipes.md`
  and each Recipe in `analysis-recipes.md` against a fresh capture to
  catch silent breakage. Done when: a small per-recipe sanity test
  exists, or the skill is marked "validated against ttnn @ commit X".

---

## Lessons-learned items (for the next skill author)

These are observations from the build-out of this skill that did not
become recipes but might inform future skill design.

- **Don't trust upstream tutorials.** `ttnn/tutorials/ttnn_visualizer.md`
  documented a Memory capture path that no longer produces output. The
  authoritative source is the code (database.py + graph_report.py
  docstrings).
- **Verify recipe assumptions on real data.** The NoC `op_name`
  field was documented as populated, but every event in our actual
  capture had `op_name: ""`. The skill should default to group keys
  that work without instrumentation.
- **Pipeline ergonomic test ≠ test of analysis usefulness.** Track A
  recipes produced actionable findings on the very first run. Track B
  recipes worked mechanically but the workload (DRAM-only) didn't
  exercise the L1-related queries. Track C recipes were all useful
  but skewed toward "what's where", not "what's slow". Future recipe
  design should ask both: does it run, and does it teach something?

---

## How to retire items

When you finish an item:
1. Delete the bullet from this file
2. Update the file the item points at (`noc-reports.md` etc.) with
   the new content
3. If it changes the SKILL.md mental model, also touch SKILL.md and
   `quick-reference.md`
4. If the change is non-trivial, mention it in the relevant case
   study or in `pitfalls.md`

Stale items rot fast. If something hasn't been touched in 3 months,
delete it or rewrite it with current context.
