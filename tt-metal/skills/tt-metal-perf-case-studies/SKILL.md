---
name: tt-metal-perf-case-studies
description: "Concrete end-to-end performance optimization case studies on Tenstorrent hardware. Each case study walks through real profile data → bottleneck identification → fix → verification, with actual numbers from production model work. Use when you want a worked example of how to apply `profiling-tt-metal` and `optimizing-ttnn-models` together, or when looking for a similar model/op pattern that's already been solved."
---

# TT-Metal Performance Optimization Case Studies

Worked examples of profile-driven optimization on Tenstorrent hardware.
Each case study is a self-contained narrative: starting state, profile
data, hypothesis, fix, measured result. They are meant to be read in
full when the reader has a similar bottleneck pattern.

## Index

| Case | Model | Hardware | Bottleneck | Fix | Speedup |
|------|-------|----------|------------|-----|---:|
| `uniad-detr-decoder-matmul.md` | UniAD | Blackhole p150b | 24 matmul ops at 8/130 cores (3477 µs avg) | `linear_flatten_batch` reshape helper | DETR decoder 155 → 115 ms (−26%); warm wall 1768 → 1560 ms |

## How to use

1. Scan the index for a bottleneck pattern that matches yours.
2. Read the full case study — the analysis steps and pitfalls are
   usually more transferable than the specific code change.
3. Cross-reference with `profiling-tt-metal` (the profiling workflow
   that produced the data) and `optimizing-ttnn-models` (the
   optimization techniques applied).

## Adding a new case study

When wrapping up an optimization pass that taught something
non-obvious, add a file here so the next engineer doesn't redo the
analysis from scratch. Structure:

1. **Background** — model, hardware, baseline numbers
2. **Step 1** — how the suspicious phase was identified
3. **Step 2-N** — the profile → hypothesis → fix → verify loop
4. **Generalizable lessons** — what transfers to other models
5. **The code change** — concise diff sketch (not the full patch)
6. **What the profile didn't find** — the human judgment part

Then add a row to the index above.

## Related skills

- `profiling-tt-metal` — how to capture and read the profiles these
  case studies start from.
- `optimizing-ttnn-models` — the optimization techniques the case
  studies apply (matmul program_config, sharding, Metal Trace, etc.).
