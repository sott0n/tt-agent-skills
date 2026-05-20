# UniAD DETR Decoder — 8-core Matmul Bottleneck

## Objective

Walk through a real profile-driven optimization pass end-to-end:
**how the 8-core matmul bottleneck in UniAD's DETR decoder was located
and fixed**, with the actual numbers and the analysis steps that
produced them.

This case study uses the workflow from the `profiling-tt-metal` skill
(Tracy build → `python -m tracy` → CSV → Python aggregation). Re-read
that skill first if any step below is unclear.

## Background

- Model: UniAD (autonomous driving e2e perception + planning)
- Hardware: Blackhole p150b (130 cores per device)
- Branch: `kyamaguchi/uniad-dcn-fp32`
- Warm wall before: ~1.68 s
- Goal: ≤ 0.5 s (production target); intermediate milestone ≤ 1 s

## Step 1 — Identify which module to drill into

Per-phase host-side timings (from `TT_UNIAD_TIMING=1`) showed:

| Phase | Wall ms (call#3) |
|---|---:|
| extract_img_feat (ResNet+FPN) | 342 |
| BEV encoder | 440 |
| **DETR decoder** | **155** (before fix) |
| simple_test_track | 1040 |
| seg_head | 460 |
| motion_head | 120 |
| planning_head | 84 |

DETR decoder isn't the largest, but it's the most *suspicious* —
matmul shapes are small (M=32, K=256, N=256/512), nothing should
take 150 ms.

## Step 2 — Profile that module in isolation

Wrote a per-module test (`test_ttnn_decoder.py::test_uniad_decoder`)
and ran Tracy on it:

```bash
tt-smi -r 0
python -m tracy -r -n uniad_decoder -m pytest \
  models/experimental/uniad/tests/pcc/test_ttnn_decoder.py::test_uniad_decoder -svv
```

Output landed at `generated/profiler/reports/uniad_decoder/2026_05_11_06_40_25/`.

## Step 3 — Aggregate the CSV

Using `profiling-tt-metal` Recipe 1 (per-op aggregate from
`analysis-recipes.md`):

```
total: 1009 ops, 164.25 ms kernel, 22.83 ms host

op                                       cnt    ker ms  avg µs   max µs  cores    %
-----------------------------------------------------------------------------------
MatmulDeviceOperation                     90    104.47    1161     5558    30.6  63.6
ReshapeViewDeviceOperation               102     19.11     187     1567   130.0  11.6
BinaryNgDeviceOperation                  132      9.35      71      569   130.0   5.7
PermuteDeviceOperation                    60      8.94     149      723    86.8   5.4
TransposeDeviceOperation                  84      3.41      41      168   130.0   2.1
TilizeWithValPaddingDeviceOperation      108      3.22      30      348    19.9   2.0
GridSampleOperation                        6      2.54     423      429   130.0   1.5
```

**Observation**: 90 matmul ops produce 63.6% of kernel time. Average
1161 µs/op, max 5558 µs. With 130 available cores, the average usage
is only 30 cores.

## Step 4 — Slice the slow op type by core count

Using `profiling-tt-metal` Recipe 2 (distribution by core count for
Matmul only):

```
 cores  count  total ms   avg µs   max µs
     8     24     83.44     3477     5558    ← bottleneck
    16      6     19.13     3188     3202    ← bottleneck
    29     48      1.43       30      174    ← same shapes, configured well
    79      6      0.13       21       22    ← same shapes, configured well
   100      6      0.35       58       59    ← same shapes, configured well
```

**Key finding**: ops at 8 cores are running at 3477 µs/op average —
**100-150× slower** than the *same* op type at 79-100 cores
(21-58 µs). They use 6.2% of the device's compute capacity (8/130).

## Step 5 — Inspect `ATTRIBUTES` of the 8-core matmuls

Using `profiling-tt-metal` Recipe 3 (worst N instances):

```
     5558 µs (  8 cores) attr={'bcast_batch':'true'; 'compute_kernel_config':...HiFi2...}
     5558 µs (  8 cores) attr={'bcast_batch':'true'; 'compute_kernel_config':...HiFi2...}
     5557 µs (  8 cores) attr={'bcast_batch':'true'; 'compute_kernel_config':...HiFi2...}
     ...
```

All 8-core matmuls share `bcast_batch: true` (i.e. batched input shape
`(B, H, M, K) @ (K, N)`). The default `ttnn.linear` heuristic for
4D inputs of shape `(1, 901, 32, 256)` sees M=32 (1 tile in tile
units) and picks a 1D N-only program_config: 8 cores along N. That
leaves 122 cores idle.

## Step 6 — Hypothesize and validate the fix

Hypothesis: reshape the input from `(1, 901, 32, K)` to
`(1, 1, 901*32, K)` = `(1, 1, 28832, K)`. The new shape has M=28832
(901 tiles), which the heuristic parallelizes across both M and N.

Validation steps:

1. Wrote a helper `linear_flatten_batch(x, weight, bias)` that
   reshapes 3D+ inputs before `ttnn.linear` and reshapes back after.
2. Applied it to the 3 weight matmuls in `TtMultiheadAttention`
   (q/k/v projection) and 4 in `TtCustomMSDeformableAttention`
   (value_proj, sampling_offsets, attention_weights, output_proj).
3. Re-ran PCC check on `test_uniad` — sdc_traj stayed 0.9909 (gate
   floor 0.9905 ✓).
4. Re-ran `TT_UNIAD_TIMING=1` warm-wall measurement.

Result: warm wall 1768 ms → 1560 ms (−208 ms, of which ~108 ms is
attributable to the DETR decoder phase: 155 → 115 ms; the rest came
from BEV encoder where the same pattern applied).

## Step 7 — Verify with a second profile (post-fix)

Re-ran the per-module profile after the fix. The expected verification:
the 8-core / 16-core rows in the core-count distribution should
collapse (those ops should now show high core counts).

(In this branch the post-fix profile was blocked by an unrelated
`ttnn.grid_sample` API mismatch between branch and rebased build —
see `profiling-tt-metal` Pitfall 5. Re-profiling after rebuilding
remains useful as a closing step.)

## Generalizable lessons

1. **Suspicious phases are not always the slowest** — look at
   throughput per ms of compute, not just total ms. DETR decoder was
   small but had the highest "cost per FLOP" ratio.

2. **Always slice the bottleneck op by `CORE COUNT`** before claiming
   anything about model architecture. A 100× per-op speedup gap
   between identical-shape ops is almost certainly config, not real
   compute.

3. **Default `ttnn.linear` / `ttnn.matmul` heuristics are pessimistic
   for batched-4D inputs with small M**. The fix is either:
   - Reshape the input to flatten leading dims into M (cheap, just
     needs the helper)
   - Pass an explicit `MatmulMultiCoreReuseMultiCastProgramConfig`
     (more control, more code)

4. **PCC gate first, perf gate second**: changing matmul configs can
   subtly change numerical results. UniAD PCC sdc_traj had to stay
   ≥ 0.99 — verified after every config change.

5. **Per-module profiling beats full-pipeline profiling for
   diagnosis**: smaller capture, no DRAM overflow, easier to compare
   pre/post-fix. Use full-pipeline only for top-line wall-clock
   measurement.

## The code change (concise diff sketch)

```python
# Before: tiny M, 8-core matmul
out = ttnn.linear(x, weight, bias=bias)  # x: (1, 901, 32, 256)

# After: linear_flatten_batch helper
def linear_flatten_batch(x, weight, bias=None, **kwargs):
    if x.dim() < 3:
        return ttnn.linear(x, weight, bias=bias, **kwargs)
    leading = x.shape[:-1]                     # all dims except K
    flat_M = 1
    for d in leading: flat_M *= d
    x2 = ttnn.reshape(x, (1, 1, flat_M, x.shape[-1]))
    y2 = ttnn.linear(x2, weight, bias=bias, **kwargs)
    return ttnn.reshape(y2, (*leading, y2.shape[-1]))

out = linear_flatten_batch(x, weight, bias=bias)
```

Applied in `models/experimental/uniad/tt/ttnn_mha.py` and
`models/experimental/uniad/tt/ttnn_deformable_attention.py`.

## What the profile didn't find

The profile correctly identified the bottleneck but didn't suggest
the *fix*. The fix required:
- Knowing that `ttnn.linear` heuristic uses input shape
- Knowing that flattening leading dims is cheap (just a view) on TTNN
- Verifying PCC after the change

A profile tells you **where** time goes; deciding **what to change**
still requires understanding the op's config / heuristic. Read the
relevant ttnn source (`ttnn/cpp/ttnn/operations/matmul/matmul.cpp` or
similar) when an op's default config is suspect.

## Reuse

For other models, the same workflow applies:

1. Top-level wall-time profile → identify suspicious phase
2. Per-module Tracy profile → CSV
3. Recipe 1 → top ops by kernel time
4. Recipe 2 (for top op) → distribution by core count
5. Spot outliers → inspect `ATTRIBUTES`
6. Hypothesize fix (config / reshape / program_config)
7. PCC + perf verify
8. Repeat
