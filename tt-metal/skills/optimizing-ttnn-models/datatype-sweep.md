# Full-Model Datatype Sweep (accuracy-gated precision selection)

Choose the **fastest** weight / activation / KV-cache / CCL datatype
configuration that still meets a stated **full-model accuracy bar**. This
is a distinct pass from per-device optimization (`llm-decoder-optimization.md`):
that pass settles parallelism, tracing, sharding, program configs, and
removes host boundaries; this pass owns the final datatype frontier and
the compute-fidelity follow-on. Start from a clean optimized baseline.

The expensive **source of truth is full-model top-1 / top-5 accuracy**.
Decoder-layer PCC and component timing only order candidates and explain
surprises. If the user gives no bar, default to **top-1 ≥ 90%, top-5 ≥
98%** (keep top-100 at the existing readiness expectation).

## Baseline

Record, with provenance, before sweeping:

- A readiness reference (e.g. AIME24 with chat template, `--gen-len 100`); regenerate to ≥100 generated tokens if shorter, unless a real capacity limit prevents it.
- Baseline top-1 / top-5 / top-100, prefill TTFT, decode t/s/u.
- The current dtype policy for weights, norms, residual stream, CCL activations, KV cache, logits, sampling, and MoE routing.

## Coarse search (do this first)

Usually captures most of the win with few full-model runs:

1. BFP8 **KV cache** as a yes/no switch.
2. BFP8 **CCL / residual-transfer activations** as a yes/no switch.
3. **BFP4 for all eligible inner-layer BFP8 matmuls**, excluding the first and last layer by default.
4. If full-model accuracy fails, **restore the highest-risk groups first** until top-1 and top-5 pass.
5. Once a passing inner-layer config is found, optionally extend the surviving choices to the first/last layer — keep only if accuracy still passes.

Restore order (most-sensitive first) for dense MLP / MoE blocks:

1. FF2 / down / expert-down projections
2. QKV / attention input projections
3. WO / attention output projection
4. FF1+FF3 / gate+up / expert gate+up projections

A heuristic, not a law — adjust for the architecture (MLA, shared
experts, gated attention, fused projections), profiler evidence, and
observed failures. When backing out a failed BFP4 trial, restore to BFP8
first unless BFP8 itself is the failing precision for that group.

## MoE policy

- Expert gate/up follow the FF1+FF3 policy; expert down follows FF2; shared experts follow dense MLP policy unless evidence says otherwise.
- Sweep routed sparse expert matmuls on the **active-expert path**, not dense all-expert debug paths.
- Router / weighting numerics are more sensitive — **keep router logits, top-k selection, routing scores, gate weighting, and expert-reduction weighting out of the coarse BFP4 sweep** (BFP8 only, low priority; they're rarely the bottleneck).
- Record route/top-k distribution or active-expert count when it can vary — the perf impact depends on how many experts actually run.

## Evaluate every kept candidate

Validate each kept config with full-model accuracy. Record: config id +
precision-config path; weight dtype groups + layer ranges; activation /
CCL / KV-cache choices; top-1 / top-5 / top-100 + token count + reference
path; TTFT + decode t/s/u; pass/fail vs the bar; exact command, commit,
hardware, mesh, env notes.

## Compute fidelity (after the datatype frontier)

A separate decision made **after** the frontier is set. Use per-decoder
`tt-perf-report` output (capture → `profiling-tt-metal`, interpret →
`analyzing-tt-profiles`) to guide it:

- BFP4 weights → usually LoFi.
- BFP8 weights → usually HiFi2.
- BF16 weights / accuracy-sensitive ops → HiFi4, or fp32 dest accumulation.

## Final selection

- Pick the **fastest** config that satisfies the bar. Within measurement noise, prefer the simpler/safer one.
- Make the selected config the model default, with a simple switch back to the safe baseline.
- Run qualitative generation if the dtype changes are large or top-1 is near the threshold; back off changes until it reads well.
- If no lower-precision config passes, keep the baseline — and leave evidence the sweep actually tested the likely wins.

A Pareto view helps communicate the tradeoff: plot every evaluated config
(decode t/s/u vs top-1, and vs top-5), draw the non-dominated frontier,
mark the selected config and the minimum-accuracy line.
