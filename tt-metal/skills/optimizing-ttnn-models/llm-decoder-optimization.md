# LLM Decoder Per-Device Optimization

How to squeeze on-device performance out of a *functionally correct*
TTNN LLM decoder (prefill + decode), without rewriting ttnn ops. Assumes
passing correctness tests already exist — they stay the correctness
floor and must be re-run against the optimized path.

This complements the general steps in this skill: data formats
(`step-01`), sharding (`step-03`), Metal Trace (`step-04`). It focuses on
the matmul / precision / data-movement decisions that dominate decoder
performance.

## Method

- Optimize **on-device** performance. For decode, always measure a
  **traced** execution run (see `step-04-metal-trace.md`). Note op/host
  gaps but still tune on-device work; tracing closes the gap.
- **Always use real model shapes** — never reduced shapes.
- Profile warmed **prefill and decode separately**.
- For precision/fidelity tuning use **real weights and recorded input
  activations**; synthetic data is not a representative signal.
- Treat `tt-perf-report` as a conversation, not an oracle: classify the
  bottleneck, try the applicable advice, keep what improves the target
  without unacceptable PCC/complexity cost, record why rejected advice
  was rejected. (Capture → `profiling-tt-metal`; column semantics + CLI
  → `analyzing-tt-profiles`.)
- "Sharding" is overloaded: **on-device sharding** (L1-sharded
  activations, DRAM-sharded weights, across one device's cores/DRAM
  banks) is in scope here. **Multi-chip sharding** (mesh mapper) is not —
  see `step-07-multi-device.md` and `tensor-parallel-llm.md`.

## Core rules

- Avoid data movement before tuning math: a slightly smaller core grid that avoids a reshard between ops can beat a faster individual op.
- Decode activations should generally stay **width-sharded in L1** across norm → attention → residual → MLP → output-projection boundaries.
- Prefill activations are large → usually **DRAM interleaved**, with 2D matmul program configs for big matmuls.
- Use **SDPA / FlashDecode** ops instead of hand-built attention when the model fits their contract.
- Explicitly set `memory_config`, `program_config`, and `compute_kernel_config` for important ops — defaults are often correct but suboptimal.
- Choose shard specs / core grids that divide tensor dims cleanly into tiles; padding in sharded paths wastes work and breeds bugs.

## Matmul choices

- **Decode** matmuls (small activation, large weight) are usually DRAM-bound → `ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`. Weights width-sharded in DRAM, activations/outputs width-sharded in L1 on the matching core grid.
- **Prefill** matmuls (large M, N) are usually compute-bound → `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` over a large 2D grid.
- **`in0_block_w` ≥ 2** whenever possible, and it must divide the tiled K dim. Higher is better until L1 pressure / correctness fails. There is a trade-off with core count for the shard spec — if the only valid `in0_block_w` is 1, prefer fewer cores to enable 2. Note DRAM-sharded matmuls have a **fixed compute-core count** (e.g. 12 on Wormhole) regardless of input/output shard core counts, which gives extra flexibility. Padding weights to enable `in0_block_w ≥ 2` can be worth it (changing the shard spec is usually preferable).
- **Output subblock** should usually be ≥ `2x1` or `1x2` when legal.
- **L1 OOM**: first increase the core count; if you can't, reduce `in0_block_w`, `out_subblock_h`, or `out_subblock_w` and keep the combo that preserves the most performance.
- If `tt-perf-report` says a matmul is DRAM-bound and it is **not** DRAM-sharded, trying DRAM-sharded is mandatory — resharding to make it work is usually worth it. If it's not a win, record that you tried and why.
- Call out explicitly any `in0_block_w` or subblock < 2 on a matmul that is a non-trivial fraction of runtime, and list what you tried to raise it.

## Precision & fidelity

Evaluate one tensor group at a time so regressions are assignable.

- Start: **BF16** activations + norms, **BFP8** attention/MLP weights.
- Try **BFP8 KV cache**; keep if PCC holds and perf/memory improve.
- Try **BFP4** for MLP FF1/FF3 (often tolerate it); FF2/down-projection is more sensitive — fall back on PCC evidence, not preference.
- **BFP8 weights → HiFi2** start (LoFi needs PCC evidence). **BFP4 weights → LoFi**. **BF16 weights / sensitive ops → HiFi4 or fp32 accumulation** where PCC demands.
- Activation size matters for CCLs — try **BFP8 activations** and check PCC / eval scores.

Full-model datatype *frontier* selection against top-1/top-5 accuracy is
a separate pass — see `datatype-sweep.md`. Hand it a clean optimized
baseline; don't do accuracy-gated frontier selection here.

### Compute-kernel configs (Wormhole starting points)

```python
compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)

compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)

compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
```

Use the architecture-appropriate config class for non-Wormhole targets.

## Advice policy

For every actionable `tt-perf-report` recommendation: try it; record
before/after latency, PCC, and any watcher/correctness issue; keep it if
it improves the target without unacceptable PCC/complexity; reject only
with evidence, then keep optimizing the rest. Don't suppress advice in
the report used to guide optimization — untried applicable advice is
remaining work, not a finished pass.

## Optimization checklist

- [ ] Decoder path fully traced, no host fallbacks
- [ ] Decode activations width-sharded in L1 across norm/attn/residual/MLP/output boundaries
- [ ] Prefill activations DRAM-interleaved; 2D matmul program configs for large prefill matmuls
- [ ] SDPA / optimized composite ops used instead of hand-built attention where the model fits
- [ ] `memory_config`, `program_config`, `compute_kernel_config` explicitly set for important ops
- [ ] Shard specs / core grids divide tensor dims cleanly into tiles; grids as large as model/HW allows
- [ ] DRAM-sharded decode matmuls
- [ ] Fused matmul-CCL ops used where possible (or profiled and discarded with evidence)
- [ ] MoE: routed active-expert path via `ttnn.sparse_matmul`, no dense all-expert runtime path
- [ ] Reduced precision/fidelity experiments done + documented with real weights/activations

## Final audit

- No unnecessary `InterleavedToSharded` / `ShardedToInterleaved` / `reshard` / `tilize` / `untilize` / `to_torch` / `from_torch` in the optimized runtime path.
- Decode **trace replay** still measures the optimized path, not a fallback.
- PCC re-checked for prefill + decode at the functional bar; paged KV cache + warmed trace replay still correct.
- Watcher still clean (`TT_METAL_WATCHER=10`).
- Program configs / compute-kernel configs described in the final summary.

## Code paths worth reading

- `models/tt_transformers/tt/model_config.py`: precision/fidelity, sharded activation configs, DRAM-sharded matmul helpers, prefill/decode program configs.
- `models/tt_transformers/PERF.md`: empirical precision/perf tradeoffs (Llama/Qwen/Mistral/Phi/Mixtral).
- `tech_reports/LLMs/llms.md` §4: best practices, matmul variants, DRAM-sharded matmul, perf-report interpretation.
- `models/common/modules/attention/attention_1d.py`, `mlp/mlp_1d.py`: reusable BFP8/BFP4, DRAM-sharded decode matmuls, SDPA + L1-sharded decode residual paths.
- `models/demos/gpt_oss/tt/experts/`, `topk.py`: routed MoE active-expert path with `ttnn.sparse_matmul`.
