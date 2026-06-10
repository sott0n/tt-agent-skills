# LLM Decoder Correctness: Defaults & Gotchas

The phases in `llm-model-bringup.md` get a decoder running. This file
collects the defaults and gotchas that decide whether it is *correct and
shippable*. Acceptance bar: **PCC ≥ 0.995** for both prefill and decode
(stricter than the 0.99 module bar), and at least one **real-weight**
test passing that bar before the decoder is done.

## TTNN correctness defaults

- Start with `ttnn.bfloat16`, `ttnn.TILE_LAYOUT`, `ttnn.DRAM_MEMORY_CONFIG`; optimize layout/precision only after PCC is stable.
- Transpose 2D torch linear weights before `ttnn.linear` (unless the helper already does it).
- Reshape norm weights so they broadcast over hidden width.
- Derive Q/K/V head reshapes from config, including GQA/MQA KV expansion.
- Q/K weights may need an HF→TTNN RoPE permutation (`reverse_permute`); check the module/test contract.
- Move all weight conversion, reshaping, dtype selection, `ttnn.as_tensor`, and cache construction into a setup/`from_state_dict` boundary. Keep runtime prefill/decode free of `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback except explicit test boundaries — **host fallback hidden in a helper is still fallback**, so inspect wrappers too.

## Paged KV cache (use this as the final path)

Non-paged cache is not worth the time for a real bringup.

- Prefill: `ttnn.experimental.paged_fill_cache`.
- Decode: `ttnn.experimental.paged_update_cache` + `ttnn.transformer.paged_scaled_dot_product_attention_decode`.
- Decode **current positions must be tensors**, not Python scalars/lists — host state breaks trace-safety.
- Test page-table permutations, nonzero slots, and random current positions to catch address/indexing bugs.
- Hybrid (sliding + full) attention: inspect **per-layer** page-table routing and cache specs — don't assume one page-table shape covers every layer.
- Gemma-style **shared-KV** layers may skip K/V projection + cache update; the consumer must still read the intended source cache. DeepSeek-style **MLA** cache updates can alias between prompt/speculative lanes — inspect masks and lane routing.

## Prefill / decode shapes

- Prefill commonly `(1, batch=1, seq_len, hidden)`; decode commonly `(1, seq_len=1, batch, hidden)`.
- **Test the full supported sequence / context length**, not tiny smoke shapes. Only reduce if measured L1/DRAM capacity forces it, and then record the capacity probe (command, failure signature, or byte calculation). Do not accept "tractability" or runtime cost as capacity evidence.
- Sliding-window models: include lengths *around the window boundary*. If full and sliding layers differ only by mask/window config, use one parameterized implementation + tests for both modes.

## Synthetic weights from real stats (CI without HF downloads)

- Treat the real HF `state_dict` as the canonical **key + shape** contract.
- For each tensor record at least name, shape, dtype, mean, std; generate synthetic weights *deterministically* from those stats in the test.
- **Always use the real config and shapes — never shrink the model** to make it tractable.
- Synthetic input activations should approximate the distribution *entering the decoder* (post-embedding/post-norm), not arbitrary huge randoms.
- Still include ≥1 real-weight test that passes PCC ≥ 0.995 at least once.

## MoE decoders

- Validate the **real router/gate + active-expert path end-to-end**, not just gate or experts in isolation.
- Non-Galaxy default = GPT-OSS active-expert pattern: router/top-k scores as a sparsity tensor, `ttnn.sparse_matmul` for gate/up/down projections, score weighting, then a model-appropriate reduce over experts. Avoid Galaxy-only fused MoE paths unless you have direct hardware/op evidence.

## Debugging low PCC

Split the decoder into components and check HF parity at each:
input norm → QKV → RoPE → SDPA → WO + residual → post-attention norm →
MLP / router+experts → final residual. Raise fidelity where it helps,
simplify the failing shape, and keep narrowing until the cause is
understood (a tt-metal bug → minimal reproducer or on-branch workaround
with evidence).

## Watcher is part of "done"

- Run correctness tests with `TT_METAL_WATCHER=10` (don't skip asserts). It catches asserts, invalid NoC coords/addresses, CB out-of-bounds transactions, L1 / stack overflow, and hardware faults.
- Run watcher and profiler/DPRINT in **separate** runs — they contend for debug resources.
- A clean run may still contain normal attach/dump/kernel-id/detach lines; it must not contain fatal watcher exceptions. If you believe a hit is a false positive, leave evidence (the exact line, why it's benign, the issue/commit proving it known).

## Repo references worth reading

- Reusable modules: `models/common/modules/attention/attention_1d.py`, `mlp/mlp_1d.py`, `rmsnorm/rmsnorm_1d.py`, `rope/rope_1d.py`, `lazy_weight.py`.
- Production composition + trace/vLLM contracts: `models/tt_transformers/tt/{decoder,generator,model,generator_vllm}.py`.
- MoE active-expert pattern: `models/demos/gpt_oss/tt/experts/`, `models/demos/gpt_oss/tt/topk.py`.
- Many norms / sliding / shared-KV: `models/demos/gemma4/tt/`.
- Dense-vs-MoE decoder kinds + MLA + row-sharded cache: `models/demos/deepseek_v3/tt/`.
- Conventions: `tech_reports/LLMs/llms.md`, `tech_reports/LLMs/vLLM_integration.md`.

## Trace & profiling (cross-links — don't reinvent here)

- The final decode pass should run **traced** TTNN execution. Capture/replay mechanics, program-cache warmup, and trace debugging live in `optimizing-ttnn-models/step-04-metal-trace.md`.
- To capture a profile see `profiling-tt-metal`; to interpret the `ops_perf_results_*.csv` / `tt-perf-report` output see `analyzing-tt-profiles`.
