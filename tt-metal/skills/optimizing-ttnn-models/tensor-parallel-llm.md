# Tensor / Expert Parallelism for LLMs (multi-chip)

How to parallelize a working **single-chip** TTNN decoder across a device
mesh. This is the LLM tensor-parallel (TP) / expert-parallel (EP)
counterpart to `step-07-multi-device.md`, which covers data-parallel
batch scaling (the right tool for CNNs and single-chip-fits models).

Read `tech_reports/LLMs/llms.md` §3.3 Multi-Device first. Goal: the
**fastest** multi-chip implementation, not the most convenient given the
single-chip starting point. Two competing objectives — apply as much
FLOPs + memory bandwidth as possible, and minimize the data movement
that requires. After parallelizing, run a per-device optimization pass
(`llm-decoder-optimization.md`); parallelism and on-device tuning
together are what get high utilization.

## Plan before coding

- For 1D meshes up to 8 chips, **1D TP** is the starting point. For Galaxy-class meshes, make a model-specific **2D plan**.
- Build a table of every tensor's shape / config / shard spec and how each changes under the scheme (usually divide by the TP factor along that mesh dim; sometimes implicit/explicit padding). Keep it in your report.
- **Decoder I/O contract**: the decoder's output layout must match its *input* layout so layers stack — but it need **not** match the single-chip I/O. Chip-sharded activations at the decoder boundary are fine and often desirable. Do the entry/exit resharding once at the whole-model boundary (adjust the *test* to feed the right sharding), not at every layer. If there are multiple decoder kinds, their input/output shardings + dtypes must be mutually compatible.

## Dense 1D TP default (≤ 8 devices)

- **WQKV**: column/output sharding → each device owns local Q/K/V heads.
- **KV cache**: per-device local KV-head cache (paged if the baseline is paged).
- **SDPA**: local to the local heads.
- **WO**: row/input sharding over concatenated local heads, then reduce-scatter / all-reduce to restore residual layout.
- **W1/W3**: column/output sharding over the intermediate dim.
- **W2**: row/input sharding over the intermediate dim, then reduce-scatter / all-reduce.
- Avoid a design that needs an extra gather/reshard *between* stacked layers unless evidence shows it's faster overall.

## Distributed RMSNorm

When hidden activations are sharded across the normalized dim, local
RMSNorm computes wrong statistics. Use the primitive trio:

1. `ttnn.rms_norm_pre_all_gather`
2. `ttnn.experimental.all_gather_async` / `ttnn.all_gather` for the stats
3. `ttnn.rms_norm_post_all_gather`

**Correct** RMSNorm is mandatory; distributed RMSNorm is not always
mandatory — a faster replicated-activation stream + local RMSNorm is fine
if it preserves the decoder chain layout and measures faster. Reusable
implementations: `models/common/modules/rmsnorm/rmsnorm_{1d,2d}.py`,
`models/demos/deepseek_v3/tt/rms_norm/distributed_rms_norm.py`,
`models/tt_transformers/tt/ccl.py`.

## MoE / expert parallelism

- For TP ≤ 8, default = run each **gate-selected active expert** with TP. Keep the active-expert path from the single-chip baseline; don't run every expert densely as the final path.
- Non-Galaxy → GPT-OSS generic experts path: router/top-k scores as a **sparsity tensor**, `ttnn.sparse_matmul` for gate/up/down, score weighting, `ttnn.sum` (or model-appropriate reduce) over experts, *then* any TP/EP collective.
- Treat expert mapping, routing-weight layout, shared-expert placement, sparse weight layout, DRAM/L1 memory configs, semaphores/preallocated buffers, and final residual layout as **correctness contracts**. Validate the full router → sparse projection → score weighting → expert reduce → collective sequence, not gate and experts in isolation.
- Galaxy 4×8 throughput-experts (dispatch tokens to expert owners on one axis, replicate/TP on the other): only if DRAM capacity for *all* layers + full KV cache at max sequence length is proven. Otherwise document the fallback (ordinary TP active experts / 2D TP / EP without replication / hybrid).

## 2D mesh planning (Galaxy)

Don't blindly flatten to TP=8. Choose which axis owns TP / EP / sequence
parallelism / replication; identify which collectives cross rows vs
columns; estimate activation + weight + expert + KV-cache memory with all
layers loaded; compare communication volume against the 1D alternative;
record why the chosen strategy improves single-user latency. GPT-OSS
`MeshConfig` is the preferred shape for expressing this plan.

## Correctness validation

- Compare multi-chip TTNN output to the **single-chip TTNN baseline** with identical synthetic/real weights, inputs, page tables, and positions. This isolates sharding/collective bugs from HF-vs-TTNN numerical differences.
- If PCC is near threshold, split into component comparisons: input RMSNorm → QKV → RoPE → SDPA out → WO + reduction → post-attn residual → post-attn RMSNorm → router/top-k → active experts → W2/down + reduction → final residual.
- **Check layouts and collectives before changing precision.** Most multi-chip bugs are: wrong sharding, wrong gather/reduce axis, bad padding/slicing, repeated bias after all-reduce, wrong local head count, wrong local KV-cache shape, or mismatched input/output residual layout.

## Runtime gotchas

- Configure the fabric that matches the mesh/topology **before** opening the mesh — prefer repo pytest `device_params`, or `ttnn.set_fabric_config(...)` before `ttnn.open_mesh_device(...)`. For 8-chip Wormhole / T3K 1D TP: `FABRIC_1D_RING` before mesh open and `ttnn.Topology.Ring` for CCL ops.
- Treat CCL failures from a raw `open_mesh_device` as *setup* evidence, not hardware evidence, until the same case fails with the correct fabric config + matching CCL topology.
- **Reset all devices** after any failed multi-chip run.
- No `from_torch` / `to_torch` / host reads/writes / tensor allocation after trace capture inside measured prefill/decode (mesh trace capture/replay → `step-04-metal-trace.md`).
- Always test with **Watcher** when using async CCLs — semaphore mistakes cause data corruption or hangs easily. Run watcher separately from the profiler.
- Make CCL semaphore ownership explicit; reuse generic CCL managers (`models/demos/gpt_oss/tt/ccl.py::CCLManager`, `models/common/modules/tt_ccl.py`) unless the model already has a better local abstraction.
- Pad CCL-sensitive hidden dims deliberately and slice at a documented boundary. If a bias is applied before an all-reduce, prove it isn't applied once per TP shard.

## Reference code

Prefer the GPT-OSS structured approach + `models/common` TTTv2 modules:

- `models/demos/gpt_oss/config.py`: `ModeConfig` / `MeshConfig`, `column_parallel` / `row_parallel` / `sequence_parallel` mapper helpers, CCL helpers.
- `models/demos/gpt_oss/tt/ccl.py`: generic `CCLManager` with ping-pong semaphores.
- `models/common/modules/attention/attention_1d.py`, `mlp/mlp_1d.py`, `rmsnorm/rmsnorm_{1d,2d}.py`, `tt_ccl.py`.
- `models/demos/gpt_oss/tt/experts/`, `topk.py`: routed MoE active-expert path.

`models/demos/llama3_70b_galaxy/` is highly optimized — useful as
**evidence** for where collectives belong (Q/K/V split → local SDPA →
gather heads → WO → reduce; W1/W3 sharded → gather for W2 → reduce), but
too model-specific to copy. Avoid copying its custom `llama_rs_*` ops,
32-chip asserts, prefetcher control flow, or env-var topology switches.
