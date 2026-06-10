---
name: serving-ttnn-with-vllm
description: "Serves an existing TTNN full model through vLLM on Tenstorrent hardware: the thin generator_vllm.py adapter, TT vLLM plugin registration, the run_vllm_server readiness runner, async-decode / on-device-sampling serving optimization, and serving-path correctness + performance evidence. Use when adding vLLM serving to a working TTNN model+generator, debugging the serving adapter / KV-cache ownership / plugin registration, or measuring/optimizing serving TTFT, ITL, and per-user decode t/s/u."
---

# Serving TTNN Models with vLLM

This stage starts from a **working TTNN full model + generator** (`tt/model.py`,
`tt/generator.py`, both passing generator-level readiness) and makes it
usable through the shared vLLM serving path. Bringing up the model and
generator is upstream (`porting-models-to-ttnn`); this skill owns the
adapter, plugin registration, serving checks, and serving-path
performance. You may make small, evidence-backed changes to
`model.py` / `generator.py` when the adapter exposes a real contract gap —
but don't turn this back into full-model bringup.

## The adapter — `tt/generator_vllm.py`

Keep it **thin**: delegate to the existing generator's low-level
`prefill_forward` / `decode_forward` wherever possible; don't duplicate
logic that already lives in `model.py` / `generator.py`. Read
`tech_reports/LLMs/vLLM_integration.md` and
`models/tt_transformers/tt/generator_vllm.py` first.

**KV-cache ownership** — preserve two modes:

- standalone readiness/generator mode: the generator owns cache allocation + reset;
- vLLM mode: `allocate_vllm_kv_cache` creates the cache and the adapter passes that exact cache through the low-level API.

Not every cache must be vLLM-owned. vLLM owns the **attention** KV cache;
constant-size recurrent / linear-attention state (conv windows, SSM /
gated-delta state) can live in the model and be carried across decode
steps. Route sliding-window and full-attention layers through vLLM's
**hybrid attention** infrastructure (per-layer KV-cache specs + block
tables) rather than forcing one uniform attention type.

Make prompt lengths, page tables, decode positions, batch dims,
trace-side state, and on-device sampling explicit. The serving decode
pass must drive the generator's **traced** decode path, not an eager
fallback (trace work → `optimizing-ttnn-models/metal-trace-debugging.md`).

## Plugin registration

vLLM discovers TT models from a hardcoded list. Add a registration call:

```text
vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models()
```

```python
_register_model_if_missing(
    ModelRegistry, "TT<Arch>ForCausalLM", "<dotted.module.path>:<ClassName>")
```

Without this the server rejects the architecture at startup.

## Serving readiness runner

Use the shared runner — it owns launch, health polling, check execution,
and shutdown:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/<path> --hf-model <hf-id-or-local-path> \
  --mesh-device <N150|N300|T3K|TG> \
  --max-num-seqs <int> --max-model-len <int> \
  --sampling-profile <full|smoke> \
  --tt-config '{"trace_region_size": <bytes>, "fabric_config": <mode>}'
```

`--stages` accepts `serve`, `sampling`, `qualitative`, `benchmark`
(default = full launch→check→shutdown). To iterate, hold a server open
with `--stages serve`, then attach checks from another shell with
`--stages sampling --server-url http://localhost:8000`.

Stages:

- **sampling** — canonical TT plugin pytest suite against the live server. `full` = whole suite; `smoke` = sanity subset for slow loops.
- **qualitative** — greedy + sampled completions for `models/common/readiness_check/vllm_prompts.txt`; judge coherence, topic, repetition, gibberish, wrong-language drift.
- **benchmark** — synthetic workload; records TTFT P50/P99, ITL P50/P99, aggregate output throughput, mean per-user decode t/s/u.

The runner enforces on-device sampling (`sample_on_device_mode: all`).
Use `--sampling-profile full` for final evidence; `smoke` for inner-loop
(and, for batch-1 MoE bring-up, `smoke` is acceptable as the final gate —
`full` can be skipped, it's very slow there).

## Serving-decode optimization

When direct traced generator decode is fast but **serving** decode is
slower, treat the gap as orchestration overhead before retuning decoder
math. Implement the vLLM async split before advertising it:

- `decode_forward(..., read_from_device=False)` returns device tensors;
- `read_decode_output(..., async_read=True)` does the minimal deferred read;
- `process_decode_output_host(...)` does host formatting.

Only set `supports_async_decode=True` after this passes with decode trace
enabled. Replay via `ttnn.execute_trace(..., blocking=False)` over
persistent trace inputs. For `sample_on_device_mode=all`, keep sampling
on device; if the model trace returns sampler-ready logits, pass them as
prepared rather than reading full logits on the host. **Remove host
greedy/top-1 argmax fast paths or prove they're unused by the measured
benchmark.** Avoid copying a full page table every token when unchanged —
but add a stale-input test before reducing any token/position/page-table
refresh. Leave prefix caching `False` unless implemented and tested.

Benchmark with the exact same runner, prompt/output lengths,
`max_num_seqs`, model length, mesh, TT config, and sampling mode as the
comparison; compare to the canonical same-machine implementation when
available.

## Gotchas

- If vLLM crashes mid-run, kill leftover `EngineCore` / `vllm.entrypoints` processes before retrying — they can hold chip locks even after `tt-smi -r`.
- **Reproducibility-only** sampling failures are out of scope *when they are the only failures* (e.g. `test_top1_is_greedy`, `test_topk`, `test_*_seed_*`, `test_*_mixed_batch`). Correctness failures, missing/wrong logprobs, crashes, and gibberish stay in scope.
- Record the working invocation (incl. `--max-model-len`, `--tt-config`, workload, env vars). Use typed runner flags; keep `--additional-server-args` for uncommon flags only.

## Evidence to leave

- Generator readiness baseline used before adding vLLM.
- Adapter class, the low-level generator methods it delegates to, KV-cache ownership contract.
- Plugin registration path + architecture name.
- Exact successful `run_vllm_server` invocation.
- Capability flags with evidence: no unproven `supports_async_decode=True`, no prefix-caching claim without tests, on-device sampling verified for the measured mode.
- Sampling results (reproducibility-only failures separated from real ones), qualitative verdict, benchmark workload config.
- Serving TTFT P50/P99, ITL P50/P99, aggregate output throughput, mean per-user decode t/s/u.

## References

| Topic | Path |
|---|---|
| vLLM integration guide | `tech_reports/LLMs/vLLM_integration.md` |
| Serving readiness runner | `models/common/readiness_check/run_vllm_server.py` |
| Qualitative prompts | `models/common/readiness_check/vllm_prompts.txt` |
| Generator contract | `models/common/readiness_check/contract.py` |
| Reference generator | `models/tt_transformers/tt/generator.py` |
| Thin vLLM adapter | `models/tt_transformers/tt/generator_vllm.py` |
| Model registration | `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models` |

## Related skills

- `porting-models-to-ttnn` — bring up the model + generator this stage serves.
- `optimizing-ttnn-models` — per-device decode tuning (`llm-decoder-optimization.md`) and trace work (`metal-trace-debugging.md`).
- `profiling-tt-metal` / `analyzing-tt-profiles` — capture and interpret serving-path profiles.
