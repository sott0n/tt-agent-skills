# Metal Trace: Program-Cache Warmup & Debugging

`step-04-metal-trace.md` covers the capture/replay APIs and the basic
DRAM/L1 patterns. This file covers what makes tracing actually work on a
real model: program-cache warmup, the generator decode split, what must
stay outside capture, multi-chip/CCL traces, and how to debug failures.

Trace replay is usually the difference between a fast device kernel set
and a usable inference path — eager Python/TTNN dispatch between many ops
can dominate decode.

## Mental model

Capture records a **static command sequence** over **stable device
tensors**; replay reuses it without per-op host dispatch. Host-originated
input changes happen *before* replay by copying into those stable
tensors. Outputs produced by ops inside capture are fine; **host writes,
host reads, and host synchronization while the trace is open are not.**

Trace-safe shape:

1. Build all weights, caches, page tables, semaphores, persistent CCL buffers, lazy module state **before** capture.
2. Create host-side input tensors (`device=None` if useful).
3. Allocate stable device input tensors before capture.
4. **Warm-compile** one forward with the exact shapes + mode of capture.
5. `begin_trace_capture` → device-only forward → `end_trace_capture`.
6. Per replay: update stable inputs *outside* capture, then `ttnn.execute_trace`.

## Program-cache warmup (the #1 cause of capture failures)

Trace capture **cannot compile programs**. A program-cache miss inside
capture forces a kernel build, which issues a host→device write and
aborts with `Writes are not supported during trace capture`. So every op
in the traced region must already be compiled with the **exact**
program-cache signature it will have during capture.

- Warm with the same shapes, dtypes, layouts, memory configs, and mode; the warm call must drive the identical op sequence and code path.
- The signature can include **arguments you wouldn't expect**: e.g. the integer `begins`/`ends`/`step` of `ttnn.slice` are compile-time constants baked into the program hash — slicing at a different offset/length is a *different program* needing its own warm-up. When in doubt, warm with the same argument **values**, not just the same tensor shapes.
- If you still hit an unexpected miss during capture, warm again **immediately** before `begin_trace_capture` (re-run the exact forward once, then capture with nothing in between). An op's signature can depend on transient device state (e.g. free L1), so an earlier warm-up may no longer match (see tenstorrent/tt-metal#46533).

## Generator decode split

Don't trace the high-level generator method. Split it:

- `prepare_decode_inputs_host(...)` → host TTNN tensors / torch values for token ids, positions, RoPE indices/tables, page tables, masks.
- `ttnn.copy_host_to_device_tensor(...)` → refresh the stable trace inputs before replay.
- `decode_forward_from_ttnn_inputs(...)` (a.k.a. `ttnn_decode_forward`) → device-only call used for both warm-compile and capture.
- `decode_next_token_traced(...)` → refresh inputs, `execute_trace`, read back only what the caller needs.

Page-table and position tensors are trace inputs — refresh their stable
buffers before replay if they can change. If sampling writes the next
token back into the input buffer, make that a **traced sampling path or a
second trace**, not a host readback inside the model trace.

## Keep outside capture

- `ttnn.from_torch(..., device=...)`, `as_tensor(..., device=...)`, `to_device`, `copy_host_to_device_tensor`.
- `ttnn.to_torch`, `.cpu()`, `get_device_tensors` + host conversion, full-logits host composition.
- `synchronize_device`, event waits/synchronization, explicit reads.
- Lazy weight / model-cache loads, first-use module init.
- Resetting KV cache, page tables, semaphores, sampling state.
- **Any Python decision that changes the op sequence, shape, memory config, or code path** — decide before capture and bind the mode at construction time.

## Multi-chip / CCL traces

- Set fabric config **before** opening the mesh; create CCL semaphore managers before capture.
- Warm-compile with the same mesh shape, page-table shape, and CCL topology intended for capture.
- If an op uses persistent output buffers and replay needs stable addresses, ensure the first allocation happens before capture (a same-code-path warm pass usually handles this). Reuse the same input allocations for capture and replay.
- If a single decoder layer traces but the **full model** doesn't, inspect terminal work separately: final distributed norm, hidden gathers, LM head, logits gather, sampling, argmax, token readback. (See `tensor-parallel-llm.md` for the mesh layout side.)

## Debugging trace failures

Fatal signatures:

- `Writes are not supported during trace capture`
- `Reads are not supported during trace capture`
- `Event Synchronization is not supported during trace capture`
- Replay returns **stale** outputs / ignores updated inputs
- Replay passes once and fails on repeated execution

Bisect by adding flushed markers around coarse blocks:

```python
print("BEGIN_TRACE", flush=True)
trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
print("TRACE_MODEL_START", flush=True)
out = model.decode_forward_from_ttnn_inputs(...)
print("TRACE_MODEL_DONE", flush=True)
ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
```

- **Write fatal**: check for host input creation, lazy weights, cache loads, page-table refresh, sampling-state updates, semaphore resets, first-use CCL buffers, or a `copy_host_to_device_tensor` hidden in a helper. Also suspect a program-cache miss — confirm with `device.set_program_cache_misses_allowed(False)` around capture (a miss then fails as `"<Op>: program cache miss occurred, but cache misses are forbidden"`, naming the offending op), then fix via warm-up.
- **Stale inputs**: compare the tensors captured by the model to the tensors refreshed before `execute_trace`. Bind model-side trace inputs before capture and refresh exactly those buffers.

## Validation evidence

- Correctness before and after tracing against the same reference.
- Repeated-replay determinism across several executions.
- **Updated-input replay test**: run two decode steps with different token + current-position values, inspect the persistent trace input tensors, and assert the output/logits changed. Cover unchanged *and* changed page tables.
- No host fallback in the captured path; warmed replay timing (prefill/decode separately). Profile the traced region — capture via `profiling-tt-metal`, interpret via `analyzing-tt-profiles`.
