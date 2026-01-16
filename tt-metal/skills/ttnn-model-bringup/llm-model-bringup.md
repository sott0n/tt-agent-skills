# LLM Model Bringup Guide

This guide covers specific steps for bringing up Large Language Models (LLMs) on Tenstorrent devices.

## Prerequisites

- Access to TT-Hardware ([Buy TT-Hardware](https://tenstorrent.com/hardware/wormhole))
- Knowledge of PyTorch and transformers
- Familiarity with [TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html) and [TT-NN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html)
- Completed general model bringup steps 1-3 (Reference Model Analysis, Operator Mapping, Per-Operator Testing)

## Reference Implementation

For transformer-based models, use the following as reference:
- **Primary Reference**: `models/tt_transformers` - Reference implementation for transformer models
- **Model Demos**: `models/demos/` - Example implementations (Llama, Falcon, etc.)

## Device Memory Considerations

| Device | Memory | Max Model Size (BFP8) |
|--------|--------|----------------------|
| Wormhole n150 | 12 GB | ~12B parameters |
| Wormhole n300 | 24 GB | ~24B parameters |
| TT-LoudBox (4x n300) | 96 GB | Multiple models or larger models |

**Recommendations:**
- Start with a single device for simpler bring-up if the model fits
- Use a smaller version of the model that fits on a single device first
- Scale up in size and across devices after initial validation

## LLM-Specific Architecture

### Decode vs Prefill Stages

**Decode Stage:**
- Supports batch=32 in llama3 implementation
- Each row is a separate user in 32x32 tiles
- Memory-bound operation

**Prefill Stage:**
- Supports batch=1
- Rows map to different input tokens
- Compute-bound (multiple batches don't benefit performance)

## Systematic Component-wise LLM Bring-Up

### Phase 1: Decode Stage Modules

Bring up decode stage modules first, implementing each module separately.

#### Standard LLM Modules

| Module | Description | Reference |
|--------|-------------|-----------|
| RMSNorm / LayerNorm | Normalization layers | `ttnn.rms_norm`, `ttnn.layer_norm` |
| RotaryEmbedding (RoPE) | Position encoding | Custom implementation |
| Attention | Multi-head attention | `ttnn.transformer.*` |
| MLP / FFN | Feed-forward network | `ttnn.linear`, `ttnn.gelu` |

#### Step-by-Step Module Bring-Up

1. **Implement module in TT-NN**
   ```python
   def ttnn_rms_norm(hidden_states, weight, *, eps=1e-6):
       return ttnn.rms_norm(hidden_states, weight=weight, epsilon=eps)
   ```

2. **Create unit test with model dimensions**
   ```python
   def test_rms_norm(device):
       torch.manual_seed(0)

       # Model dimensions
       batch_size = 32
       seq_len = 1  # Decode mode
       hidden_size = 4096

       # Create input
       torch_input = torch.randn(batch_size, seq_len, hidden_size)
       torch_weight = torch.randn(hidden_size)

       # Reference
       torch_output = torch_rms_norm(torch_input, torch_weight)

       # TTNN
       ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
       ttnn_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

       ttnn_output = ttnn_rms_norm(ttnn_input, ttnn_weight)

       output = ttnn.to_torch(ttnn_output)
       assert_with_pcc(torch_output, output, 0.999)
   ```

3. **Feed random data activations with real weights**

4. **Verify PCC matches reference output**

### Phase 2: Compose Modules

Compose all modules into higher-level components:

1. **Single Layer Decoder**
   ```python
   def decoder_layer(hidden_states, *, parameters, config):
       # Pre-attention norm
       residual = hidden_states
       hidden_states = ttnn.rms_norm(hidden_states, weight=parameters.input_layernorm.weight)

       # Self-attention
       hidden_states = attention(hidden_states, parameters=parameters.self_attn, config=config)
       hidden_states = ttnn.add(residual, hidden_states)

       # Post-attention norm
       residual = hidden_states
       hidden_states = ttnn.rms_norm(hidden_states, weight=parameters.post_attention_layernorm.weight)

       # MLP
       hidden_states = mlp(hidden_states, parameters=parameters.mlp)
       hidden_states = ttnn.add(residual, hidden_states)

       return hidden_states
   ```

2. **Full Decoder (Stack of Layers)**
   ```python
   def decoder(hidden_states, *, parameters, config):
       for i in range(config.num_hidden_layers):
           hidden_states = decoder_layer(
               hidden_states,
               parameters=parameters.layers[i],
               config=config
           )
       return hidden_states
   ```

### Phase 3: Decode Mode Implementation

1. **Implement decode mode first**
   - Use decode to run prefill initially
   - Test model configuration without dedicated prefill

2. **Create full model test with real inputs**
   ```python
   def test_decode_mode(device, model, tokenizer):
       # Input text
       prompt = "Hello, how are you?"
       input_ids = tokenizer.encode(prompt, return_tensors="pt")

       # Run decode
       output_ids = model.generate(input_ids, max_new_tokens=20)
       output_text = tokenizer.decode(output_ids[0])

       print(f"Generated: {output_text}")
   ```

### Phase 4: Accuracy Validation

1. **Teacher Forcing Validation**
   - Run same inputs through reference and TT-NN models
   - Compare hidden states at each layer

   ```python
   def validate_with_teacher_forcing(torch_model, ttnn_model, input_ids):
       # Get reference outputs
       with torch.no_grad():
           torch_outputs = torch_model(input_ids, output_hidden_states=True)

       # Get TTNN outputs
       ttnn_outputs = ttnn_model(input_ids, output_hidden_states=True)

       # Compare each layer
       for i, (torch_h, ttnn_h) in enumerate(zip(
           torch_outputs.hidden_states,
           ttnn_outputs.hidden_states
       )):
           pcc = calculate_pcc(torch_h, ttnn.to_torch(ttnn_h))
           print(f"Layer {i}: PCC = {pcc:.6f}")
   ```

2. **Token Generation Comparison**
   - Generate tokens from both reference and TT-NN models
   - Input reference tokens into both models in next iteration
   - Measure top1/top5 accuracy

   ```python
   def compare_token_generation(torch_model, ttnn_model, input_ids, max_tokens=100):
       torch_tokens = input_ids.clone()
       ttnn_tokens = input_ids.clone()

       top1_matches = 0
       top5_matches = 0

       for _ in range(max_tokens):
           # Reference prediction
           with torch.no_grad():
               torch_logits = torch_model(torch_tokens).logits[:, -1, :]
           torch_next = torch_logits.argmax(dim=-1)
           torch_top5 = torch.topk(torch_logits, 5, dim=-1).indices

           # TTNN prediction (using reference tokens for teacher forcing)
           ttnn_logits = ttnn_model(torch_tokens)
           ttnn_next = ttnn_logits.argmax(dim=-1)

           # Check accuracy
           if ttnn_next == torch_next:
               top1_matches += 1
           if torch_next in torch_top5:
               top5_matches += 1

           # Update for next iteration
           torch_tokens = torch.cat([torch_tokens, torch_next.unsqueeze(0)], dim=1)

       print(f"Top-1 Accuracy: {top1_matches / max_tokens * 100:.2f}%")
       print(f"Top-5 Accuracy: {top5_matches / max_tokens * 100:.2f}%")
   ```

3. **Output Verification**
   - Verify tokens are meaningful and coherent
   - Compare similarity to reference model tokens
   - Note: Due to floating point differences, tokens may not be exact matches

### Phase 5: Prefill Implementation

1. **Bring up layer-by-layer** (similar to decode)
2. **Run entire model** including prefill and decode
3. **Optimize prefill** for compute-bound characteristics

```python
def prefill(input_ids, *, parameters, config):
    """Prefill stage - process all input tokens at once"""
    batch_size, seq_len = input_ids.shape

    # Embeddings
    hidden_states = embeddings(input_ids, parameters=parameters.embed_tokens)

    # Process through all layers
    for i in range(config.num_hidden_layers):
        hidden_states = decoder_layer(
            hidden_states,
            parameters=parameters.layers[i],
            config=config,
            is_prefill=True,  # Flag for prefill-specific optimizations
        )

    # Final norm
    hidden_states = ttnn.rms_norm(hidden_states, weight=parameters.norm.weight)

    return hidden_states
```

## Performance Optimization

After functional bring-up, optimize performance:

- **Metal Trace**: Capture and replay operations
- **Async Mode**: Overlap compute and data movement
- **Multiple Command Queues**: Parallel operation execution

See: [Advanced Performance Optimizations](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md)

## Multi-Device Scaling

### Data Parallel

For models that fit on a single device:
- Replicate weights on different devices
- Send different inputs to different devices
- Use device mesh APIs in TT-NN

```python
# Example: Data parallel setup
device_mesh = ttnn.open_device_mesh(device_ids=[0, 1, 2, 3])

# Replicate model on all devices
for device in device_mesh.get_devices():
    parameters = load_parameters(device)

# Run inference with different inputs
outputs = []
for i, device in enumerate(device_mesh.get_devices()):
    output = model(inputs[i], device=device)
    outputs.append(output)
```

### Tensor Parallel

For models too large for a single device:
- Distribute single operations across devices
- Required for models like Falcon 40B

See: [Multi-Device Reference](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md#33-multi-device)

## Reference Documentation

- [LLMs Bring-up in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md)
- [Converting Torch Model to TT-NN](https://docs.tenstorrent.com/docs-test/ttnn/latest/ttnn/converting_torch_model_to_ttnn.html)
- [TT-Metalium Tech Reports](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#tt-metalium-tech-reports)

## Checklist

- [ ] Baseline validation on CPU/GPU completed
- [ ] Individual decode modules implemented and tested
- [ ] Modules composed into decoder layers
- [ ] Full decode mode working
- [ ] Teacher forcing validation passed
- [ ] Token generation accuracy verified
- [ ] Prefill implementation completed
- [ ] End-to-end model running
- [ ] Performance optimizations applied (optional)
- [ ] Multi-device scaling implemented (if needed)
