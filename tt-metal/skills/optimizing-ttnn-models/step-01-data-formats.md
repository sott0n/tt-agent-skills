# Step 1: Data Format Optimization

## Objective

Select optimal data formats to balance precision and performance.

## Available Data Formats

| Format | Bits | Exponent | Mantissa | Notes |
|--------|------|----------|----------|-------|
| `float32` | 32 | 8 | 23 | Full precision |
| `bfloat16` | 16 | 8 | 7 | Default for activations |
| `bfloat8_b` | 8 | shared | 7 | Block float, good for weights |
| `bfloat4_b` | 4 | shared | 3 | Aggressive compression |

## Block Float Formats

Block float formats (bfloat8_b, bfloat4_b) share exponents across 16 values:
- 16 values share a single 8-bit exponent
- Each value stores only mantissa bits
- Requires TILE_LAYOUT

```python
# Convert weights to bfloat8_b
weight_bf8 = ttnn.from_torch(
    torch_weight,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device
)
```

## Recommendations

### Activations
- Use `bfloat16` for general activations
- Use `float32` only when debugging precision issues

### Weights
- Use `bfloat8_b` for most weights (2x memory reduction)
- Use `bfloat16` if PCC drops below threshold
- Use `bfloat4_b` for aggressive memory savings (verify PCC)

### Accumulation
```python
# Enable fp32 accumulation for higher precision
compute_config = ttnn.WormholeComputeKernelConfig(
    fp32_dest_acc_en=True,  # Accumulate in float32
    packer_l1_acc=True,     # L1 accumulation
)
```

## Math Fidelity

Math Fidelity controls precision vs performance by specifying how many times an operation runs to consume full input precision.

### How It Works

Wormhole multipliers are 5-bit × 7-bit:
- SrcA (operand 0): 5 bits used
- SrcB (operand 1): 7 bits used

| Fidelity | SrcA Usage | SrcB Usage | Passes |
|----------|------------|------------|--------|
| LoFi | 1 hidden + 4 MSB | 1 hidden + 6 MSB | 1 |
| HiFi2 | 1 hidden + 4 LSB | 1 hidden + 6 MSB | 2 |
| HiFi3 | 1 hidden + 4 MSB | 1 hidden + 6 LSB | 3 |
| HiFi4 | 1 hidden + 4 LSB | 1 hidden + 6 LSB | 4 |

### Fidelity Options

| Fidelity | TFLOPS | Use Case |
|----------|--------|----------|
| LoFi | 4.0 | High performance, low accuracy |
| HiFi2 | 2.0 | Balanced performance/accuracy |
| HiFi3 | 1.33 | Higher precision |
| HiFi4 | 1.0 | Highest accuracy operations |

Note: Accumulation precision (FP16/FP32) is controlled separately via `fp32_dest_acc_en`.

### Performance by Operation

**Matrix Multiplication (8x16 × 16x16 = 8x16 per cycle)**

| Fidelity | TFLOPS | Relative |
|----------|--------|----------|
| LoFi | 4.0 | 100% |
| HiFi2 | 2.0 | 50% |
| HiFi3 | 1.33 | 33% |
| HiFi4 | 1.0 | 25% |

**Reduction (Max/Average/Sum)**

| Fidelity | TFLOPS | Notes |
|----------|--------|-------|
| LoFi | 0.512 | Reduce max ignores fidelity |
| HiFi2 | 0.256 | |
| HiFi3 | 0.171 | |
| HiFi4 | 0.128 | |

**Elementwise (Add/Sub/Mul)**

| Fidelity | TFLOPS | Notes |
|----------|--------|-------|
| LoFi | 0.128 | Add/Sub ignore fidelity |
| HiFi2 | 0.064 | Mul uses fidelity |
| HiFi3 | 0.043 | |
| HiFi4 | 0.032 | |

### Configuration

```python
compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,   # High performance, low accuracy
    # math_fidelity=ttnn.MathFidelity.HiFi2, # Balanced
    # math_fidelity=ttnn.MathFidelity.HiFi3, # Higher precision
    # math_fidelity=ttnn.MathFidelity.HiFi4, # Highest accuracy
    fp32_dest_acc_en=False,  # False: FP16 accumulation, True: FP32 accumulation
)

# Apply to operations
output = ttnn.matmul(a, b, compute_kernel_config=compute_config)
```

### Accumulation Precision

Accumulation precision is controlled separately from Math Fidelity via `fp32_dest_acc_en`:
- **fp32_dest_acc_en=True**: FP32 accumulation (higher precision, but dest register holds half as many tiles)
- **fp32_dest_acc_en=False**: FP16 accumulation (memory efficient, uses less dest register space)

### Optimization Strategy

1. **Start with LoFi** for maximum performance
2. **Check PCC** against reference
3. **Increase fidelity** if PCC is insufficient:
   - LoFi → HiFi2 (2x slower, balanced accuracy)
   - HiFi2 → HiFi3 (1.5x slower, higher precision)
   - HiFi3 → HiFi4 (1.33x slower, highest accuracy)
4. **Consider accumulation precision**:
   - If L1 is tight, use `fp32_dest_acc_en=False` (FP16 accumulation)
   - If precision matters, use `fp32_dest_acc_en=True` (FP32 accumulation)

```python
def find_optimal_fidelity(model, reference_output, threshold=0.999):
    """Find minimum fidelity that meets PCC threshold"""
    fidelities = [
        ttnn.MathFidelity.LoFi,   # Try first: high perf, low accuracy
        ttnn.MathFidelity.HiFi2,  # Balanced
        ttnn.MathFidelity.HiFi3,  # Higher precision
        ttnn.MathFidelity.HiFi4,  # Highest accuracy fallback
    ]
    for fidelity in fidelities:
        config = ttnn.WormholeComputeKernelConfig(math_fidelity=fidelity)
        output = model(input, compute_kernel_config=config)
        pcc = calculate_pcc(reference_output, ttnn.to_torch(output))
        print(f"{fidelity}: PCC = {pcc:.6f}")
        if pcc >= threshold:
            return fidelity
    return ttnn.MathFidelity.HiFi4

# Per-layer fidelity tuning
layer_fidelity = {
    "attention": ttnn.MathFidelity.HiFi2,  # Softmax needs precision
    "ffn": ttnn.MathFidelity.LoFi,         # FFN tolerates low precision
    "layernorm": ttnn.MathFidelity.HiFi2,  # Normalization needs precision
}
```

### Math Approx Mode

Some SFPU operations support approximate mode for higher performance:

```python
compute_config = ttnn.WormholeComputeKernelConfig(
    math_approx_mode=True,  # Enable approximations
)
```

Supported operations with approximations:
- Exponential (`exp`)
- GELU
- Square root (`sqrt`)
- Reciprocal

## Mantissa Rounding

When converting to lower precision:
- Rounds to nearest, ties to even
- Block floats normalize to shared exponent first
- Overflow saturates to max value

## Validation

After changing data formats, verify PCC:

```python
def test_format_change(device, torch_model, ttnn_model_bf8):
    with torch.no_grad():
        torch_output = torch_model(input)

    ttnn_output = ttnn_model_bf8(ttnn_input)

    # bfloat8_b may have slightly lower PCC
    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), 0.99)
```

## Checklist

- [ ] Weights converted to bfloat8_b where possible
- [ ] PCC verified after format changes
- [ ] Math fidelity tuned per layer/operation
- [ ] Started with LoFi, increased only where needed
- [ ] Math approx mode enabled for supported ops
- [ ] fp32_dest_acc_en considered for precision-critical paths
- [ ] Memory reduction measured
