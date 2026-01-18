# Step 7: End-to-End Model Testing

## Objective

Validate the complete TTNN model against PyTorch reference with full end-to-end testing and debugging.

## Prerequisites

- Completed Step 6 with full model implementation
- All modules pass individual tests

## Tasks

### 7.1 Create End-to-End Test

**test_model.py:**
```python
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.demos.<your_model>.tt.ttnn_model import TtnnBertModel


class TestBertModelE2E:

    @pytest.fixture
    def model_name(self):
        return "bert-base-uncased"

    @pytest.fixture
    def torch_model(self, model_name):
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_name).eval()

    @pytest.fixture
    def ttnn_model(self, model_name, device):
        return TtnnBertModel.from_pretrained(model_name, device)

    @pytest.fixture
    def sample_input(self, model_name):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text = "Hello, this is a test sentence for BERT model."
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128)
        return inputs

    def test_e2e_inference(self, device, torch_model, ttnn_model, sample_input):
        """Test full model inference"""
        torch.manual_seed(0)

        input_ids = sample_input["input_ids"]
        attention_mask = sample_input["attention_mask"]
        token_type_ids = sample_input.get("token_type_ids", torch.zeros_like(input_ids))

        # PyTorch reference
        with torch.no_grad():
            torch_outputs = torch_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            torch_hidden_states = torch_outputs.last_hidden_state

        # TTNN model
        ttnn_hidden_states, _ = ttnn_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Convert and compare
        output = ttnn.to_torch(ttnn_hidden_states)
        assert_with_pcc(torch_hidden_states, output, 0.99)

    def test_e2e_batch_inference(self, device, torch_model, ttnn_model):
        """Test with batch inputs"""
        torch.manual_seed(0)

        batch_size = 4
        seq_len = 128

        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        # PyTorch reference
        with torch.no_grad():
            torch_outputs = torch_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            torch_hidden_states = torch_outputs.last_hidden_state

        # TTNN model
        ttnn_hidden_states, _ = ttnn_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        output = ttnn.to_torch(ttnn_hidden_states)
        assert_with_pcc(torch_hidden_states, output, 0.99)

    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512])
    def test_various_sequence_lengths(self, device, torch_model, ttnn_model, seq_len):
        """Test with various sequence lengths"""
        torch.manual_seed(0)

        input_ids = torch.randint(0, 30000, (1, seq_len))
        attention_mask = torch.ones(1, seq_len)

        with torch.no_grad():
            torch_outputs = torch_model(input_ids=input_ids, attention_mask=attention_mask)

        ttnn_hidden_states, _ = ttnn_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        output = ttnn.to_torch(ttnn_hidden_states)
        assert_with_pcc(torch_outputs.last_hidden_state, output, 0.99)
```

### 7.2 Use Comparison Mode for Debugging

If the E2E test fails, enable comparison mode to identify the failing operation:

```python
def test_with_comparison_mode(device, torch_model, ttnn_model, sample_input):
    """Debug with comparison mode"""
    import os

    # Enable comparison mode
    os.environ['TTNN_CONFIG_OVERRIDES'] = '''{
        "enable_fast_runtime_mode": false,
        "enable_comparison_mode": true,
        "comparison_mode_should_raise_exception": true,
        "comparison_mode_pcc": 0.99
    }'''

    # This will raise an exception at the first operation that fails PCC
    try:
        ttnn_hidden_states, _ = ttnn_model(
            input_ids=sample_input["input_ids"],
            attention_mask=sample_input["attention_mask"],
        )
    except Exception as e:
        print(f"Failed at operation: {e}")
        raise
```

Or via command line:

```bash
export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": false,
    "enable_comparison_mode": true,
    "comparison_mode_should_raise_exception": true,
    "comparison_mode_pcc": 0.99
}'

pytest tests/ttnn/unit_tests/models/<your_model>/test_model.py -v
```

### 7.3 Layer-by-Layer Debugging

If comparison mode doesn't pinpoint the issue, debug layer by layer:

```python
def debug_layer_by_layer(torch_model, ttnn_model, input_ids, device):
    """Compare outputs at each layer"""

    # Get PyTorch intermediate outputs
    torch_hidden_states = []

    def torch_hook(module, input, output):
        torch_hidden_states.append(output[0].clone())

    # Register hooks on each encoder layer
    hooks = []
    for layer in torch_model.encoder.layer:
        hooks.append(layer.register_forward_hook(torch_hook))

    # Run PyTorch
    with torch.no_grad():
        torch_output = torch_model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compare TTNN outputs at each layer
    # (You'll need to modify your TTNN model to return intermediate outputs)
    ttnn_hidden_states = ttnn_model.forward_with_intermediates(input_ids)

    for i, (torch_h, ttnn_h) in enumerate(zip(torch_hidden_states, ttnn_hidden_states)):
        ttnn_h_torch = ttnn.to_torch(ttnn_h)
        pcc = calculate_pcc(torch_h, ttnn_h_torch)
        print(f"Layer {i}: PCC = {pcc:.6f}")

        if pcc < 0.99:
            print(f"  WARNING: Low PCC at layer {i}")
            # Debug this specific layer
            debug_single_layer(torch_model.encoder.layer[i], ...)
```

### 7.4 Performance Benchmarking

After validating correctness, benchmark performance:

```python
import time

def benchmark_model(ttnn_model, input_ids, attention_mask, num_iterations=100):
    """Benchmark model performance"""

    # Warmup
    for _ in range(10):
        _ = ttnn_model(input_ids, attention_mask=attention_mask)

    # Benchmark
    ttnn.synchronize_device(ttnn_model.device)
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        _ = ttnn_model(input_ids, attention_mask=attention_mask)

    ttnn.synchronize_device(ttnn_model.device)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations
    throughput = 1.0 / avg_time

    print(f"Average inference time: {avg_time * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} inferences/sec")

    return avg_time, throughput
```

### 7.5 Test Real-World Inputs

Test with realistic inputs:

```python
def test_real_world_text(device, torch_model, ttnn_model):
    """Test with real text examples"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "BERT is a transformer-based machine learning technique for NLP.",
        "Tenstorrent makes AI accelerators for efficient inference.",
        # Add more diverse test cases
    ]

    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", max_length=128)

        with torch.no_grad():
            torch_output = torch_model(**inputs).last_hidden_state

        ttnn_output, _ = ttnn_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        output = ttnn.to_torch(ttnn_output)
        pcc = calculate_pcc(torch_output, output)

        print(f"Sentence: {sentence[:50]}...")
        print(f"  PCC: {pcc:.6f}")

        assert pcc >= 0.99, f"PCC too low for: {sentence}"
```

### 7.6 Create Demo Script

**demo_inference.py:**
```python
#!/usr/bin/env python3
"""Demo script for TTNN BERT inference"""

import argparse
import torch
import ttnn

from models.demos.<your_model>.tt.ttnn_model import TtnnBertModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--text", default="Hello, this is a test.")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    # Open device
    device = ttnn.open_device(device_id=args.device_id)

    try:
        # Load model
        print(f"Loading model: {args.model}")
        model = TtnnBertModel.from_pretrained(args.model, device)

        # Tokenize input
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        inputs = tokenizer(args.text, return_tensors="pt", padding="max_length", max_length=128)

        # Run inference
        print(f"Running inference on: {args.text}")
        hidden_states, pooled_output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Convert output
        output = ttnn.to_torch(hidden_states)
        print(f"Output shape: {output.shape}")
        print(f"Output (first 5 values): {output[0, 0, :5]}")

        # Compare with PyTorch (optional)
        from transformers import AutoModel
        torch_model = AutoModel.from_pretrained(args.model).eval()
        with torch.no_grad():
            torch_output = torch_model(**inputs).last_hidden_state

        from tests.ttnn.utils_for_testing import check_with_pcc
        pcc = check_with_pcc(torch_output, output)
        print(f"PCC vs PyTorch: {pcc:.6f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
```

### 7.7 Run Full Test Suite

```bash
# Run all E2E tests
pytest tests/ttnn/unit_tests/models/<your_model>/test_model.py -v

# Run with comparison mode for debugging
TTNN_CONFIG_OVERRIDES='{"enable_comparison_mode": true, "comparison_mode_pcc": 0.99}' \
    pytest tests/ttnn/unit_tests/models/<your_model>/test_model.py -v

# Run demo
python models/demos/<your_model>/demo/demo_inference.py --text "Your test text here"

# Run benchmark
python -c "
import ttnn
from models.demos.<your_model>.tt.ttnn_model import TtnnBertModel
device = ttnn.open_device(0)
model = TtnnBertModel.from_pretrained('bert-base-uncased', device)
# benchmark code...
"
```

## Deliverables

1. **E2E test suite** with comprehensive coverage
2. **Demo script** for easy testing
3. **Performance benchmark** results
4. **Final PCC validation** report

## Checklist

- [ ] E2E tests pass with PCC >= 0.99
- [ ] Various input shapes tested
- [ ] Real-world text inputs validated
- [ ] Comparison mode debugging documented
- [ ] Demo script works correctly
- [ ] Performance benchmarked

## Final PCC Summary

| Test Case | Batch | Seq Len | PCC | Status |
|-----------|-------|---------|-----|--------|
| Basic inference | 1 | 128 | 0.993 | ✅ |
| Batch inference | 4 | 128 | 0.992 | ✅ |
| Long sequence | 1 | 512 | 0.991 | ✅ |
| Real text | 1 | var | 0.994 | ✅ |

## Model Bringup Complete!

Congratulations! You have successfully brought up a model on TTNN.

### Next Steps (Optional)

For further optimization, consider:
- **Performance optimization** - See optimization guides
- **Memory optimization** - Sharding, L1 caching
- **Multi-device support** - Distributed inference
- **Quantization** - Lower precision for faster inference
