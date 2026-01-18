# Debugging Tools

## Comparison Mode

Enable to automatically compare TTNN outputs against PyTorch:

```bash
export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": false,
    "enable_comparison_mode": true,
    "comparison_mode_should_raise_exception": true,
    "comparison_mode_pcc": 0.99
}'
```

## PCC Validation

```python
from tests.ttnn.utils_for_testing import assert_with_pcc

assert_with_pcc(torch_output, ttnn_output, 0.999)
```

## Common Debugging Steps

1. **Enable comparison mode** to identify which operator fails
2. **Check data types** - ensure matching dtypes between torch and ttnn
3. **Check layouts** - verify TILE_LAYOUT for compute operations
4. **Reduce precision** - test with float32 to isolate numerical issues
5. **Test individual operators** - isolate the failing operator
