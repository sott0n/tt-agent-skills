---
name: optimizing-ttnn-models
description: "Optimizes TTNN model performance on Tenstorrent hardware. Covers data formats, sharding, Metal Trace, and multi-device scaling. Use when model runs too slow, needs memory optimization, profiling bottlenecks, tuning Conv2d, enabling Metal Trace, or scaling to multiple devices."
---

# TTNN Model Optimization

## Optimization Workflow

| Step | Name | Description | Document |
|------|------|-------------|----------|
| 1 | Data Formats | Choose optimal precision (bfloat16, bfloat8_b) | `step-01-data-formats.md` |
| 2 | Tensor Layouts | Configure tile layout and shapes | `step-02-tensor-layouts.md` |
| 3 | Memory & Sharding | Optimize L1/DRAM usage, double buffering | `step-03-memory-sharding.md` |
| 4 | Metal Trace | Capture and replay operations | `step-04-metal-trace.md` |
| 5 | Multi-CQ | Overlap I/O with compute | `step-05-multi-cq.md` |
| 6 | Conv2d Optimization | Sharding, block tuning, BN folding | `step-06-conv2d-optimization.md` |
| 7 | Multi-Device | Scale across multiple devices | `step-07-multi-device.md` |

### Choosing Optimization Path

**LLM (transformers, attention-based)?**
→ Focus on: Data formats (Step 1), Metal Trace (Step 4), Multi-CQ (Step 5)

**CNN (convolutions, image models)?**
→ Focus on: Sharding (Step 3), Conv2d optimization (Step 6), act_block_h tuning

**Memory-bound?**
→ Focus on: Memory & Sharding (Step 3), double buffering

**Host dispatch overhead?**
→ Focus on: Metal Trace (Step 4), Multi-CQ (Step 5)

### Optimization Feedback Loop

```
1. Profile current performance (baseline)
2. Apply ONE optimization
3. Verify PCC ≥ 0.99
4. If PCC drops → revert and try different approach
5. Measure performance improvement
6. Repeat until target met
```

### Progress Checklist

```
Optimization Progress:
- [ ] Baseline performance measured
- [ ] Step 1: Data formats optimized (bfloat8_b for weights)
- [ ] Step 2: Tile layout configured
- [ ] Step 3: Sharding strategy applied
- [ ] Step 4: Metal Trace enabled
- [ ] Step 5: Multi-CQ configured (if applicable)
- [ ] Step 6: Conv2d tuned (if CNN)
- [ ] Step 7: Multi-device scaling (if applicable)
- [ ] Final PCC verified ≥ 0.99
```

## Quick Reference

- **Key Concepts**: See [key-concepts.md](key-concepts.md) for data formats, layouts, memory hierarchy, sharding, and math fidelity
- **Code Examples**: See [quick-reference.md](quick-reference.md) for memory configs, compute configs, Metal Trace, and multi-CQ examples

## Instructions for Claude

1. **Verify functional correctness** before optimizing
2. **Profile first** to identify bottlenecks
3. **Read step files** for detailed optimization techniques
4. **Test after each optimization** to verify PCC maintained

### Example Interactions

**Example 1: Model too slow**
```
User: "My ResNet inference is 50ms, target is 10ms"

Claude response:
1. Identifies model type: CNN
2. Reads step-06-conv2d-optimization.md
3. Checks: sharding strategy, act_block_h_override, double buffering
4. Suggests enabling Metal Trace (step-04-metal-trace.md)
```

**Example 2: Memory optimization**
```
User: "Running out of L1 memory during convolution"

Claude response:
1. Reads step-03-memory-sharding.md
2. Suggests: reduce act_block_h_override, store skip connections in DRAM
3. Recommends memory profiling with ttnn.device.dump_device_memory_state()
```

**Example 3: Enabling Metal Trace**
```
User: "How do I use Metal Trace for my LLM?"

Claude response:
1. Reads step-04-metal-trace.md
2. Provides pattern: allocate_tensor → begin_trace_capture → model() → end_trace_capture
3. Notes constraints: no dynamic shapes, fixed compute graph
```

## Files in This Skill

```
.claude/skills/optimizing-ttnn-models/
├── SKILL.md                        # This file
├── key-concepts.md                 # Data formats, layouts, memory, sharding, math fidelity
├── quick-reference.md              # Code examples for common patterns
├── step-01-data-formats.md         # Data format optimization
├── step-02-tensor-layouts.md       # Tensor layout optimization
├── step-03-memory-sharding.md      # Memory, sharding, double buffering
├── step-04-metal-trace.md          # Metal Trace guide
├── step-05-multi-cq.md             # Multiple command queues
├── step-06-conv2d-optimization.md  # Conv2d tuning (vision models)
└── step-07-multi-device.md         # Multi-device scaling
```

## Reference

### Tech Reports
- `tech_reports/AdvancedPerformanceOptimizationsForModels/` - Metal Trace & Multi-CQ
- `tech_reports/data_formats/data_formats.md` - Data format details
- `tech_reports/tensor_layouts/tensor_layouts.md` - Layout concepts
- `tech_reports/tensor_sharding/tensor_sharding.md` - Sharding strategies
- `tech_reports/memory/allocator.md` - Memory allocation
- `tech_reports/matrix_engine/matrix_engine.md` - Compute config

### TT-CNN Library (Vision Models)
- `models/tt_cnn/README.md` - Pipeline and Builder API
- `models/tt_cnn/TT_CNN_MODEL_BRINGUP_GUIDE.md` - Vision model optimization guide

### Documentation
- `docs/source/ttnn/ttnn/api.rst` - TTNN API reference
