# Step 3: Memory and Sharding Optimization

## Contents
- Memory Types and Layouts
- Sharding Strategies (Height, Width, Block)
- Memory Config for Sharding
- Reshard Operation
- Memory Reports (debugging)
- Double Buffering
- Activation Block Tuning
- Skip Connections and Concatenation
- Best Practices and Checklist

## Objective

Optimize memory layout and sharding for performance.

## Memory Types

| Storage | Location | Capacity | Speed |
|---------|----------|----------|-------|
| L1 | Per core | ~1MB | Fast |
| DRAM | Shared | 12-32GB | Slower |

## Memory Layouts

### Interleaved (Default)
Pages distributed round-robin across banks.

```python
# L1 interleaved
tensor = ttnn.to_device(tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)

# DRAM interleaved
tensor = ttnn.to_device(tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

### Sharded
Tensor divided into shards, each on a specific core.

```python
# Height sharding
mem_config = ttnn.create_sharded_memory_config(
    shape=(shard_height, width),
    core_grid=ttnn.CoreGrid(y=8, x=1),
    strategy=ttnn.ShardStrategy.HEIGHT
)
```

## Sharding Strategies

### Height Sharding
Split along height dimension. Each core gets consecutive rows.

```python
core_ranges = ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))  # 8 cores
})

spec = ttnn.TensorSpec(
    shape=(256, 768),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    buffer_type=ttnn.BufferType.L1
).height_sharded(core_ranges)
# Each core: 32 rows × 768 cols
```

### Width Sharding
Split along width dimension. Each core gets consecutive columns.

```python
spec = ttnn.TensorSpec(
    shape=(64, 512),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    buffer_type=ttnn.BufferType.L1
).width_sharded(core_ranges)
# Each core: 64 rows × 64 cols (512/8)
```

### Block Sharding
2D grid split. Each core gets a rectangular block.

```python
core_ranges = ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))  # 4x4 grid
})

spec = ttnn.TensorSpec(
    shape=(256, 256),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    buffer_type=ttnn.BufferType.L1
).block_sharded(core_ranges)
# Each core: 64×64 block
```

## Memory Config for Sharding

```python
# Advanced sharding configuration
memory_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        grid=ttnn.num_cores_to_corerangeset(
            target_num_cores=8,
            grid_size=[8, 7],
            row_wise=True,
        ),
        shard_shape=[64, 512],
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
```

## Reshard Operation

Convert between sharding strategies:

```python
# Move from DRAM to L1 sharded
l1_tensor = ttnn.to_memory_config(dram_tensor, l1_sharded_config)

# Reshard to different layout
resharded = ttnn.reshard(tensor, new_shard_config)
```

## Memory Reports

Debug memory usage:

```python
# Dump memory state
ttnn.device.dump_device_memory_state(device, prefix="debug")

# Enable continuous logging
ttnn.device.EnableMemoryReports()
# ... run operations ...
ttnn.device.DisableMemoryReports()

# Reports generated in $TT_METAL_HOME/generated/
# - l1_usage_summary.csv
# - memory_usage_summary.csv
# - detailed_memory_usage.csv
```

## Double Buffering

Enable double buffering to overlap data movement with computation:

```python
# For convolution operations
conv_config = ttnn.Conv2dConfig(
    # ... other config ...
    enable_act_double_buffer=True,   # Double buffer activations
    enable_weights_double_buffer=True,  # Double buffer weights
)
```

**Benefits:**
- Overlaps memory reads with compute
- Hides memory latency
- Increases throughput

**Trade-off:**
- Requires more L1 memory (2x buffer size)

## Activation Block Tuning

For convolution operations, tune activation block sizes for optimal performance:

```python
# Tune act_block_h_override for convolutions
# Must be multiple of 32, maximize without running out of L1
sharding_config = ttnn.Conv2dConfig(
    act_block_h_override=15 * 32,  # 480 rows per block
    # Larger values = more L1 per core, better performance
)
```

**Guidelines:**
- Start with smaller blocks, increase until L1 runs out
- Each core processes `act_block_h_override` rows at a time
- Larger blocks reduce kernel launch overhead
- Must fit in L1 with weights and intermediates

## Skip Connections and Concatenation

For architectures with skip connections (UNet, ResNet):

```python
# Store skip connections in DRAM to free L1
skip = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

# Later, bring back to L1 with matching sharding
skip_l1 = ttnn.to_memory_config(skip, target_memory_config)

# Concatenate along channel dimension (HWC format: dim=3)
concat = ttnn.concat([upsampled, skip_l1], dim=3)
```

## Best Practices

1. **Use L1 for hot data**
   - Activations during compute
   - Frequently accessed intermediates

2. **Use DRAM for cold data**
   - Weights (load to L1 as needed)
   - Large tensors that don't fit L1
   - Skip connections that aren't immediately needed

3. **Match sharding to operation**
   - Height sharding for row-wise ops
   - Width sharding for column-wise ops
   - Block sharding for 2D operations

4. **Minimize resharding**
   - Plan shard layouts to match operation chain
   - Reshard only when necessary

5. **Enable double buffering**
   - For compute-bound operations
   - When L1 memory allows

6. **Maximize activation block sizes**
   - Tune `act_block_h_override` for convolutions
   - Increase until L1 is fully utilized

## Checklist

- [ ] Critical paths use L1 memory
- [ ] Sharding strategy matches operations
- [ ] Memory usage profiled and optimized
- [ ] Reshard operations minimized
- [ ] Double buffering enabled where beneficial
- [ ] Activation block sizes tuned for convolutions
- [ ] Skip connections stored in DRAM when not immediately needed
