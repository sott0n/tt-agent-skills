# Step 2: Tensor Layout Optimization

## Objective

Configure optimal tensor layouts for compute operations.

## Layout Types

### ROW_MAJOR_LAYOUT
- Standard row-major storage
- Each row = one page in memory
- Flexible shapes, no padding required
- Limited compute support

```python
ttnn_tensor = ttnn.from_torch(tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
```

### TILE_LAYOUT
- 32x32 tile-based storage
- Required for most compute operations
- Shapes must be padded to tile multiples

```python
ttnn_tensor = ttnn.from_torch(tensor, layout=ttnn.TILE_LAYOUT)
```

## Tile Structure

### Default Tile: 32x32
- 4 faces of 16x16 each
- Faces stored contiguously: face0 → face1 → face2 → face3
- Matrix engine operates on 16x16 faces

```
┌─────────────┬─────────────┐
│   Face 0    │   Face 1    │
│   (16x16)   │   (16x16)   │
├─────────────┼─────────────┤
│   Face 2    │   Face 3    │
│   (16x16)   │   (16x16)   │
└─────────────┴─────────────┘
```

### Supported Tile Shapes
- 32x32 (default, full support)
- 16x32 (limited support, some ops like matmul)
- 4x32, 2x32, 1x32 (hardware supported, limited TT-Metalium support)

```python
# Custom tile shape
ttnn_tensor = ttnn.from_torch(
    tensor,
    layout=ttnn.TILE_LAYOUT,
    tile=ttnn.Tile((16, 32))
)
```

## Layout Conversion

```python
# Row-major to tile
tiled = ttnn.to_layout(row_major_tensor, ttnn.TILE_LAYOUT)

# Tile to row-major
row_major = ttnn.to_layout(tiled_tensor, ttnn.ROW_MAJOR_LAYOUT)
```

## Padding Considerations

TILE_LAYOUT requires dimensions to be multiples of tile size:

```python
def pad_to_tile(shape, tile_size=32):
    """Calculate padded shape for tile layout"""
    h, w = shape[-2], shape[-1]
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    return (*shape[:-2], h + pad_h, w + pad_w)

# Example: (1, 100, 50) → (1, 128, 64) for 32x32 tiles
```

## Best Practices

1. **Use TILE_LAYOUT for compute**
   - Most TTNN operations require tile layout
   - Better utilization of matrix engine

2. **Convert early, convert once**
   - Convert to tile layout after loading to device
   - Avoid repeated layout conversions

3. **Mind the padding**
   - Padding adds memory overhead
   - Choose shapes that are tile-aligned when possible

4. **Use ROW_MAJOR for I/O**
   - Host tensors typically row-major
   - Convert to tile after device transfer

## Checklist

- [ ] All compute tensors using TILE_LAYOUT
- [ ] Shape padding accounted for
- [ ] Layout conversions minimized
- [ ] Tile-aligned shapes used where possible
