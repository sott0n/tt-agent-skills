# Key Concepts

## Tensor Layout
- `ttnn.ROW_MAJOR_LAYOUT`: Standard row-major storage
- `ttnn.TILE_LAYOUT`: 32x32 tile-based storage (required for most compute operations)

## Data Types
- `ttnn.bfloat16`: Default for activations
- `ttnn.bfloat8_b`: Block floating point, good for weights (requires TILE_LAYOUT)
- `ttnn.float32`: Higher precision when needed

## Memory Configs
- `ttnn.DRAM_MEMORY_CONFIG`: Store in DRAM (larger capacity)
- `ttnn.L1_MEMORY_CONFIG`: Store in L1 cache (faster access)
- `ttnn.create_sharded_memory_config()`: Distributed across cores

## PCC (Pearson Correlation Coefficient)
- >= 0.999: Excellent - target for individual operators
- >= 0.99: Good - acceptable for modules and full model
- < 0.99: Investigate numerical issues
