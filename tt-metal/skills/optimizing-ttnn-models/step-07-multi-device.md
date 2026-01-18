# Step 7: Multi-Device Optimization

## Objective

Scale models across multiple Tenstorrent devices for increased throughput.

## Overview

Multi-device support enables:
- Data parallelism (same model, different data)
- Model parallelism (different parts of model on different devices)
- Increased aggregate memory and compute

## Device Mesh Setup

```python
import ttnn

# Open multiple devices as a mesh
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 2),  # 1x2 mesh (2 devices)
    mesh_type=ttnn.MeshType.Ring,
)

# Get individual device count
num_devices = mesh_device.get_num_devices()
```

## Tensor Distribution Strategies

### Replicate Tensor to Mesh

Replicate the same tensor to all devices (weights):

```python
# Replicate weights across all devices
weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

weight = ttnn.from_torch(
    torch_weight,
    dtype=ttnn.bfloat8_b,
    mesh_mapper=weights_mesh_mapper,
)
```

### Shard Tensor to Mesh

Distribute tensor across devices (batch parallelism):

```python
# Shard inputs along batch dimension
inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)

input_tensor = ttnn.from_torch(
    torch_input,  # Shape: [batch, ...]
    dtype=ttnn.bfloat16,
    mesh_mapper=inputs_mesh_mapper,
)
# Each device gets batch/num_devices samples
```

## Data Parallel Pattern

```python
# Setup mesh mappers
inputs_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)  # Shard batch
weights_mapper = ttnn.ReplicateTensorToMesh(mesh_device)    # Replicate weights

# Preprocess model parameters with replication
def create_preprocessor_with_mesh(mesh_device):
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def preprocessor(model, name, ttnn_module_args):
        # Convert weights with mesh mapper
        return {
            "weight": ttnn.from_torch(
                model.weight,
                dtype=ttnn.bfloat8_b,
                mesh_mapper=mesh_mapper,
            ),
            "bias": ttnn.from_torch(
                model.bias.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                mesh_mapper=mesh_mapper,
            ),
        }
    return preprocessor

parameters = preprocess_model_parameters(
    initialize_model=lambda: reference_model,
    custom_preprocessor=create_preprocessor_with_mesh(mesh_device),
    device=None,
)

# Run inference
# Input is sharded across devices, weights replicated
input_tensor = ttnn.from_torch(batch, mesh_mapper=inputs_mapper)
output = model(input_tensor)

# Gather outputs from all devices
output_torch = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(dim=0))
```

## Mesh Composers

Gather results from distributed tensors:

```python
# Concatenate along dimension
composer = ttnn.ConcatMeshToTensor(dim=0)
gathered = ttnn.to_torch(distributed_tensor, mesh_composer=composer)

# List of tensors (one per device)
composer = ttnn.ListMeshToTensor()
tensors_list = ttnn.to_torch(distributed_tensor, mesh_composer=composer)
```

## Memory Configuration for Mesh

```python
# Create DRAM config for mesh
dram_config = ttnn.create_sharded_memory_config(
    shape=input_shape,
    core_grid=mesh_device.dram_grid_size(),
    strategy=ttnn.ShardStrategy.WIDTH,
)
```

## Synchronization

```python
# Synchronize all devices in mesh
ttnn.synchronize_mesh_device(mesh_device)
```

## Best Practices

1. **Use data parallelism for batch scaling**
   - Shard inputs along batch dimension
   - Replicate weights to all devices
   - Linear scaling with device count

2. **Minimize inter-device communication**
   - Keep independent operations on separate devices
   - Use local reductions before cross-device communication

3. **Balance memory across devices**
   - Distribute large tensors evenly
   - Monitor per-device memory usage

4. **Handle non-divisible batches**
   - Pad batch size to be divisible by device count
   - Or use dynamic batch handling

## Example: Multi-Device Vision Model

```python
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

# Open mesh device
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 2),
)

# Setup mappers
inputs_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
weights_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

# Preprocess with mesh mapper
parameters = preprocess_model_parameters(
    initialize_model=lambda: reference_model,
    custom_preprocessor=create_preprocessor(mesh_device, mesh_mapper=weights_mapper),
    device=None,
)

# Create model
configs = create_model_configs(parameters, input_height, input_width, batch_size)
model = create_model(configs, mesh_device)

# Run inference with batched input
batch_input = torch.randn(batch_size * num_devices, C, H, W)
ttnn_input = ttnn.from_torch(batch_input, mesh_mapper=inputs_mapper)
output = model(ttnn_input)

# Gather results
output_torch = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(dim=0))
```

## Cleanup

```python
# Close mesh device when done
ttnn.close_mesh_device(mesh_device)
```

## Checklist

- [ ] Mesh device opened with correct shape
- [ ] Weights replicated to all devices
- [ ] Inputs sharded along batch dimension
- [ ] Batch size divisible by device count
- [ ] Outputs gathered with appropriate composer
- [ ] All devices synchronized before reading results
- [ ] Mesh device closed on cleanup
