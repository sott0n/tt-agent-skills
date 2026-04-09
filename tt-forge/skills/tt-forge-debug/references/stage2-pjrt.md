# Stage 2: PJRT Interface (tt-xla)

- **Modifiable**: Yes
- **Input**: StableHLO/VHLO from Torch XLA
- **Output**: Parsed MLIR module ready for conversion

## Contents

- [What Happens in This Stage](#what-happens-in-this-stage)
- [Key Classes](#key-classes)
- [Common Issues](#common-issues)
- [Environment Variables](#environment-variables)
- [Distributed Runtime](#distributed-runtime)
- [Debugging](#debugging)

## What Happens in This Stage

```
PJRT_Client_Create
    │
    ▼
ClientInstance::initialize()
    ├── launchDistributedRuntime()  [if TT_RUNTIME_ENABLE_DISTRIBUTED=1]
    ├── setMemoryLogLevel()
    ├── populateDevices()
    │   ├── Load/query system descriptor
    │   ├── Create DeviceInstance for each chip
    │   └── Open initial mesh device
    └── populateMemories()
        └── Create MemoryInstance (host + device)
    │
    ▼
PJRT_Client_Compile
    │
    ▼
ClientInstance::compileMlirProgram()
    │
    ▼
ModuleBuilder::buildModule()
    ├── createVHLOModule()        → vhlo_*.mlir
    └── convertFromVHLOToSHLO()   → shlo_*.mlir
```

## Key Classes

### ClientInstance
Singleton PJRT client managing devices, memories, and compilation.

```cpp
class ClientInstance {
  // Device management
  tt_pjrt_status populateDevices();
  tt_pjrt_status populateMemories();

  // Mesh device management
  tt::runtime::Device getOrCreateMeshDevice(const std::vector<uint32_t> &shape);
  tt::runtime::Device getOrCreateOptimizerSubmesh(const std::vector<uint32_t> &shape);
  void closeMeshDevice();

  // Compilation entry point
  tt_pjrt_status compileMlirProgram(...);

  // Mode flags
  bool isCompileOnly() const;  // TT_COMPILE_ONLY_SYSTEM_DESC mode
};
```

### GlobalClientInstanceSingleton
Ensures proper cleanup when torch_xla doesn't call `PJRT_Client_Destroy`.

### DeviceInstance / MemoryInstance
Represent TT devices and memory spaces exposed to PJRT.

### ModuleBuilder
Handles MLIR parsing and conversion pipeline.

---

## Common Issues

### Device Initialization Failures

**Symptom**: `Found no addressable devices in the system`

**Cause**: Hardware not detected or tt-metal not initialized

**Debug**:
```bash
# Check device connectivity
tt-smi

# Verify tt-metal environment
echo $TT_METAL_HOME
echo $TTMLIR_TOOLCHAIN_DIR
```

**Fix**: Ensure tt-metal is properly installed and devices are visible

### System Descriptor Errors

**Symptom**: `Failed to store the system descriptor to the disk using path: /tmp/tt_pjrt_system_descriptor`

**Cause**: Permission issues or disk space

**Fix**: Check write permissions for `/tmp` directory

### VHLO Module Parse Errors

**Symptom**: `Failed to create VHLO module from the input program code`

**Cause**: Invalid or malformed MLIR from frontend

**Debug**:
```bash
export TTXLA_LOGGER_LEVEL=DEBUG
```

Check raw MLIR being passed. If `export_path` is set, check if `vhlo_*.mlir` was created.

### VHLO to StableHLO Conversion Errors

**Symptom**: `Failed to convert from VHLO to SHLO module`

**Cause**:
- Incompatible VHLO version
- Unsupported operations in VHLO

**Debug**:
- Check `vhlo_*.mlir` in export_path
- Look for unsupported ops in the VHLO module

### Program Format Error

**Symptom**: `Program code format "X" is not supported, only MLIR format is currently supported`

**Cause**: Frontend sent non-MLIR format

**Fix**: This is typically a frontend configuration issue

### Invalid Device ID

**Symptom**: `Invalid device ID N in DeviceAssignment`

**Cause**: Device assignment specifies non-existent device

**Debug**: Check `jax.devices('tt')` or device count

---

## Compile-Only Mode

Compile for a different system without physical hardware.

```bash
# Set system descriptor path (disables execution)
export TT_COMPILE_ONLY_SYSTEM_DESC=/path/to/system.ttsys

# Generate system descriptor from a machine with hardware:
# tt-mlir-tools system-desc --save /path/to/system.ttsys
```

**Note**: Execution (`tt::runtime::submit()`) is not supported in compile-only mode.

---

## Distributed Runtime

### Required Environment Variables

```bash
# Enable distributed runtime
export TT_RUNTIME_ENABLE_DISTRIBUTED=1

# tt-metal runtime root (required)
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal

# Rank binding configuration (required) - choose one:
export TT_DISTRIBUTED_RANK_BINDING=2x4_multiprocess
export TT_DISTRIBUTED_RANK_BINDING=dual_bh_quietbox
export TT_DISTRIBUTED_RANK_BINDING=dual_t3k
export TT_DISTRIBUTED_RANK_BINDING=dual_galaxy
export TT_DISTRIBUTED_RANK_BINDING=quad_galaxy

# Worker path (required)
export TT_DISTRIBUTED_WORKER_PATH=/path/to/worker

# Hosts - choose ONE of:
export TT_DISTRIBUTED_HOSTS_LIST="host1,host2,host3"
# OR
export TT_DISTRIBUTED_HOSTS_FILE=/path/to/hostfile
```

### Optional Distributed Variables

```bash
# Controller hostname (for multi-node)
export TT_DISTRIBUTED_CONTROLLER_HOST_NAME=controller.local

# Network interface for MPI
export TT_DISTRIBUTED_TCP_IFACE=enp10s0f1np1

# Custom RSH agent for MPI
export TT_DISTRIBUTED_PLM_RSH_AGENT=/path/to/remote_docker.sh
```

### Rank Binding Configurations

| Name | Path | Use Case |
|------|------|----------|
| `2x4_multiprocess` | `tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml` | Standard 2x4 mesh |
| `dual_bh_quietbox` | `tests/scale_out/4x_bh_quietbox/rank_bindings/2x4.yaml` | Dual BH quietbox |
| `dual_t3k` | `tests/tt_metal/distributed/config/dual_t3k_1x16_experimental_bigmesh_rank_bindings.yaml` | Dual T3K cluster |
| `dual_galaxy` | `tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml` | Dual Galaxy |
| `quad_galaxy` | `tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml` | Quad Galaxy |

### Distributed Runtime Errors

**Symptom**: `TT_DISTRIBUTED_RANK_BINDING environment variable is not set`

**Fix**: Set rank binding to one of the valid configurations

**Symptom**: `Invalid rank binding: X`

**Fix**: Use one of the predefined rank binding names

**Symptom**: `TT_DISTRIBUTED_HOSTS_LIST and TT_DISTRIBUTED_HOSTS_FILE are mutually exclusive`

**Fix**: Set only one of these variables

---

## Environment Variables

### Core Debugging

```bash
# Debug logging (shows compilation stages)
export TTXLA_LOGGER_LEVEL=DEBUG

# Verbose logging (includes IR printing)
export TTXLA_LOGGER_LEVEL=VERBOSE
```

### Memory Logging

```bash
# Memory usage logging level
export TT_RUNTIME_MEMORY_LOG_LEVEL=none       # No logging
export TT_RUNTIME_MEMORY_LOG_LEVEL=program    # Per-program logging
export TT_RUNTIME_MEMORY_LOG_LEVEL=operation  # Per-operation logging
export TT_RUNTIME_MEMORY_LOG_LEVEL=any        # All memory events
```

### Runtime Configuration

```bash
# Program cache (enabled by default)
export TT_RUNTIME_ENABLE_PROGRAM_CACHE=0  # Disable

# Trace region size (for enable_trace option)
export TT_RUNTIME_TRACE_REGION_SIZE=10000000

# Dual T3K cluster workaround (forces FABRIC_2D)
export TT_RUNTIME_USING_DUALT3K=1
```

### Codegen

```bash
# Required for codegen backends
export TT_MLIR_HOME=/path/to/tt-mlir
```

---

## Mesh Device Management

### Mesh Creation and Reuse

The client maintains a single mesh device. When a different shape is needed:
1. Tensors are moved to host (via `TensorPool::move_tensors_to_host()`)
2. Current mesh is closed
3. New mesh is opened with target shape

```cpp
// Mesh is reused if shape and fabric config match
tt::runtime::Device getOrCreateMeshDevice(const std::vector<uint32_t> &target_mesh_shape);
```

### Fabric Configuration

Fabric config is computed based on system descriptor and mesh shape:
- Single device: `DISABLED`
- Multi-device (distributed): `FABRIC_1D` (or `FABRIC_2D` for dual T3K)
- Multi-device (local): Computed from system descriptor

---

## Compile Options Affecting This Stage

Only export options affect Stage 2 directly:

```python
options = {
    "export_path": "/path/to/dump",      # Dump vhlo_*.mlir, shlo_*.mlir
    "export_model_name": "model_name",   # Filename prefix
}
```

See [compile-options.md](compile-options.md) for complete options reference.

---

## Key Source Files

- `pjrt_implementation/src/api/client_instance.cc` - Client initialization, mesh management
- `pjrt_implementation/inc/api/client_instance.h` - ClientInstance class
- `pjrt_implementation/src/api/device_instance.cc` - Device management
- `pjrt_implementation/src/api/memory_instance.cc` - Memory management
- `pjrt_implementation/src/api/module_builder/module_builder.cc` - MLIR parsing/conversion

---

## Checking if Stage 2 Succeeded

Check `<export_path>/irs/` for:
- `vhlo_*.mlir` - VHLO module was parsed successfully
- `shlo_*.mlir` - StableHLO conversion succeeded

| Files Present | Status |
|---------------|--------|
| None | Stage 2 failed (VHLO parse) or export_path not set |
| `vhlo_*.mlir` only | VHLO→StableHLO conversion failed |
| `vhlo_*.mlir` + `shlo_*.mlir` | Stage 2 succeeded, proceed to Stage 3 |

---

## Error Messages Reference

| Error Message | Likely Cause | Fix |
|--------------|--------------|-----|
| `Found no addressable devices in the system` | No TT devices detected | Check tt-smi, tt-metal installation |
| `Failed to store the system descriptor to the disk` | Permission/disk issue | Check /tmp permissions |
| `Failed to create VHLO module from the input program code` | Invalid MLIR input | Check frontend, enable DEBUG logging |
| `Failed to convert from VHLO to SHLO module` | VHLO version mismatch | Update Torch XLA version |
| `Program code format "X" is not supported` | Non-MLIR format | Frontend configuration issue |
| `Invalid device ID N in DeviceAssignment` | Bad device assignment | Check device count |
| `TT_DISTRIBUTED_RANK_BINDING environment variable is not set` | Missing env var | Set rank binding |
| `Invalid rank binding: X` | Unknown rank binding | Use valid binding name |
| `TT_METAL_RUNTIME_ROOT environment variable is not set` | Missing env var | Set metal runtime root |
| `Client device lookup failed for device with ID: N` | Device not found | Check device configuration |
