# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TT-Metal is Tenstorrent's open-source software stack for AI accelerator hardware (Wormhole, Blackhole, Grayskull chips). It consists of two main components:

- **TT-Metalium**: Low-level programming model for kernel development on Tenstorrent hardware
- **TT-NN (ttnn)**: High-level Python & C++ neural network operation library built on TT-Metalium

## Build Commands

```bash
# Standard build
./build_metal.sh

# Build with all tests
./build_metal.sh --build-tests

# Build specific test categories
./build_metal.sh --build-ttnn-tests
./build_metal.sh --build-metal-tests

# Build everything (tests, examples, tt-train)
./build_metal.sh --build-all

# Debug build
CONFIG=Debug ./build_metal.sh

# Clean build artifacts
./build_metal.sh --clean

# Manual CMake build
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja && ninja install
```

## Testing

```bash
# Run post-commit tests (required before merging)
./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type post_commit

# Run Python unit tests
pytest tests/ttnn/unit_tests/ -vvv

# Run specific pytest file
pytest tests/ttnn/unit_tests/operations/eltwise/test_add.py -vvv

# Run C++ gtest with filter
./build/test/tt_metal/unit_tests_api --gtest_filter="TestName*"

# Run slow dispatch mode tests
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests/unit_tests_api

# Model performance tests (bare metal)
./tests/scripts/run_tests.sh --tt-arch $ARCH_NAME --pipeline-type models_performance_bare_metal
```

## Linting

```bash
# Run clang-tidy
cmake --preset clang-tidy
cmake --build --preset clang-tidy

# Pre-commit hooks (install once)
pip install pre-commit
pre-commit install

# Run pre-commit manually
pre-commit run --all-files
```

## Architecture

### Core Components

```
tt_metal/          # TT-Metalium: core runtime, device APIs, allocators, low-level kernels
  ├── impl/        # Device implementation, dispatch, allocators
  ├── llrt/        # Low-level runtime, HAL
  ├── hw/          # Hardware firmware
  ├── jit_build/   # JIT kernel compilation
  ├── distributed/ # Multi-device coordination
  └── fabric/      # Network fabric for multi-chip

ttnn/              # TT-NN: high-level op layer and Python/C++ bindings
  ├── cpp/         # C++ implementation and nanobind Python bindings
  ├── ttnn/        # Python package
  └── tutorials/   # Jupyter notebook tutorials

tt-train/          # Training library built on top of ttnn
models/            # Production model implementations (LLMs, CNNs, etc.)
tools/             # Debugging and scaleout tools
```

### Tensix Programming Model

Each Tensix core has 5 RISC-V CPUs coordinating specialized hardware units. Typical kernels are:
- **Reader kernel**: Data input via NoC0
- **Compute kernel**: Matrix/vector operations using FPU/SFPU
- **Writer kernel**: Data output via NoC1

Kernels coordinate through circular buffers in SRAM (1.5MB L1 per core). Data flows: NoC → Unpacker → Compute → Packer → NoC.

### Hardware Targets

- **ARCH_NAME**: grayskull, wormhole, wormhole_b0, blackhole
- Multi-device configs: n150, n300 (Wormhole), p100, p150 (Blackhole), QuietBox (8-chip), Galaxy (32-chip)

## Code Style

- C++20 codebase with heavy template use; minimize compile times (forward declarations, PIMPL)
- Follow `.clang-format` and `.clang-tidy` rules
- SPDX license headers required on all source files
- Avoid macros when templates/constexpr suffice
- Avoid SFINAE/enable_if unless necessary
- Python: Black (line-length=120), isort

## Environment Variables

```bash
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=wormhole_b0  # or grayskull, blackhole

# Debugging
export TT_LOGGER_LEVEL=Debug
export TT_METAL_WATCHER=10              # Watcher checks every 10 seconds
export TT_METAL_DPRINT_CORES=(0,0)-(4,4) # Enable kernel printing
```

## Key Documentation

- [METALIUM_GUIDE.md](METALIUM_GUIDE.md): Architecture deep-dive and programming model
- [CONTRIBUTING.md](CONTRIBUTING.md): Development workflow, debugging guide, git conventions
- [tech_reports/](tech_reports/): Performance optimization guides for models
- [models/README.md](models/README.md): Model matrix with performance benchmarks

## Debugging Device Code

1. Enable Watcher for development: `TT_METAL_WATCHER=10 ./your_program`
2. Use DPRINT for kernel debugging: `#include "api/debug/dprint.h"` then `DPRINT << x << ENDL();`
3. Check watcher logs at `generated/watcher/watcher.log`
4. For hangs, use `./build/tools/watcher_dump --devices=<ids>`

## Hardware Reset

```bash
tt-smi -r 0        # Single card
tt-smi -r 0,1,2,3  # T3000/QuietBox
```
