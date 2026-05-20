# Tracy-enabled Build Setup

## Objective

Build TT-Metal with Tracy instrumentation enabled and verify the
profiler binaries are on `PATH`.

## Why a separate build?

Tracy instrumentation adds device-side markers (`DeviceZoneScopedMainChildN`,
host-side scope macros). On a non-Tracy build, the profiler module
(`python -m tracy ...`) exits with:

```
ERROR | tracy:run_report_setup:115 -
Tracy tools were not found. Please make sure you are on a Tracy-enabled build (default).
```

`ENABLE_TRACY=ON` is the cmake flag that turns it on. The standard
`./build_metal.sh` *does* enable Tracy by default, but if you have a
custom build (e.g. `build_Release` vs `build`), confirm the flag.

## Build with Tracy

```bash
./build_metal.sh                  # default is RelWithDebInfo + Tracy ON
# or, manually:
mkdir build_Release && cd build_Release
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TRACY=ON
ninja && ninja install
```

Verify after build:

```bash
grep ENABLE_TRACY build_Release/CMakeCache.txt
# ENABLE_TRACY:BOOL=ON
```

## Profiler binaries

Tracy ships two host-side binaries that the Python wrapper invokes for
post-processing:

```
build_Release/tools/profiler/bin/capture-release
build_Release/tools/profiler/bin/csvexport-release
```

The Python wrapper looks them up via `PATH`. The typical mistake is
forgetting to extend `PATH`:

```bash
export PATH=$TT_METAL_HOME/build_Release/tools/profiler/bin:$PATH
which capture-release csvexport-release
```

If these are missing, `python -m tracy -r ...` will report:

```
ERROR | tracy:run_report_setup:115 - Tracy tools were not found.
```

## Standard environment for profiling

```bash
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=blackhole  # or wormhole_b0
export PATH=$TT_METAL_HOME/build_Release/tools/profiler/bin:$PATH
source python_env/bin/activate

# sanity
python -c "import ttnn; print(ttnn.__version__ if hasattr(ttnn,'__version__') else 'ok')"
which capture-release
```

## Why two build directories sometimes appear (`build` vs `build_Release`)

`build_metal.sh` sets the build dir based on `CONFIG`:

- `CONFIG=Debug ./build_metal.sh` → `build` (Debug)
- default → `build_Release` (RelWithDebInfo)

When swapping between configs, **double-check which directory has the
Tracy binaries** before adding to `PATH`. The host-side .so used at
runtime is normally taken from the most recent install, but the
profiler binaries are pinned to whichever build you point `PATH` at.

## Reset before first run

```bash
tt-smi -ls           # list devices
tt-smi -r 0          # reset device 0
```

A fresh reset avoids inheriting state from a previous run that may have
been profiling-instrumented.

## Common gotchas

- **Mixing branches and build**: Building from `main`, then running a
  feature branch that uses an older API of an op (e.g. `ttnn.grid_sample`
  signature change) causes `TypeError` mid-run. The profiler still
  generates a CSV up to the failure point, but the CSV is partial. Fix
  the API call (or check out a matching build commit) before profiling.
- **JIT compile errors during profiling** (e.g. `compile_time_args.h:27
  static assertion failed: Index out of range`) indicate kernel-side
  Tracy macros expanding into an older kernel that has fewer compile-time
  args. Same root cause as above: branch / build mismatch. Rebuild or
  fix the kernel.

## Checklist

- [ ] `ENABLE_TRACY=ON` in CMakeCache
- [ ] `capture-release` and `csvexport-release` on `PATH`
- [ ] `TT_METAL_HOME` and `PYTHONPATH` set
- [ ] `ARCH_NAME` matches hardware
- [ ] `python_env` activated
- [ ] Device reset
