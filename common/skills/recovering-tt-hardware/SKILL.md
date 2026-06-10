---
name: recovering-tt-hardware
description: "Recover Tenstorrent hardware (Wormhole, Blackhole, Grayskull) from a wedged/bad state via tt-smi reset, and fall back to tt-flash firmware reflash when reset fails. Use when a test or program fails with a device RuntimeError (hang, NoC timeout, unrecoverable assertion, \"device is wedged\") and re-running reproduces the error."
---

# Recovering Tenstorrent Hardware

Recovery procedure for Tenstorrent accelerators (Wormhole, Blackhole, Grayskull)
that are stuck in a bad state. This applies to any project running on TT hardware
(tt-metal, tt-forge, etc.), not just one codebase.

Escalate in order: **(1) reset → (2) firmware reflash**. Do not skip to reflash
before trying reset.

## When to recover

If a test or program fails with a `RuntimeError` indicating the device is in a
bad state — e.g. hang, NoC timeout, unrecoverable assertion, "device is wedged" —
the device must be recovered before retrying. Re-attempting without recovery
typically reproduces the same error.

## Step 1: Reset with tt-smi

```bash
tt-smi -ls               # List devices and confirm PCI Dev IDs
tt-smi -r <device_id>    # Reset a single device, e.g. tt-smi -r 0
tt-smi -r 0,1,2,3        # Reset multiple devices (T3000 / QuietBox)
```

Use the PCI Dev ID shown by `tt-smi -ls`. Always reset before re-running the
failing test.

## Step 2: Firmware Reflash (fallback when reset fails)

If `tt-smi -r <device_id>` cannot recover the HW state (the device stays wedged
after a reset), reflash the firmware as the next-best recovery step. Run
`tt-flash` from the same path/environment where `tt-smi` is installed:

```bash
tt-flash flash <fw_bundle_file> --force
```

`--force` reflashes even when the installed firmware version matches.

### Getting the firmware bundle

Firmware bundles are published at
https://github.com/tenstorrent/tt-system-firmware/releases — each release
provides a `.fwbundle` file following the naming pattern:

```
https://github.com/tenstorrent/tt-system-firmware/releases/download/v<version>/fw_pack-<version>.fwbundle
# e.g. for v19.11.0-rc1:
#   .../download/v19.11.0-rc1/fw_pack-19.11.0-rc1.fwbundle
```

Pass the downloaded `.fwbundle` file as the argument to `tt-flash flash`. Pick
the `<version>` that matches what is already installed (see below) — the example
version above is illustrative, not a version to install.

### Important: reflash the SAME version, do not upgrade

Do **not** upgrade to the latest firmware. Reflash the *same* firmware version
that is already installed:

1. Check the current installed version with `tt-smi -ls`.
2. Download the matching `.fwbundle` from the releases page.
3. Reflash it with `tt-flash flash <matching.fwbundle> --force`.

The goal is to recover the HW state, not to change the firmware version. Use this
only after `tt-smi -r` has failed to clear the bad state.

**Caution:** A firmware reflash is destructive. Confirm you are targeting the
correct device, and do not interrupt `tt-flash` while it is writing.

After reflashing, re-run the failing test to confirm the device has recovered.
