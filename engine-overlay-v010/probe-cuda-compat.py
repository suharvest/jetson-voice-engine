#!/usr/bin/env python3
"""Fail when the quant/export torch CUDA runtime is newer than the GPU driver."""

from __future__ import annotations

import ctypes
import os
import sys


def _driver_version() -> int:
    override = os.environ.get("EDGELLM_PROBE_DRIVER_VERSION")
    if override:
        return int(override)
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0) != 0:
        raise RuntimeError("cuInit failed")
    value = ctypes.c_int()
    if cuda.cuDriverGetVersion(ctypes.byref(value)) != 0:
        raise RuntimeError("cuDriverGetVersion failed")
    return value.value


def _torch_cuda() -> str:
    override = os.environ.get("EDGELLM_PROBE_TORCH_CUDA")
    if override:
        return override
    import torch

    if not torch.version.cuda:
        raise RuntimeError("torch is not a CUDA build")
    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA is unavailable with the installed driver")
    return torch.version.cuda


def main() -> int:
    try:
        driver = _driver_version()
        torch_cuda = _torch_cuda()
        driver_major = driver // 1000
        torch_major = int(torch_cuda.split(".", 1)[0])
    except Exception as exc:
        print(f"ERROR: CUDA compatibility probe failed: {exc}", file=sys.stderr)
        return 7

    print(f"CUDA compatibility: driver={driver} torch={torch_cuda}")
    if torch_major > driver_major:
        print(
            "ERROR: quant/export torch CUDA runtime is newer than the driver "
            f"({torch_cuda} > driver major {driver_major}); refusing to run",
            file=sys.stderr,
        )
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
