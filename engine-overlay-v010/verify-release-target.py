#!/usr/bin/env python3
"""Fail closed unless a probed build host matches the v0.10.0 release tuple."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--embedded-target", required=True)
    parser.add_argument("--l4t", required=True)
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--tensorrt", required=True)
    args = parser.parse_args()

    expected = {
        "sm": "87",
        "platform": "tegra",
        "embedded_target": "jetson-orin",
        "l4t": "36.4.3",
        "cuda": "12.6",
    }
    actual = vars(args)
    mismatches = [
        f"{key}={actual[key]!r} (required {wanted!r})"
        for key, wanted in expected.items()
        if actual[key] != wanted
    ]
    if not args.tensorrt.startswith("10.3"):
        mismatches.append(
            f"tensorrt={args.tensorrt!r} (required 10.3.x)"
        )
    if mismatches:
        parser.error("release target mismatch: " + "; ".join(mismatches))
    print(
        "release target: PASS "
        f"(SM87, tegra/jetson-orin, L4T {args.l4t}, "
        f"CUDA {args.cuda}, TensorRT {args.tensorrt})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
