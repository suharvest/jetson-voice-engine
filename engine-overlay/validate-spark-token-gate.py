#!/usr/bin/env python3
"""Validate the SparkTTS semantic-token acceptance record."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate-spark-token-gate.py <token-gate.json> <mode>", file=sys.stderr)
        return 2
    path, expected_mode = Path(sys.argv[1]), sys.argv[2]
    try:
        gate = json.loads(path.read_text())
    except Exception as exc:
        print(f"ERROR: invalid Spark token gate: {exc}", file=sys.stderr)
        return 10
    if (
        gate.get("passed") is not True
        or gate.get("mode") != expected_mode
        or gate.get("global_token_count") != 32
    ):
        print(
            "ERROR: Spark token gate requires passed=true, matching mode, "
            "and global_token_count=32",
            file=sys.stderr,
        )
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
