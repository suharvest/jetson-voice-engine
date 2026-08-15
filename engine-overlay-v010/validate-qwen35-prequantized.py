#!/usr/bin/env python3
"""Fail-closed contract for the qualified Qwen3.5 Orin AWQ checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-quant-config-sha256", required=True)
    args = parser.parse_args()

    root = args.checkpoint
    config_path = root / "config.json"
    quant_path = root / "hf_quant_config.json"
    model_path = root / "model.safetensors"
    revision_path = root / "SOURCE_REVISION"
    for path in (config_path, quant_path, model_path, revision_path):
        _require(path.is_file() and path.stat().st_size > 0, f"missing checkpoint file: {path}")

    revision = revision_path.read_text(encoding="utf-8").strip()
    _require(revision == args.expected_revision, f"checkpoint revision {revision!r} is not qualified")
    _require(_sha256(model_path) == args.expected_model_sha256, "model.safetensors SHA-256 mismatch")
    _require(_sha256(config_path) == args.expected_config_sha256, "config.json SHA-256 mismatch")
    _require(
        _sha256(quant_path) == args.expected_quant_config_sha256,
        "hf_quant_config.json SHA-256 mismatch",
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    hf_quant = json.loads(quant_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    config_quant = config.get("quantization_config", {})
    quant = hf_quant.get("quantization", {})
    producer = hf_quant.get("producer", {})

    _require(config.get("model_type") == "qwen3_5", "model_type must be qwen3_5")
    _require(text_config.get("num_hidden_layers") == 32, "expected 32 base layers")
    _require(text_config.get("mtp_num_hidden_layers") == 1, "expected one MTP layer")
    _require(config_quant.get("quant_algo") == "W4A16_AWQ", "config quant_algo must be W4A16_AWQ")
    _require(config_quant.get("quant_method") == "modelopt", "quant_method must be modelopt")
    _require(producer == {"name": "modelopt", "version": "0.42.0"}, "producer must be ModelOpt 0.42.0")
    _require(quant.get("quant_algo") == "W4A16_AWQ", "HF quant_algo must be W4A16_AWQ")
    _require(quant.get("group_size") == 128, "AWQ group_size must be 128")
    _require(quant.get("has_zero_point") is False, "AWQ zero point must be disabled")
    _require(quant.get("pre_quant_scale") is True, "AWQ pre_quant_scale must be enabled")
    _require(quant.get("kv_cache_quant_algo") is None, "KV cache quantization must remain disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
