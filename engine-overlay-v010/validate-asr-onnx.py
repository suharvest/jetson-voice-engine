#!/usr/bin/env python3
"""Fail closed unless a v0.10 Qwen3-ASR export matches the product ABI."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import onnx


def _require_files(root: Path, relative_paths: tuple[str, ...]) -> None:
    for relative in relative_paths:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing or empty ASR export artifact: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_root", type=Path)
    args = parser.parse_args()
    root = args.export_root.resolve()

    _require_files(
        root,
        (
            "llm/model.onnx",
            "llm/model.onnx.data",
            "llm/config.json",
            "llm/embedding.safetensors",
            "llm/tokenizer.json",
            "llm/processed_chat_template.json",
            "audio/model.onnx",
            "audio/config.json",
        ),
    )

    llm_path = root / "llm/model.onnx"
    audio_path = root / "audio/model.onnx"
    onnx.checker.check_model(str(llm_path))
    onnx.checker.check_model(str(audio_path))

    llm = onnx.load(str(llm_path), load_external_data=False)
    audio = onnx.load(str(audio_path), load_external_data=False)
    ops = collections.Counter(node.op_type for node in llm.graph.node)
    v1_count = ops["Int4GroupwiseGemmPlugin"]
    v2_count = ops["Int4GroupwiseGemmPluginV2"]
    if v1_count != 196 or v2_count != 0:
        raise SystemExit(
            "unsafe ASR INT4 backend: expected 196 legacy v1 nodes and no "
            f"CuTe-DSL v2 nodes, got v1={v1_count} v2={v2_count}"
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "llm_nodes": len(llm.graph.node),
                "audio_nodes": len(audio.graph.node),
                "int4_gemm_v1_nodes": v1_count,
                "int4_gemm_v2_nodes": v2_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
