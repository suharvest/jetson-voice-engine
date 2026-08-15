#!/usr/bin/env python3
"""Fail closed unless a v0.10 Qwen3-TTS export matches the product ABI."""

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
            raise SystemExit(f"missing or empty TTS export artifact: {path}")


def _check_model(path: Path) -> collections.Counter[str]:
    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path), load_external_data=False)
    return collections.Counter(node.op_type for node in graph.graph.node)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_root", type=Path)
    parser.add_argument(
        "--expected-tts-model-type",
        choices=("base", "custom_voice", "voice_design"),
        required=True,
    )
    parser.add_argument(
        "--talker-int4-plugin-version", choices=("none", "1"), required=True
    )
    parser.add_argument("--require-clone-encoders", action="store_true")
    args = parser.parse_args()
    root = args.export_root.resolve()

    required = (
        "llm/model.onnx",
        "llm/model.onnx.data",
        "llm/config.json",
        "llm/embedding.safetensors",
        "llm/tokenizer.json",
        "llm/processed_chat_template.json",
        "code_predictor/model.onnx",
        "code_predictor/model.onnx.data",
        "code_predictor/config.json",
        "code_predictor/codec_embeddings.safetensors",
        "code_predictor/lm_heads.safetensors",
        "code2wav/model.onnx",
        "code2wav/model.onnx.data",
        "code2wav/config.json",
    )
    if args.require_clone_encoders:
        required += (
            "clone_encoders/speaker_encoder.onnx",
            "clone_encoders/speaker_encoder.onnx.data",
            "clone_encoders/speech_tokenizer_encoder.onnx",
            "clone_encoders/speech_tokenizer_encoder.onnx.data",
        )
    _require_files(root, required)

    with (root / "llm/config.json").open(encoding="utf-8") as handle:
        talker_config = json.load(handle)
    actual_kind = talker_config.get("tts_model_type")
    if actual_kind != args.expected_tts_model_type:
        raise SystemExit(
            "wrong TTS model type: expected "
            f"{args.expected_tts_model_type!r}, got {actual_kind!r}"
        )

    model_paths = {
        "talker": root / "llm/model.onnx",
        "code_predictor": root / "code_predictor/model.onnx",
        "code2wav": root / "code2wav/model.onnx",
    }
    if args.require_clone_encoders:
        model_paths.update(
            {
                "speaker_encoder": root / "clone_encoders/speaker_encoder.onnx",
                "speech_tokenizer_encoder": root
                / "clone_encoders/speech_tokenizer_encoder.onnx",
            }
        )
    ops = {name: _check_model(path) for name, path in model_paths.items()}

    talker_v1 = ops["talker"]["Int4GroupwiseGemmPlugin"]
    all_v2 = sum(
        component_ops["Int4GroupwiseGemmPluginV2"]
        for component_ops in ops.values()
    )
    expected_v1 = 196 if args.talker_int4_plugin_version == "1" else 0
    if talker_v1 != expected_v1 or all_v2 != 0:
        raise SystemExit(
            "unsafe TTS INT4 backend: expected talker "
            f"v1={expected_v1} and all-components v2=0, got "
            f"talker v1={talker_v1}, v2={all_v2}"
        )
    non_talker_v1 = sum(
        component_ops["Int4GroupwiseGemmPlugin"]
        for name, component_ops in ops.items()
        if name != "talker"
    )
    if non_talker_v1 != 0:
        raise SystemExit(
            "unexpected INT4 plugin in non-Talker component: "
            f"v1={non_talker_v1}"
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "tts_model_type": actual_kind,
                "talker_int4_gemm_v1_nodes": talker_v1,
                "all_int4_gemm_v2_nodes": all_v2,
                "components": sorted(model_paths),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
