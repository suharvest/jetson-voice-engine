from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
OVERLAY = ROOT / "engine-overlay"
PIN = "7f061f21f0a581ba234a1e233c9315b89d8e47d6"
SPEAKER_REL = "engines/tts_base_spk_encoder/speaker_encoder/spk_encoder.engine"
GDN_BASE_REL = "engines/gdn-base/llm.engine"
GDN_MTP_FILES = (
    "engines/gdn-mtp/spec_base.engine",
    "engines/gdn-mtp/spec_draft.engine",
    "engines/gdn-mtp/config.json",
    "engines/gdn-mtp/PROVENANCE.md",
)


def test_active_pin_and_patch_series_are_exact_and_contiguous():
    pin = next(
        line.strip()
        for line in (OVERLAY / "UPSTREAM_PIN").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert pin == PIN
    patches = sorted((OVERLAY / "patches" / "v091-candidate").glob("*.patch"))
    assert len(patches) == 41
    assert [int(p.name[:4]) for p in patches] == list(range(1, 42))


def test_build_wrapper_uses_only_active_series():
    text = (OVERLAY / "build.sh").read_text()
    assert 'PATCH_DIR="${HERE}/patches/v091-candidate"' in text
    assert "expected 41 v0.9.1 patches" in text
    active = text[text.index("# --- 3. apply the active") :]
    assert "patches/v090-sparktts-00*.patch" not in active
    assert 'apply_one "${HERE}/patches/0001-orin' not in active


def test_manifests_pin_v091_and_require_provenance():
    for path in sorted((OVERLAY / "manifests").glob("*.toml")):
        data = tomllib.loads(path.read_text())
        assert data["upstream"]["pin"] == PIN, path
        assert data["patches"]["count"] == 41, path
        assert data["artifacts"]["provenance_required"] is True, path
        assert data["artifacts"]["sha256_required"] is True, path


def test_engine_builder_has_explicit_mtp_fail_loud_hook():
    text = (OVERLAY / "build-engines-for-device.sh").read_text()
    assert "qwen3.5-4b-base" in text
    assert "qwen3.5-4b-mtp" in text
    assert "EDGELLM_MTP_BUILD_SCRIPT" in text
    for name in ("MTP_UPSTREAM", "MTP_MODEL", "MTP_EXPORT", "MTP_SPEC_DIR", "MTP_PLUGIN"):
        assert name in text
    assert "MTP_BASE_ENGINE_DIR" not in text
    assert "MTP_DRAFT_ENGINE_DIR" not in text
    assert "MTP_MODEL_DIR" not in text
    assert "MTP_EXPORT_DIR" not in text
    assert "MTP_SPEC_ENGINE_DIR" not in text
    assert "MTP_PLUGIN_PATH" not in text
    assert 'build_llm_base "${m}"' not in text
    assert "spec_base.engine" in text and "spec_draft.engine" in text
    assert "config.json" in text and "PROVENANCE.md" in text
    assert re.search(r"exit 6", text)
    assert "thinker-b1" in text and "thinker-b2" in text
    assert "tensorrt-edgellm-export-audio " not in text
    assert "QWEN3_TTS_BASE_SPEAKER_ENCODER_ONNX_DIR" in text
    assert "QWEN3_TTS_BASE_SPEAKER_ENCODER_REVISION" in text
    assert "QWEN3_TTS_BASE_SPEAKER_ENCODER_SHA256" in text
    assert "--minShapes=mel:1x10x128" in text
    assert "--optShapes=mel:1x555x128" in text
    assert "--maxShapes=mel:1x2000x128" in text
    assert re.search(r"exit 11", text)
    assert "EDGELLM_TTS_BASE_DRIVER" in text
    for name in (
        "TTS_BASE_UPSTREAM",
        "TTS_BASE_MODEL",
        "TTS_BASE_OUTPUT",
        "TTS_BASE_PRECISION",
    ):
        assert name in text
    assert re.search(r"exit 12", text)


def test_voice_build_contracts_fail_loud_and_keep_production_asr_shape():
    engines = (OVERLAY / "build-engines-for-device.sh").read_text()
    build = (OVERLAY / "build.sh").read_text()
    tts_manifest = tomllib.loads(
        (OVERLAY / "manifests/qwen3-tts-highperf-sm87.toml").read_text()
    )
    asr_manifest = tomllib.loads(
        (OVERLAY / "manifests/qwen3-asr-sm87.toml").read_text()
    )
    dockerfile = (
        WORKSPACE / "deploy/docker/Dockerfile.jetson.edgellm-v091-overlay"
    ).read_text()

    assert "tts-int4-drivers" not in engines
    assert "EDGELLM_TTS_INT4_DRIVER" in engines
    for required in ("DRIVER_REVISION", "PROVENANCE.md"):
        assert required in engines
        assert required in tts_manifest["model"]["customvoice"]["required_driver_outputs"]
        assert f"tts_customvoice_int4/talker/{required}" in dockerfile

    assert 'max_input="${ASR_MAX_INPUT_LEN:-1024}"' in engines
    assert 'max_kv="${ASR_MAX_KV_CACHE_CAPACITY:-1536}"' in engines
    assert "ASR_LONG_CONTEXT" in engines
    assert 'suffix="-longctx-${max_input}-${max_kv}"' in engines
    assert 'thinker-b1${suffix}' in engines and 'thinker-b2${suffix}' in engines
    assert asr_manifest["model"]["max_input_len"] == 1024
    assert asr_manifest["model"]["max_kv_cache_capacity"] == 1536
    assert asr_manifest["model"]["long_context_opt_in_env"] == "ASR_LONG_CONTEXT"

    assert "--target audio_build" in build
    assert 'build/examples/multimodal/audio_build" ]; then' in build
    assert 'examples/multimodal/audio_build"; do' in engines
    assert 'LD_PRELOAD="${PLUGIN}' in engines
    assert engines.count("run_builder ") >= 6


def test_base_driver_is_pinned_fail_loud_and_provenance_complete():
    driver = OVERLAY / "drivers/export-qwen3-tts-base-v091.sh"
    text = driver.read_text()
    manifest = tomllib.loads(
        (OVERLAY / "manifests/qwen3-tts-highperf-sm87.toml").read_text()
    )

    assert driver.exists()
    assert PIN in text
    assert "TTS_BASE_MODEL_REVISION" in text
    assert "TTS_BASE_STAGE2_CHECKPOINT" in text
    assert "TTS_BASE_STAGE2_REVISION" in text
    assert 'config.get("tts_model_type") != "base"' in text
    assert "onnx.checker.check_model" in text
    assert 'PYTHONPATH="${upstream}' in text
    assert "python3 -m tensorrt_edgellm.scripts.export" in text
    assert 'driver_sha="$(sha256sum "${BASH_SOURCE[0]}"' in text
    assert "SHA256SUMS" in text
    assert "tensorrt-edgellm-export-audio" not in text
    assert "tensorrt-edgellm-export-llm" not in text

    base = manifest["model"]["base"]
    assert base["precision"] == "int4"
    assert base["required_driver_env"] == "EDGELLM_TTS_BASE_DRIVER"
    assert base["required_model_revision_env"] == "TTS_BASE_MODEL_REVISION"
    assert set(base["required_int4_stage2_env"]) == {
        "TTS_BASE_STAGE2_CHECKPOINT",
        "TTS_BASE_STAGE2_REVISION",
    }


def test_base_driver_refuses_unmodified_official_exporter(tmp_path):
    driver = OVERLAY / "drivers/export-qwen3-tts-base-v091.sh"
    upstream = tmp_path / "upstream"
    model = tmp_path / "base-model"
    fake_bin = tmp_path / "bin"
    output = tmp_path / "output"
    (upstream / ".git").mkdir(parents=True)
    (upstream / "tensorrt_edgellm/scripts").mkdir(parents=True)
    (upstream / "tensorrt_edgellm/scripts/export.py").write_text(
        'message = "Only Qwen3-TTS CustomVoice checkpoints are supported"\n'
    )
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3_tts", "tts_model_type": "base"})
    )
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(f"#!/bin/sh\nprintf '%s\\n' '{PIN}'\n")
    fake_git.chmod(0o755)

    result = subprocess.run(
        [str(driver)],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "TTS_BASE_UPSTREAM": str(upstream),
            "TTS_BASE_MODEL": str(model),
            "TTS_BASE_MODEL_REVISION": "immutable-test-revision",
            "TTS_BASE_OUTPUT": str(output),
            "TTS_BASE_PRECISION": "fp16",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 12
    assert "Base exporter extension is not applied" in result.stderr
    assert not output.exists()


def test_base_export_patch_is_upstream_neutral():
    patch = (
        OVERLAY
        / "patches/v091-candidate/"
        "0035-fix-export-demote-Qwen3-TTS-non-CustomVoice-guard-to.patch"
    ).read_text()

    assert "CustomVoice and Base checkpoints are" in patch
    assert "does not include a speaker" in patch
    assert 'tts_model_type not in ("custom_voice", "base")' in patch
    assert "P3a local patch" not in patch
    assert "v0.8.0 port" not in patch


def test_export_wrappers_use_v091_unified_cli(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    asr_out = tmp_path / "asr"
    tts_out = tmp_path / "tts"
    asr = ROOT / "scripts/export_qwen3_asr_onnx.sh"
    tts = ROOT / "scripts/export_qwen3_tts_onnx.sh"
    setup = ROOT / "scripts/setup_trt_export_env.sh"

    for path in (asr, tts, setup):
        text = path.read_text()
        assert "tensorrt-edgellm-export-llm" not in text
        assert "tensorrt-edgellm-export-audio" not in text
        assert "tensorrt-edgellm-export" in text

    asr_result = subprocess.run(
        [str(asr), "--model-dir", str(model), "--out", str(asr_out), "--dry-run"],
        text=True,
        capture_output=True,
    )
    assert asr_result.returncode == 0, asr_result.stderr
    assert "--components thinker,audio" in asr_result.stdout

    tts_result = subprocess.run(
        [
            str(tts),
            "--model-dir",
            str(model),
            "--out",
            str(tts_out),
            "--official-only",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )
    assert tts_result.returncode == 0, tts_result.stderr
    assert "--components talker,code_predictor,code2wav" in tts_result.stdout


def test_spark_routes_are_separate_and_token_gated():
    text = (OVERLAY / "build-engines-for-device.sh").read_text()
    manifest = tomllib.loads(
        (OVERLAY / "manifests/sparktts-sm87-v091.toml").read_text()
    )
    dockerfile = (
        WORKSPACE / "deploy/docker/Dockerfile.jetson.edgellm-v091-overlay"
    ).read_text()
    assert "sparktts-bf16" in text
    assert "sparktts-w4a16" in text
    assert "EDGELLM_SPARK_LLM_DRIVER" in text
    assert "validate-spark-token-gate.py" in text
    assert "v0.9.0 SparkTTS mixed-precision export regression" not in text
    assert manifest["model"]["bf16"]["mode"] == "bf16"
    assert manifest["model"]["w4a16"]["mode"] == "w4a16"
    assert manifest["driver"]["token_gate_global_count"] == 32
    for mode in ("bf16", "w4a16"):
        for name in ("llm.engine", "token-gate.json", "DRIVER_REVISION", "PROVENANCE.md"):
            rel = f"engines/sparktts-{mode}/{name}"
            assert rel in manifest["artifacts"]["required"]
            assert f"/opt/edgellm-v091/{rel}" in dockerfile


def test_cuda_probe_rejects_newer_torch_runtime():
    probe = OVERLAY / "probe-cuda-compat.py"
    env = {
        **os.environ,
        "EDGELLM_PROBE_DRIVER_VERSION": "12060",
        "EDGELLM_PROBE_TORCH_CUDA": "13.0",
    }
    rejected = subprocess.run(
        [sys.executable, str(probe)], env=env, text=True, capture_output=True
    )
    assert rejected.returncode == 7
    assert "newer than the driver" in rejected.stderr

    env["EDGELLM_PROBE_TORCH_CUDA"] = "12.6"
    accepted = subprocess.run(
        [sys.executable, str(probe)], env=env, text=True, capture_output=True
    )
    assert accepted.returncode == 0


def test_spark_token_gate_requires_mode_and_32_tokens(tmp_path):
    validator = OVERLAY / "validate-spark-token-gate.py"
    gate = tmp_path / "token-gate.json"
    gate.write_text(
        json.dumps({"passed": True, "mode": "bf16", "global_token_count": 32})
    )
    passed = subprocess.run(
        [sys.executable, str(validator), str(gate), "bf16"],
        text=True,
        capture_output=True,
    )
    assert passed.returncode == 0

    failed = subprocess.run(
        [sys.executable, str(validator), str(gate), "w4a16"],
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 10


def test_speaker_encoder_release_path_matches_build_manifest_profile_and_image():
    build = (OVERLAY / "build-engines-for-device.sh").read_text()
    trt10_fix = OVERLAY / "fix-qwen3-tts-speaker-encoder-trt10.py"
    manifest = tomllib.loads(
        (OVERLAY / "manifests" / "qwen3-tts-highperf-sm87.toml").read_text()
    )
    profile = json.loads(
        (
            WORKSPACE
            / "configs/profiles/jetson-edgellm-v091-qwen3ttsbase.json"
        ).read_text()
    )
    dockerfile = (
        WORKSPACE / "deploy/docker/Dockerfile.jetson.edgellm-v091-overlay"
    ).read_text()

    assert (
        'spk_engine_dir="${EXPORT_ROOT}/tts_base_spk_encoder/speaker_encoder"'
        in build
    )
    assert trt10_fix.is_file()
    assert "model.source.onnx" in build
    assert str(trt10_fix.name) in build
    assert "SHA256SUMS.trt10" in build
    assert 'node.name == "/enc/If"' in trt10_fix.read_text()
    assert SPEAKER_REL in manifest["artifacts"]["required"]
    assert f"/opt/edgellm-v091/{SPEAKER_REL}" in {
        item["engine_path"] for item in profile["required_engines"]
    }
    assert f"/opt/edgellm-v091/{SPEAKER_REL}" in dockerfile


def test_gdn_base_and_mtp_release_paths_match_builder_and_image():
    build = (OVERLAY / "build-engines-for-device.sh").read_text()
    dockerfile = (
        WORKSPACE / "deploy/docker/Dockerfile.jetson.edgellm-v091-overlay"
    ).read_text()

    assert 'out="${EXPORT_ROOT}/gdn-base"' in build
    assert 'spec_dir="${MTP_SPEC_DIR:-${EXPORT_ROOT}/gdn-mtp}"' in build
    assert f"/opt/edgellm-v091/{GDN_BASE_REL}" in dockerfile
    for rel in GDN_MTP_FILES:
        assert f"/opt/edgellm-v091/{rel}" in dockerfile
