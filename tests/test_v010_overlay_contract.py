from __future__ import annotations

import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "engine-overlay-v010"
PIN = "71dd1bae032e70771265917ec74d3ff4cad07a10"


def _series_entries(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_entries(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        assert len(fields) == 2, line
        entries.append((fields[1], fields[0]))
    return entries


def _assert_series_and_hashes(directory: Path, expected_count: int) -> list[str]:
    series = _series_entries(directory / "series")
    patches = sorted(path.name for path in directory.glob("*.patch"))
    assert len(series) == expected_count
    assert sorted(series) == patches

    checksums = _checksum_entries(directory / "SHA256SUMS")
    assert [name for name, _ in checksums] == series
    assert len(checksums) == expected_count
    for name, expected in checksums:
        assert _sha256(directory / name) == expected, name
    return series


def test_v010_pin_and_4_plus_32_series_are_hash_locked():
    pin = next(
        line.strip()
        for line in (OVERLAY / "UPSTREAM_PIN").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert pin == PIN

    upstream = _assert_series_and_hashes(
        OVERLAY / "patches/upstream-v010-prs", expected_count=4
    )
    product = _assert_series_and_hashes(
        OVERLAY / "patches/v010-candidate", expected_count=32
    )

    assert upstream == [
        "0001-pr118-respect-explicit-cuda-architectures.patch",
        "0002-pr118-static-target-wrap-interface.patch",
        "0003-pr146-normalize-linear-mrope.patch",
        "0004-pr149-checkpoint-dtype.patch",
    ]
    assert all(name.startswith(("000", "001", "002", "003", "004")) for name in product)


def test_retired_generic_and_product_patches_are_not_active():
    upstream_series = _series_entries(
        OVERLAY / "patches/upstream-v010-prs/series"
    )
    for retired in (
        "0003-pr145-trt-fp4-guard.patch",
        "0005-pr147-trt-stream-reader.patch",
        "0006-pr148-fmha-mask-scoped-load.patch",
    ):
        assert retired not in upstream_series

    product_dir = OVERLAY / "patches/v010-candidate"
    product_series = _series_entries(product_dir / "series")
    for prefix in ("0024-", "0035-", "0036-"):
        assert not any(name.startswith(prefix) for name in product_series)
    state = (product_dir / "PATCH-STATE.md").read_text(encoding="utf-8")
    for retired in ("0024", "0035", "0036"):
        assert retired in state
    assert "Retired on v0.10.0" in state


def test_native_voice_clone_export_is_host_portable():
    patch = (
        OVERLAY
        / "patches/v010-candidate/0025-feat-qwen3-tts-native-voice-clone.patch"
    ).read_text(encoding="utf-8")
    assert 'device: str | None = None' in patch
    assert 'device = "cuda" if torch.cuda.is_available() else "cpu"' in patch
    assert 'mimi_config._attn_implementation = "eager"' in patch
    assert "auto const dtype = chunkAudio.waveform->getDataType()" in patch
    assert "dtype == nvinfer1::DataType::kFLOAT" in patch
    assert "dtype == nvinfer1::DataType::kHALF" in patch
    assert 'nvinfer1::DataType::kFLOAT, "streaming_pcm"' in patch


def test_v010_manifests_have_release_provenance_and_fresh_hashes():
    upstream_dir = OVERLAY / "patches/upstream-v010-prs"
    product_dir = OVERLAY / "patches/v010-candidate"
    for manifest_path in sorted((OVERLAY / "manifests").glob("*.toml")):
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["upstream"]["version"] == "v0.10.0", manifest_path
        assert data["upstream"]["pin"] == PIN, manifest_path
        assert data["upstream"]["remote"] == (
            "https://github.com/NVIDIA/TensorRT-Edge-LLM.git"
        )
        assert data["proposed_upstream_patches"]["count"] == 4
        assert data["patches"]["count"] == 32
        assert data["patches"]["directory"] == "patches/v010-candidate"
        assert data["artifacts"]["provenance_required"] is True
        assert data["artifacts"]["sha256_required"] is True
        assert data["target"] == {
            "device": "jetson-orin-nx",
            "sm": "87",
            "jetpack": "6.2",
            "l4t": "36.4.3",
            "cuda": "12.6",
            "tensorrt": "10.3",
            "embedded_target": "jetson-orin",
            "aarch64_build": True,
        }
        assert data["build"]["type"] == "Release"
        assert data["build"]["enable_cute_dsl"] == "fmha"

        proposed = data["proposed_upstream_patches"]
        assert proposed["series_sha256"] == _sha256(upstream_dir / "series")
        assert proposed["lock_sha256"] == _sha256(upstream_dir / "LOCK")
        assert proposed["checksums_sha256"] == _sha256(
            upstream_dir / "SHA256SUMS"
        )
        product = data["patches"]
        assert product["series_sha256"] == _sha256(product_dir / "series")
        assert product["checksums_sha256"] == _sha256(
            product_dir / "SHA256SUMS"
        )

        result = subprocess.run(
            [
                sys.executable,
                str(OVERLAY / "validate-manifest.py"),
                str(OVERLAY),
                str(manifest_path),
                PIN,
            ],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr


def test_build_wrapper_replays_exact_counts_and_fails_closed():
    build = OVERLAY / "build.sh"
    text = build.read_text(encoding="utf-8")
    assert (
        'load_series "${UPSTREAM_PATCH_DIR}" "${UPSTREAM_PATCH_DIR}/series" 4 '
        '"proposed-upstream"'
    ) in text
    assert (
        'load_series "${PATCH_DIR}" "${PATCH_DIR}/series" 32 '
        '"local-product"'
    ) in text
    assert '[ -f "${PATCH_DIR}/series" ]' in text
    assert "ERROR: missing v0.10 product patch series" in text
    assert "exit 5" in text

    missing_manifest = subprocess.run(
        [str(build)], text=True, capture_output=True
    )
    assert missing_manifest.returncode != 0
    assert "build manifest required" in missing_manifest.stderr


def test_readme_records_v010_identity_and_runtime_constraints():
    readme = (OVERLAY / "README.md").read_text(encoding="utf-8")
    for required in (
        "TensorRT-Edge-LLM v0.10.0",
        "patches/upstream-v010-prs/",
        "patches/v010-candidate/",
        "complete 4+32",
        "Existing v0.9.1 images and artifacts remain the",
        "CuTe FMHA-v2",
        "ENABLE_CUTE_DSL=fmha",
    ):
        assert required in readme
    assert "v0.9.1 patch files are" in readme


def test_v010_aggregate_engine_driver_is_pin_and_revision_locked():
    driver = OVERLAY / "build-engines-for-device.sh"
    text = driver.read_text(encoding="utf-8")
    assert f'EXPECTED_UPSTREAM_PIN="{PIN}"' in text
    assert "not qualified for v0.10.0" not in text
    assert "native v0.10 export --mtp" in text
    assert "--specBase" in text
    assert "--specDraft" in text
    assert "speaker_encoder.engine" in text
    assert "speech_tokenizer_encoder.engine" in text
    assert "EDGELLM_TTS_BASE_DRIVER" not in text
    assert "QWEN3_TTS_BASE_SPEAKER_ENCODER_ONNX_DIR" not in text
    assert "--revision" in text
    for revision in (
        "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "85e237c12c027371202489a0ec509ded67b5e4b5",
        "5d83992436eae1d760afd27aff78a71d676296fc",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "5ecdb67327fd37bb2e042aab12ff7391903235d3",
        "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf",
    ):
        assert revision in text
    assert "qwen3-tts-int4" in text
    assert 'PREC[qwen3-tts]=fp16' in text
    assert "build_llm_dflash" in text
    assert "--dflash-base" in text
    assert "--dflash-draft" in text
    assert "base_config.json" in text
    assert "draft_config.json" in text

    customvoice = tomllib.loads(
        (OVERLAY / "manifests/customvoice-v010.toml").read_text(encoding="utf-8")
    )
    highperf = tomllib.loads(
        (OVERLAY / "manifests/qwen3-tts-highperf-sm87-v010.toml").read_text(
            encoding="utf-8"
        )
    )
    for manifest in (customvoice, highperf):
        int4_artifacts = [
            path
            for path in manifest["artifacts"]["required"]
            if "qwen3-tts-int4" in path
        ]
        assert len(int4_artifacts) == 5
    assert customvoice["model"]["int4_gemm_plugin_version"] == 1
    assert highperf["model"]["customvoice"]["int4_gemm_plugin_version"] == 1
    assert 'TTS_INT4_GEMM_PLUGIN_VERSION="1"' in text

    result = subprocess.run([str(driver)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "set UPSTREAM" in result.stderr


def test_v010_asr_worker_fails_closed_on_unsafe_native_audio_batch():
    worker = (
        ROOT / "native/edgellm_voice_worker/qwen3_asr_worker.cpp"
    ).read_text(encoding="utf-8")
    assert "native_audio_batch_unsafe_v010" in worker
    assert "batch.requests.size() > 1" in worker
    assert "worker finalizations are serialized for audio isolation" in worker


def test_v010_asr_quantization_contract_is_explicit_and_frozen():
    manifest = tomllib.loads(
        (OVERLAY / "manifests/qwen3-asr-sm87.toml").read_text(
            encoding="utf-8"
        )
    )
    model = manifest["model"]
    assert model["precision"] == "int4_awq"
    assert model["quantization_algorithm"] == "W4A16_AWQ"
    assert model["quantization_group_size"] == 128
    assert model["int4_gemm_plugin_version"] == 1
    assert model["calibration_dataset"] == "librispeech"
    assert model["calibration_samples"] == 128
    assert model["audio_tower_precision"] == "fp16"
    assert model["lm_head_quantized"] is False

    driver = (OVERLAY / "build-engines-for-device.sh").read_text(
        encoding="utf-8"
    )
    assert "--audio_dataset librispeech" in driver
    assert "--num_samples 128" in driver
    assert "--int4-gemm-plugin-version 1" in driver

    validator = (OVERLAY / "validate-asr-onnx.py").read_text(
        encoding="utf-8"
    )
    assert 'ops["Int4GroupwiseGemmPlugin"]' in validator
    assert 'ops["Int4GroupwiseGemmPluginV2"]' in validator
    assert "v1_count != 196 or v2_count != 0" in validator
