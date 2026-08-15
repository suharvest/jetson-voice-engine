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
        expected_cute = (
            "ALL"
            if manifest_path.name == "qwen35-gdn-mtp-sm87-v010.toml"
            else "fmha"
        )
        assert data["build"]["enable_cute_dsl"] == expected_cute

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


def test_moss_and_spark_sources_are_immutable_and_weight_hash_locked():
    moss = tomllib.loads(
        (OVERLAY / "manifests/moss-tts-sm87-v010.toml").read_text(
            encoding="utf-8"
        )
    )
    assert moss["model"]["tts"] == {
        "repo": "OpenMOSS-Team/MOSS-TTS-Nano-100M",
        "revision": "44502f80dbf9743528fa921cc544d662c685ebec",
        "weights_sha256": "24003f2f11ac8a2cbf70514db2d8f1c02fb451aa6b3c0bffc9da09f31cd7caa5",
    }
    assert moss["model"]["codec"] == {
        "repo": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        "revision": "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68",
        "weights_sha256": "34d9880d805eecb21bde975202b1c256dbd0eb98c8680b9d3aeffd2bc6ac2f67",
    }
    assert moss["build"]["precision_recipe"] == "mix1"

    spark = tomllib.loads(
        (OVERLAY / "manifests/sparktts-sm87-v010.toml").read_text(
            encoding="utf-8"
        )
    )
    assert spark["source"]["revision"] == (
        "642071559bfc6346c2359d19dcb6be3f9dd8a05d"
    )
    assert spark["source"]["llm_weights_sha256"] == (
        "54825baf0a2f6076eb3c78fa1d22a95aee225f59070a8b295f8169db860eb109"
    )
    assert spark["source"]["bicodec_weights_sha256"] == (
        "e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec"
    )

    driver = (OVERLAY / "build-engines-for-device.sh").read_text(
        encoding="utf-8"
    )
    for artifact in (
        "moss_tts_prefill.plan",
        "moss_tts_decode_step.plan",
        "moss_tts_local_decoder.plan",
        "moss_tts_local_cached_step.plan",
        "moss_tts_local_fixed_sampled_frame.plan",
        "codec_decode_step.plan",
    ):
        assert artifact in driver
    assert "mix1-fp32-global-bf16-local-fp32-codec" in driver


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
    for model_id in (
        "qwen3.5-4b",
        "qwen3.5-4b-base",
        "qwen3.5-4b-mtp",
        "qwen3.5-4b-mtp-4k",
        "qwen3.5-4b-mtp-8k",
        "qwen3.5-4b-dflash",
    ):
        assert f"PREC[{model_id}]=int4_awq" in text
    assert 'QWEN35_FP8_KV_CACHE:-0' in text
    assert '*-mtp-4k) max_input=4096; max_kv=4096' in text
    assert '*-mtp-8k) max_input=8192; max_kv=8192' in text
    assert "run_gdn_builder" in text
    assert 'LD_PRELOAD="${gdn_plugin}' in text

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
    assert highperf["model"]["base"]["precision"] == "int4"
    assert highperf["model"]["base"]["int4_gemm_plugin_version"] == 1
    assert highperf["model"]["base"]["stage2_revision"] == (
        "ff2318e66525365b2ed9f55811bf5d2381280ed8"
    )
    assert 'TTS_INT4_GEMM_PLUGIN_VERSION="1"' in text
    int4_driver = OVERLAY / "drivers/export-qwen3-tts-int4-v010.sh"
    assert int4_driver.stat().st_mode & 0o111
    int4_driver_text = int4_driver.read_text(encoding="utf-8")
    assert f'PIN="{PIN}"' in int4_driver_text
    assert "TTS_INT4_STAGE2_CHECKPOINT" in int4_driver_text
    assert "TTS_INT4_STAGE2_REVISION" in int4_driver_text
    assert "--int4-gemm-plugin-version 1" in int4_driver_text
    for checksum_field in (
        "model_config_sha256",
        "model_weights_sha256",
        "stage2_config_sha256",
        "stage2_quant_config_sha256",
        "stage2_weights_sha256",
    ):
        assert checksum_field in int4_driver_text
    validator = (OVERLAY / "validate-tts-onnx.py").read_text(encoding="utf-8")
    assert 'expected_v1 = 196 if args.talker_int4_plugin_version == "1" else 0' in validator
    assert 'component_ops["Int4GroupwiseGemmPluginV2"]' in validator
    assert "--require-clone-encoders" in validator
    assert 'python3 "${HERE}/validate-tts-onnx.py"' in text

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


def test_v010_qwen35_orin_contract_preserves_awq_and_context_profiles():
    manifest = tomllib.loads(
        (OVERLAY / "manifests/qwen35-gdn-mtp-sm87-v010.toml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["build"]["enable_cute_dsl"] == "ALL"
    assert manifest["build"]["plugin_preload_required"] is True
    model = manifest["model"]
    assert model["precision"] == "int4_awq"
    assert model["quantization_algorithm"] == "W4A16_AWQ"
    assert model["quantization_group_size"] == 128
    assert model["int4_gemm_plugin_version"] == 1
    assert model["kv_cache_quantization"] == "none"
    assert manifest["profiles"]["4k"]["max_input_len"] == 4096
    assert manifest["profiles"]["4k"]["max_kv_cache_capacity"] == 4096
    assert manifest["profiles"]["8k"]["max_input_len"] == 8192
    assert manifest["profiles"]["8k"]["max_kv_cache_capacity"] == 8192
