#!/usr/bin/env bash
# build-engines-for-device.sh — self-service engine builder for a NEW device.
#
# Codifies the manual DGX-Spark (GB10) bring-up into one command so any new
# device (Thor sm_110, a desktop RTX50 sm_120, a future Jetson) reproduces the
# full engine set with just its detected arch. Replaces the old model:
# "engines are pre-published per device-class on HF; unknown device -> hard
# fail" (engine_resolver F1). This IS the local build path that resolver
# intentionally lacked.
#
# SCOPE — the edge-llm TensorRT family (ONE quantize -> export -> llm_build /
# audio_build pipeline). Add a model = add a MODELS[] manifest line:
#   qwen3-asr       | Qwen/Qwen3-ASR-0.6B                    | asr | int4_awq (b1+b2)
#   qwen3-tts       | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice   | tts | int4
#   qwen3-tts-base  | Qwen/Qwen3-TTS-12Hz-0.6B-Base          | tts | int4 (+ speaker encoder)
#   qwen3.5-4b      | compatibility alias for qwen3.5-4b-base
#   qwen3.5-4b-base | Qwen/Qwen3.5-4B                        | llm | nvfp4
#   qwen3.5-4b-mtp  | same source; base plus explicit MTP integration hook
#   sparktts-bf16  | <SparkTTS-0.5B repo>                | tts | bf16
#   sparktts-w4a16 | <SparkTTS-0.5B repo>                | tts | w4a16
#   moss        | <MOSS-TTS-Nano repo>                   | tts | fp16         (edge-llm worker — add + validate)
#
# OUT OF SCOPE — ONNX/sherpa backends (paraformer, sensevoice, kokoro, matcha,
# diarization). Those use onnxruntime-gpu / sherpa-onnx / model-specific TRT and
# are mostly MORE portable (ONNX runs cross-GPU); they need a separate track.
#
# Usage:
#   EXPORT_ROOT=/work/build/export MODEL_ROOT=/work/build/models \
#   UPSTREAM=/work/build/upstream  bash build-engines-for-device.sh qwen3-asr qwen3-tts qwen3.5-4b
#   # arch auto-detected via detect-target.sh; override with TARGET_SM / precision via PRECISION_<model>.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UPSTREAM="${UPSTREAM:?set UPSTREAM=<patched TensorRT-Edge-LLM tree>}"
EXPORT_ROOT="${EXPORT_ROOT:-${UPSTREAM}/../export}"
MODEL_ROOT="${MODEL_ROOT:-${UPSTREAM}/../models}"
BUILD="${UPSTREAM}/build"                 # voice-worker build (ENABLE_CUTE_DSL=OFF)
BUILD_GDN="${UPSTREAM}/build-gdn"         # GDN build (ENABLE_CUTE_DSL=fmha;gdn) — for llm
PLUGIN="${BUILD}/libNvInfer_edgellm_plugin.so"
HF="${HF_ENDPOINT:-https://huggingface.co}"

# Quantization/export imports torch before any engine is built. Reject a
# CUDA-13 torch environment on the JP6.2 CUDA-12.6 driver instead of failing
# later with misleading kernel/runtime errors.
python3 "${HERE}/probe-cuda-compat.py"

# Builders are release inputs, not optional conveniences.
for required_builder in \
  "${BUILD}/examples/llm/llm_build" \
  "${BUILD}/examples/multimodal/audio_build"; do
  [ -x "${required_builder}" ] || {
    echo "ERROR: required builder is missing or not executable: ${required_builder}" >&2
    exit 4
  }
done
[ -e "${PLUGIN}" ] || {
  echo "ERROR: required TensorRT Edge-LLM plugin is missing: ${PLUGIN}" >&2
  exit 4
}

run_builder() {
  # TensorRT's ONNX parser discovers Edge-LLM custom-op creators through the
  # process-wide plugin registry. LD_LIBRARY_PATH alone does not load the
  # library, so a clean process otherwise fails with "Plugin not found".
  LD_PRELOAD="${PLUGIN}${LD_PRELOAD:+:${LD_PRELOAD}}" "$@"
}

# --- single source of truth for arch/platform ---
eval "$(bash "${HERE}/detect-target.sh")"
echo "==> target: SM=${TARGET_SM} platform=${TARGET_PLATFORM} arch=${CMAKE_CUDA_ARCH}"

export LD_LIBRARY_PATH="${BUILD}:${BUILD_GDN}:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export EDGELLM_PLUGIN_PATH="${PLUGIN}"

# trtexec (system tool) location differs by platform: Jetson ships it under
# /usr/src/tensorrt/bin, the sbsa/x86 TensorRT tarball under /opt/tensorrt/bin.
# Used by the ONNX->engine backends (moss). Override with TRTEXEC=<path>.
_find_trtexec() {
  for p in "${TRTEXEC:-}" /usr/src/tensorrt/bin/trtexec /opt/tensorrt/bin/trtexec; do
    [ -n "${p}" ] && [ -x "${p}" ] && { printf '%s' "${p}"; return; }
  done
  command -v trtexec 2>/dev/null || { echo "trtexec not found (set TRTEXEC=)" >&2; exit 3; }
}

_dl() { # repo -> local dir (idempotent)
  local repo="$1" dst="${MODEL_ROOT}/$(basename "$1")"
  [ -f "${dst}/config.json" ] || HF_ENDPOINT="${HF}" hf download "${repo}" --local-dir "${dst}" >/dev/null
  printf '%s' "${dst}"
}

_meta() { # engine-file: write host-matched .meta.json sidecar (resolver cache hit)
  python3 - "$1" <<'PY'
import sys; sys.path.insert(0, __import__("os").environ.get("SEEED","/work/seeed"))
from pathlib import Path
from server.core.engine_resolver import detect_host_signature, _write_meta
_write_meta(Path(sys.argv[1]), detect_host_signature(), "local_build", None)
PY
}

build_asr() { # $1 model_id  $2 hf_repo  $3 precision(int4_awq|fp16)
  local m="$1" repo="$2" prec="${3:-int4_awq}" src out max_input max_kv suffix
  src="$(_dl "${repo}")"; out="${EXPORT_ROOT}/${m}"
  max_input="${ASR_MAX_INPUT_LEN:-1024}"
  max_kv="${ASR_MAX_KV_CACHE_CAPACITY:-1536}"
  if { [ "${max_input}" != "1024" ] || [ "${max_kv}" != "1536" ]; } \
      && [ "${ASR_LONG_CONTEXT:-0}" != "1" ]; then
    echo "ERROR: non-production ASR context requires ASR_LONG_CONTEXT=1" >&2
    exit 8
  fi
  suffix=""
  if [ "${ASR_LONG_CONTEXT:-0}" = "1" ] \
      && { [ "${max_input}" != "1024" ] || [ "${max_kv}" != "1536" ]; }; then
    suffix="-longctx-${max_input}-${max_kv}"
  fi
  echo "==> [asr:${m}] quantize ${prec} -> export(--fp8-embedding) -> thinker + audio encoder"
  ( cd "${UPSTREAM}"
    tensorrt-edgellm-quantize llm --model_dir "${src}" --output_dir "${out}/_q" --quantization "${prec}"
    tensorrt-edgellm-export "${out}/_q" "${out}/onnx" --fp8-embedding )
  # Keep distinct b1/b2 artifacts: b1 is the low-footprint rollback; b2 is
  # required for two independent SessionLaneManager lanes. Long-context
  # opt-ins use suffixed directories and cannot overwrite the production pair.
  run_builder "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/thinker-b1${suffix}" --maxBatchSize 1 \
      --maxInputLen "${max_input}" --maxKVCacheCapacity "${max_kv}"
  run_builder "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/thinker-b2${suffix}" --maxBatchSize 2 \
      --maxInputLen "${max_input}" --maxKVCacheCapacity "${max_kv}"
  run_builder "${BUILD}/examples/multimodal/audio_build" --onnxDir "${out}/onnx/audio" --engineDir "${out}/audio_encoder"
  _meta "${out}/thinker-b1${suffix}/llm.engine"
  _meta "${out}/thinker-b2${suffix}/llm.engine"
  _meta "${out}/audio_encoder/audio/audio_encoder.engine"
}

build_tts() { # $1 model_id  $2 hf_repo  $3 precision(int4|fp16)
  local m="$1" repo="$2" prec="${3:-int4}" src out
  local tts_batch tts_max_input tts_max_kv tts_engine_suffix
  src="$(_dl "${repo}")"; out="${EXPORT_ROOT}/${m}"
  # Product defaults are the v0.8 Base limits already validated on Jetson.
  # Concurrency is a separate artifact choice: the default build is the
  # low-footprint N=1 engine; opt-in N=2 builds use distinct directories so
  # they cannot overwrite the production rollback.
  tts_batch="${TTS_MAX_BATCH_SIZE:-1}"
  tts_max_input="${TTS_MAX_INPUT_LEN:-1024}"
  tts_max_kv="${TTS_MAX_KV_CACHE_CAPACITY:-1536}"
  case "${tts_batch}" in
    1) tts_engine_suffix="" ;;
    2) tts_engine_suffix="-b2" ;;
    *)
      echo "ERROR: TTS_MAX_BATCH_SIZE must be 1 or 2 (got ${tts_batch})" >&2
      exit 13
      ;;
  esac
  if [ "${tts_max_input}" != "1024" ] || [ "${tts_max_kv}" != "1536" ]; then
    echo "ERROR: TTS context is frozen at input=1024/KV=1536; rebuild only after updating the validated product contract" >&2
    exit 13
  fi
  echo "==> [tts:${m}] talker(${prec}) + code_predictor + code2wav (fp16)"
  ( cd "${UPSTREAM}"
    if [ "${m}" = "qwen3-tts-base" ]; then
      # Official v0.9.1 rejects Qwen3-TTS Base checkpoints and supports only
      # CustomVoice. Keep Base as a reviewed local integration until upstream
      # implements it; never credit a historical ONNX tree as a v0.9.1 export.
      if [ -z "${EDGELLM_TTS_BASE_DRIVER:-}" ] \
          || [ ! -x "${EDGELLM_TTS_BASE_DRIVER:-}" ]; then
        echo "ERROR: Qwen3-TTS Base requires executable EDGELLM_TTS_BASE_DRIVER" >&2
        exit 12
      fi
      TTS_BASE_UPSTREAM="${UPSTREAM}" \
      TTS_BASE_MODEL="${src}" \
      TTS_BASE_OUTPUT="${out}/onnx" \
      TTS_BASE_PRECISION="${prec}" \
        "${EDGELLM_TTS_BASE_DRIVER}"
      for required in \
        llm/model.onnx llm/config.json \
        code_predictor/model.onnx code_predictor/config.json \
        code2wav/model.onnx code2wav/config.json \
        DRIVER_REVISION PROVENANCE.md SHA256SUMS; do
        [ -s "${out}/onnx/${required}" ] || {
          echo "ERROR: Base driver did not produce ${out}/onnx/${required}" >&2
          exit 12
        }
      done
    elif [ "${prec}" = "int4" ]; then
      # v0.9.1 does not ship the product INT4 talker driver. Require a reviewed,
      # executable integration and provenance instead of referencing a
      # repository-local driver directory that upstream does not provide.
      if [ -z "${EDGELLM_TTS_INT4_DRIVER:-}" ] \
          || [ ! -x "${EDGELLM_TTS_INT4_DRIVER:-}" ]; then
        echo "ERROR: CustomVoice INT4 requires executable EDGELLM_TTS_INT4_DRIVER" >&2
        exit 9
      fi
      TTS_INT4_UPSTREAM="${UPSTREAM}" \
      TTS_INT4_MODEL="${src}" \
      TTS_INT4_OUTPUT="${out}/onnx/llm" \
      TTS_INT4_PRECISION="int4" \
        "${EDGELLM_TTS_INT4_DRIVER}"
      for required in model.onnx config.json DRIVER_REVISION PROVENANCE.md; do
        [ -s "${out}/onnx/llm/${required}" ] || {
          echo "ERROR: INT4 driver did not produce ${out}/onnx/llm/${required}" >&2
          exit 9
        }
      done
      tensorrt-edgellm-export "${src}" "${out}/onnx" --components code_predictor,code2wav
    else
      tensorrt-edgellm-export "${src}" "${out}/onnx" --components talker,code_predictor,code2wav
    fi )
  run_builder "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/talker${tts_engine_suffix}" \
      --maxBatchSize "${tts_batch}" \
      --maxInputLen "${tts_max_input}" \
      --maxKVCacheCapacity "${tts_max_kv}"
  if [ "${prec}" = "int4" ]; then
    cp -p "${out}/onnx/llm/DRIVER_REVISION" "${out}/talker${tts_engine_suffix}/DRIVER_REVISION"
    cp -p "${out}/onnx/llm/PROVENANCE.md" "${out}/talker${tts_engine_suffix}/PROVENANCE.md"
  elif [ "${m}" = "qwen3-tts-base" ]; then
    cp -p "${out}/onnx/DRIVER_REVISION" "${out}/talker${tts_engine_suffix}/DRIVER_REVISION"
    cp -p "${out}/onnx/PROVENANCE.md" "${out}/talker${tts_engine_suffix}/PROVENANCE.md"
  fi
  run_builder "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/code_predictor" \
      --engineDir "${out}/code_predictor${tts_engine_suffix}" \
      --maxBatchSize "${tts_batch}" \
      --maxInputLen "${tts_max_input}" \
      --maxKVCacheCapacity "${tts_max_kv}"
  # Production default: maxCodeLen=512 (~41 s at 12.5 Hz). The upstream
  # audio_build default of 2000 reserves about 2.95 GiB of activation memory
  # and OOMs a 16GB Orin NX when GDN is resident. Operators can still request
  # the full-range profile explicitly through the environment.
  run_builder "${BUILD}/examples/multimodal/audio_build" \
      --onnxDir "${out}/onnx/code2wav" \
      --engineDir "${out}/code2wav" \
      --minCodeLen "${QWEN3_TTS_CODE2WAV_MIN_CODE_LEN:-1}" \
      --optCodeLen "${QWEN3_TTS_CODE2WAV_OPT_CODE_LEN:-128}" \
      --maxCodeLen "${QWEN3_TTS_CODE2WAV_MAX_CODE_LEN:-512}"
  if [ "${m}" = "qwen3-tts-base" ]; then
    local trtexec spk_engine_dir spk_onnx_dir spk_revision spk_source
    local spk_expected_sha spk_actual_sha
    # v0.9.1 has no speaker-encoder component and does not install the
    # historical `tensorrt-edgellm-export-audio` command. Treat the encoder as
    # an explicit, version-independent ONNX input rather than pretending it
    # was exported by Edge-LLM. Its source revision is mandatory provenance.
    spk_onnx_dir="${QWEN3_TTS_BASE_SPEAKER_ENCODER_ONNX_DIR:-}"
    spk_revision="${QWEN3_TTS_BASE_SPEAKER_ENCODER_REVISION:-}"
    spk_expected_sha="${QWEN3_TTS_BASE_SPEAKER_ENCODER_SHA256:-}"
    if [ -s "${spk_onnx_dir}/model.onnx" ]; then
      spk_source="${spk_onnx_dir}/model.onnx"
    elif [ -s "${spk_onnx_dir}/speaker_encoder.onnx" ]; then
      spk_source="${spk_onnx_dir}/speaker_encoder.onnx"
    else
      spk_source=""
    fi
    if [ -z "${spk_source}" ] || [ -z "${spk_revision}" ] \
        || [ -z "${spk_expected_sha}" ]; then
      echo "ERROR: qwen3-tts-base requires QWEN3_TTS_BASE_SPEAKER_ENCODER_ONNX_DIR" >&2
      echo "       with model.onnx or speaker_encoder.onnx, plus non-empty" >&2
      echo "       QWEN3_TTS_BASE_SPEAKER_ENCODER_REVISION and" >&2
      echo "       QWEN3_TTS_BASE_SPEAKER_ENCODER_SHA256." >&2
      exit 11
    fi
    spk_actual_sha="$(sha256sum "${spk_source}" | awk '{print $1}')"
    if [ "${spk_actual_sha}" != "${spk_expected_sha}" ]; then
      echo "ERROR: speaker encoder SHA-256 mismatch" >&2
      echo "       expected ${spk_expected_sha}" >&2
      echo "       actual   ${spk_actual_sha}" >&2
      exit 11
    fi
    echo "==> [tts:${m}] import + build explicit speaker_encoder for voice cloning"
    mkdir -p "${out}/onnx/speaker_encoder"
    cp -p "${spk_source}" "${out}/onnx/speaker_encoder/model.source.onnx"
    python3 "${HERE}/fix-qwen3-tts-speaker-encoder-trt10.py" \
      "${out}/onnx/speaker_encoder/model.source.onnx" \
      "${out}/onnx/speaker_encoder/model.onnx"
    printf '%s  model.source.onnx\n' "${spk_actual_sha}" \
      > "${out}/onnx/speaker_encoder/SHA256SUMS.source"
    sha256sum "${out}/onnx/speaker_encoder/model.onnx" \
      "${HERE}/fix-qwen3-tts-speaker-encoder-trt10.py" \
      > "${out}/onnx/speaker_encoder/SHA256SUMS.trt10"
    printf '%s\n' \
      '{"model_type":"qwen3_tts_speaker_encoder","input":{"name":"mel","dtype":"float32","shape":[1,"time",128]},"output":{"name":"speaker_embedding","dtype":"float32","shape":[1024]},"sample_rate":24000}' \
      > "${out}/onnx/speaker_encoder/config.json"
    printf 'source_revision: %s\nsource_file: %s\nsource_sha256: %s\n' \
      "${spk_revision}" "${spk_source}" "${spk_actual_sha}" \
      > "${out}/onnx/speaker_encoder/PROVENANCE.md"
    # audio_build supports only audio_encoder/code2wav in v0.9.1. The speaker
    # encoder is a standalone ONNX graph, so use TensorRT's documented builder.
    trtexec="$(_find_trtexec)"
    spk_engine_dir="${EXPORT_ROOT}/tts_base_spk_encoder/speaker_encoder"
    mkdir -p "${spk_engine_dir}"
    "${trtexec}" --onnx="${out}/onnx/speaker_encoder/model.onnx" \
      --fp16 \
      --minShapes=mel:1x10x128 \
      --optShapes=mel:1x555x128 \
      --maxShapes=mel:1x2000x128 \
      --saveEngine="${spk_engine_dir}/spk_encoder.engine"
    cp -p "${out}/onnx/speaker_encoder/PROVENANCE.md" \
      "${out}/onnx/speaker_encoder/SHA256SUMS.source" \
      "${out}/onnx/speaker_encoder/SHA256SUMS.trt10" \
      "${spk_engine_dir}/"
    _meta "${spk_engine_dir}/spk_encoder.engine"
  fi
  # tokenizer dir the C++ worker loadFromHF() needs: tokenizer.json + config + chat template
  cp -n "${out}/talker/tokenizer.json" "${out}/talker/processed_chat_template.json" "${src}/" 2>/dev/null || true
  for e in "${out}/talker/llm.engine" "${out}/code_predictor/llm.engine" "${out}/code2wav/code2wav/code2wav.engine"; do _meta "${e}"; done
}

build_llm_base() { # $1 model_id  $2 hf_repo  $3 precision(nvfp4|fp16)
  local m="$1" repo="$2" prec="${3:-nvfp4}" src out
  src="$(_dl "${repo}")"; out="${EXPORT_ROOT}/gdn-base"
  echo "==> [llm:${m}] quantize ${prec}+fp8-kv -> export(--skip-visual) -> GDN engine"
  ( cd "${UPSTREAM}"
    tensorrt-edgellm-quantize llm --model_dir "${src}" --output_dir "${out}/_q" \
      --quantization "${prec}" --kv_cache_quantization fp8 --dataset abisee/cnn_dailymail
    tensorrt-edgellm-export "${out}/_q" "${out}/onnx" --skip-visual --skip-audio )
  # GDN needs the CuTe-enabled llm_build (build-gdn); tag auto-inferred from arch
  EDGELLM_PLUGIN_PATH="${BUILD_GDN}/libNvInfer_edgellm_plugin.so" \
  "${BUILD_GDN}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}" --maxBatchSize 1 --maxInputLen 4096 --maxKVCacheCapacity 8192
  _meta "${out}/llm.engine"
}

build_llm_mtp() { # $1 model_id  $2 hf_repo  $3 precision
  local m="$1" repo="$2" prec="$3" src export_dir spec_dir
  src="$(_dl "${repo}")"
  export_dir="${EXPORT_ROOT}/${m}/onnx"
  spec_dir="${MTP_SPEC_DIR:-${EXPORT_ROOT}/gdn-mtp}"
  # v0.9.1 exposes MTP model data, but this repository has not yet established
  # a stable official export/build CLI contract. Do not guess one. A reviewed
  # integration script owns the MTP export/build as one coherent operation and
  # must create spec_base.engine + spec_draft.engine + config/provenance in the
  # same spec directory. Vanilla GDN is a separate qwen3.5-4b-base build.
  if [ -z "${EDGELLM_MTP_BUILD_SCRIPT:-}" ] || [ ! -x "${EDGELLM_MTP_BUILD_SCRIPT:-}" ]; then
    echo "ERROR: qwen3.5-4b-mtp requires executable EDGELLM_MTP_BUILD_SCRIPT." >&2
    echo "       The hook receives MTP_UPSTREAM, MTP_MODEL, MTP_EXPORT," >&2
    echo "       MTP_SPEC_DIR and MTP_PLUGIN." >&2
    exit 6
  fi
  MTP_UPSTREAM="${UPSTREAM}" \
  MTP_MODEL="${src}" \
  MTP_EXPORT="${export_dir}" \
  MTP_SPEC_DIR="${spec_dir}" \
  MTP_PLUGIN="${BUILD_GDN}/libNvInfer_edgellm_plugin.so" \
    "${EDGELLM_MTP_BUILD_SCRIPT}"
  if [ ! -s "${spec_dir}/spec_base.engine" ] \
      || [ ! -s "${spec_dir}/spec_draft.engine" ] \
      || [ ! -s "${spec_dir}/config.json" ] \
      || [ ! -s "${spec_dir}/PROVENANCE.md" ]; then
    echo "ERROR: MTP hook must produce spec_base.engine, spec_draft.engine," >&2
    echo "       config.json and PROVENANCE.md under ${spec_dir}" >&2
    exit 6
  fi
  _meta "${spec_dir}/spec_base.engine"
  _meta "${spec_dir}/spec_draft.engine"
}

build_moss() { # $1 model_id  $2 (unused) $3 precision(mix1)
  # MOSS-TTS-Nano does NOT use the edge-llm quantize->export pipeline. It ships a
  # pre-exported ONNX bundle that trtexec turns into engines directly (arch is
  # auto-handled by trtexec — naturally portable). Provide the bundle via
  #   MOSS_ONNX_BUNDLE=<dir>  containing:
  #     tts/   moss_tts_{prefill,decode_step,local_decoder,local_cached_step,
  #            local_fixed_sampled_frame}.onnx + *.data + tts_browser_onnx_meta.json
  #            + browser_poc_manifest.json + tokenizer.model
  #     codec/ moss_audio_tokenizer_decode_step.onnx + codec_browser_onnx_meta.json (+ .data)
  # (re-export from the HF model with models/moss-tts-nano/vendor/export_hf_to_tts_onnx.py
  #  if you only have weights). Precision recipe = mix1 (fp32 globals+codec, bf16
  #  local) — a naive all-fp16 build compiles clean but is SILENT. See the build
  #  script header. Build the fp32 globals from the CLEAN (non-paged) ONNX.
  local m="$1" out="${EXPORT_ROOT}/$1" trtexec
  local bundle="${MOSS_ONNX_BUNDLE:?set MOSS_ONNX_BUNDLE=<dir with tts/ + codec/ ONNX> for moss}"
  trtexec="$(_find_trtexec)"
  echo "==> [moss:${m}] trtexec mix1 (fp32 globals+codec / bf16 local) via ${trtexec}"
  ONNX_DIR="${bundle}/tts" CODEC_ONNX_DIR="${bundle}/codec" \
    OUT_DIR="${out}" TRTEXEC="${trtexec}" \
    bash "${HERE}/../models/moss-tts-nano/build_moss_tts_engines.sh"
  _meta "${out}/engines/moss_tts_prefill.plan"
}

build_sparktts() { # $1 model_id  $2 (unused)  $3 mode(bf16|w4a16)
  # SparkTTS-0.5B = Qwen2.5-0.5B LLM + BiCodec vocoder + speaker decoder. The
  # vocoder/speaker ONNX are exported by the (arch-independent) committed scripts
  # from the SparkTTS checkpoint; provide it via
  #   SPARKTTS_MODEL_DIR=<dir with BiCodec/>  SPARKTTS_REPO=<Spark-TTS repo root>
  # then trtexec (bicodec fp16 dyn-T, speaker fp32). BF16 and W4A16 are separate
  # reviewed driver routes with independent artifacts and a 32-global-token gate.
  local m="$1" mode="$3" out="${EXPORT_ROOT}/$1" trtexec
  local model="${SPARKTTS_MODEL_DIR:?set SPARKTTS_MODEL_DIR=<dir with BiCodec/> for sparktts}"
  local repo="${SPARKTTS_REPO:?set SPARKTTS_REPO=<Spark-TTS repo root> for sparktts}"
  case "${mode}" in
    bf16|w4a16) ;;
    *) echo "ERROR: SparkTTS mode must be bf16 or w4a16, got ${mode}" >&2; exit 10 ;;
  esac
  trtexec="$(_find_trtexec)"
  local eng="${out}/sparktts-engines" scripts="${HERE}/../scripts"
  mkdir -p "${eng}"
  echo "==> [sparktts:${m}] export BiCodec + speaker decoder ONNX (arch-independent)"
  SPARKTTS_MODEL_DIR="${model}" SPARKTTS_REPO="${repo}" SPARKTTS_OUT_DIR="${eng}" \
    python3 "${scripts}/export_sparktts_bicodec_decoder.py"
  SPARKTTS_MODEL_DIR="${model}" SPARKTTS_REPO="${repo}" SPARKTTS_OUT_DIR="${eng}" \
    python3 "${scripts}/export_sparktts_speaker_decoder.py"
  echo "==> [sparktts:${m}] trtexec bicodec fp16 (dyn-T) + speaker fp32"
  "${trtexec}" --onnx="${eng}/bicodec_decoder_dynT.onnx" --fp16 \
    --minShapes=semantic_tokens:1x50,d_vector:1x1024 \
    --optShapes=semantic_tokens:1x200,d_vector:1x1024 \
    --maxShapes=semantic_tokens:1x600,d_vector:1x1024 \
    --saveEngine="${eng}/bicodec_decoder_dynT.fp16.engine" 2>&1 | tail -3
  "${trtexec}" --onnx="${eng}/sparktts_speaker_decoder.onnx" \
    --saveEngine="${eng}/sparktts_speaker_decoder.fp32.engine" 2>&1 | tail -3
  if [ -z "${EDGELLM_SPARK_LLM_DRIVER:-}" ] \
      || [ ! -x "${EDGELLM_SPARK_LLM_DRIVER:-}" ]; then
    echo "ERROR: SparkTTS ${mode} requires executable EDGELLM_SPARK_LLM_DRIVER" >&2
    exit 10
  fi
  echo "==> [sparktts:${m}] reviewed ${mode} LLM driver + token gate"
  SPARK_LLM_UPSTREAM="${UPSTREAM}" \
  SPARK_LLM_MODEL="${model}/LLM" \
  SPARK_LLM_OUTPUT="${out}" \
  SPARK_LLM_MODE="${mode}" \
  SPARK_LLM_PLUGIN="${BUILD}/libNvInfer_edgellm_plugin.so" \
    "${EDGELLM_SPARK_LLM_DRIVER}"
  for required in llm.engine DRIVER_REVISION PROVENANCE.md token-gate.json; do
    [ -s "${out}/${required}" ] || {
      echo "ERROR: Spark ${mode} driver did not produce ${out}/${required}" >&2
      exit 10
    }
  done
  python3 "${HERE}/validate-spark-token-gate.py" \
    "${out}/token-gate.json" "${mode}"
  _meta "${eng}/bicodec_decoder_dynT.fp16.engine"
  _meta "${eng}/sparktts_speaker_decoder.fp32.engine"
  _meta "${out}/llm.engine"
}

# --- manifest: model_id | hf_repo | kind | default_precision ---
declare -A REPO KIND PREC
REPO[qwen3-asr]="Qwen/Qwen3-ASR-0.6B";                  KIND[qwen3-asr]=asr;  PREC[qwen3-asr]=int4_awq
REPO[qwen3-tts]="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"; KIND[qwen3-tts]=tts;  PREC[qwen3-tts]=int4
REPO[qwen3-tts-base]="Qwen/Qwen3-TTS-12Hz-0.6B-Base";   KIND[qwen3-tts-base]=tts; PREC[qwen3-tts-base]=fp16
REPO[qwen3.5-4b]="Qwen/Qwen3.5-4B";                     KIND[qwen3.5-4b]=llm_base; PREC[qwen3.5-4b]=nvfp4
REPO[qwen3.5-4b-base]="Qwen/Qwen3.5-4B";                KIND[qwen3.5-4b-base]=llm_base; PREC[qwen3.5-4b-base]=nvfp4
REPO[qwen3.5-4b-mtp]="Qwen/Qwen3.5-4B";                 KIND[qwen3.5-4b-mtp]=llm_mtp; PREC[qwen3.5-4b-mtp]=nvfp4
# moss/sparktts are not plain HF-repo pipelines — sources come from env
# (MOSS_ONNX_BUNDLE / SPARKTTS_MODEL_DIR+SPARKTTS_REPO); REPO holds a hint only.
REPO[moss]="env:MOSS_ONNX_BUNDLE";                      KIND[moss]=moss;     PREC[moss]=mix1
REPO[sparktts-bf16]="env:SPARKTTS_MODEL_DIR";           KIND[sparktts-bf16]=sparktts; PREC[sparktts-bf16]=bf16
REPO[sparktts-w4a16]="env:SPARKTTS_MODEL_DIR";          KIND[sparktts-w4a16]=sparktts; PREC[sparktts-w4a16]=w4a16

for m in "$@"; do
  [ -n "${REPO[$m]:-}" ] || { echo "unknown model '${m}' (add a manifest line)"; exit 2; }
  # per-model precision override: PRECISION_qwen3_5_4b=fp16 bash ...
  ov="PRECISION_${m//[.-]/_}"; prec="${!ov:-${PREC[$m]}}"
  case "${KIND[$m]}" in
    asr)      build_asr "$m" "${REPO[$m]}" "${prec}" ;;
    tts)      build_tts "$m" "${REPO[$m]}" "${prec}" ;;
    llm_base) build_llm_base "$m" "${REPO[$m]}" "${prec}" ;;
    llm_mtp)  build_llm_mtp "$m" "${REPO[$m]}" "${prec}" ;;
    moss)     build_moss "$m" "${REPO[$m]}" "${prec}" ;;
    sparktts) build_sparktts "$m" "${REPO[$m]}" "${prec}" ;;
  esac
done
echo "==> done. engines under ${EXPORT_ROOT}/<model>/  (host-matched .meta.json written)"
