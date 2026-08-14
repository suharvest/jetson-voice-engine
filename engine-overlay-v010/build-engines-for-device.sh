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
#   qwen3-tts-base  | Qwen/Qwen3-TTS-12Hz-0.6B-Base          | tts | fp16 (+ native clone encoders)
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
EXPECTED_UPSTREAM_PIN="71dd1bae032e70771265917ec74d3ff4cad07a10"
ACTUAL_UPSTREAM_PIN="$(git -C "${UPSTREAM}" rev-parse HEAD 2>/dev/null || true)"
if [ -z "${ACTUAL_UPSTREAM_PIN}" ]; then
  # Release builders may consume a verified source archive without .git.
  # The archive producer must bind the exact base in trusted metadata and pass
  # it explicitly; arbitrary bypasses and floating version strings are rejected.
  ACTUAL_UPSTREAM_PIN="${EDGELLM_UPSTREAM_BASE_REVISION:-}"
fi
[ "${ACTUAL_UPSTREAM_PIN}" = "${EXPECTED_UPSTREAM_PIN}" ] || {
  echo "ERROR: expected TensorRT-Edge-LLM v0.10.0 base ${EXPECTED_UPSTREAM_PIN}" >&2
  echo "       got ${ACTUAL_UPSTREAM_PIN:-<not-a-git-checkout>} from ${UPSTREAM}" >&2
  exit 5
}
EXPORT_ROOT="${EXPORT_ROOT:-${UPSTREAM}/../export}"
MODEL_ROOT="${MODEL_ROOT:-${UPSTREAM}/../models}"
BUILD="${UPSTREAM}/build"                 # voice-worker build (ENABLE_CUTE_DSL=fmha)
BUILD_GDN="${UPSTREAM}/build-gdn"         # GDN build (ENABLE_CUTE_DSL=ALL) — for GDN/spec decode
PLUGIN="${BUILD}/libNvInfer_edgellm_plugin.so"
HF="${HF_ENDPOINT:-}"
if [ "${HF}" != "https://hf-mirror.com" ]; then
  echo "ERROR: model downloads require HF_ENDPOINT=https://hf-mirror.com" >&2
  echo "       provision the device with: fleet bootstrap <device> --profile edge-mirror" >&2
  exit 4
fi

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

_dl() { # repo revision -> local dir (idempotent, immutable)
  local repo="$1" revision="$2" dst="${MODEL_ROOT}/$(basename "$1")"
  [ -n "${revision}" ] || {
    echo "ERROR: immutable HF revision is required for ${repo}" >&2
    exit 14
  }
  if [ ! -f "${dst}/config.json" ]; then
    HF_ENDPOINT="${HF}" hf download "${repo}" --revision "${revision}" --local-dir "${dst}" >/dev/null
    printf '%s\n' "${revision}" > "${dst}/SOURCE_REVISION"
  elif [ ! -s "${dst}/SOURCE_REVISION" ] \
      || [ "$(cat "${dst}/SOURCE_REVISION")" != "${revision}" ]; then
    echo "ERROR: existing model directory has missing/mismatched SOURCE_REVISION: ${dst}" >&2
    echo "       expected ${revision}; use a clean MODEL_ROOT for v0.10" >&2
    exit 14
  fi
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

_provenance() { # artifact-root repo revision precision
  local root="$1" repo="$2" revision="$3" precision="$4"
  mkdir -p "${root}"
  {
    printf 'upstream_version: v0.10.0\n'
    printf 'upstream_revision: %s\n' "${EXPECTED_UPSTREAM_PIN}"
    printf 'model_repo: %s\n' "${repo}"
    printf 'model_revision: %s\n' "${revision}"
    printf 'precision: %s\n' "${precision}"
    printf 'target_sm: %s\n' "${TARGET_SM}"
    printf 'target_platform: %s\n' "${TARGET_PLATFORM}"
  } > "${root}/PROVENANCE.md"
  sha256sum "${root}/PROVENANCE.md" > "${root}/SHA256SUMS.provenance"
}

build_asr() { # $1 model_id  $2 hf_repo  $3 precision  $4 immutable revision
  local m="$1" repo="$2" prec="${3:-int4_awq}" src out max_input max_kv suffix export_src
  src="$(_dl "${repo}" "$4")"; out="${EXPORT_ROOT}/${m}"
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
  export_src="${src}"
  if [ "${prec}" != "fp16" ] && [ "${prec}" != "bf16" ]; then
    export_src="${out}/_q"
    echo "==> [asr:${m}] quantize ${prec}"
    ( cd "${UPSTREAM}"
      tensorrt-edgellm-quantize llm \
        --model_dir "${src}" \
        --output_dir "${export_src}" \
        --quantization "${prec}" \
        --text_dataset cnn_dailymail )
  fi
  echo "==> [asr:${m}] v0.10 export -> thinker + audio encoder"
  ( cd "${UPSTREAM}"
    if [ "${ASR_FP8_EMBEDDING:-0}" = "1" ]; then
      tensorrt-edgellm-export "${export_src}" "${out}/onnx" --fp8-embedding
    else
      tensorrt-edgellm-export "${export_src}" "${out}/onnx"
    fi )
  for required in \
    llm/model.onnx llm/config.json llm/embedding.safetensors \
    llm/tokenizer.json llm/processed_chat_template.json \
    audio/model.onnx audio/config.json; do
    [ -s "${out}/onnx/${required}" ] || {
      echo "ERROR: v0.10 ASR exporter did not produce ${out}/onnx/${required}" >&2
      exit 8
    }
  done
  # Keep distinct b1/b2 artifacts: b1 is the low-footprint rollback; b2 is
  # required for two independent SessionLaneManager lanes. Long-context
  # opt-ins use suffixed directories and cannot overwrite the production pair.
  run_builder "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/thinker-b1${suffix}" --maxBatchSize 1 \
      --maxInputLen "${max_input}" --maxKVCacheCapacity "${max_kv}"
  run_builder "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/thinker-b2${suffix}" --maxBatchSize 2 \
      --maxInputLen "${max_input}" --maxKVCacheCapacity "${max_kv}"
  run_builder "${BUILD}/examples/multimodal/audio_build" \
      --onnxDir "${out}/onnx/audio" \
      --engineDir "${out}/audio_encoder" \
      --minTimeSteps "${ASR_AUDIO_MIN_TIME_STEPS:-100}" \
      --maxTimeSteps "${ASR_AUDIO_MAX_TIME_STEPS:-3000}"
  _meta "${out}/thinker-b1${suffix}/llm.engine"
  _meta "${out}/thinker-b2${suffix}/llm.engine"
  _meta "${out}/audio_encoder/audio/audio_encoder.engine"
  _provenance "${out}" "${repo}" "$4" "${prec}"
}

build_tts() { # $1 model_id  $2 hf_repo  $3 precision(int4|fp16)  $4 immutable revision
  local m="$1" repo="$2" prec="${3:-int4}" src out
  local tts_batch tts_max_input tts_max_kv tts_engine_suffix
  src="$(_dl "${repo}" "$4")"; out="${EXPORT_ROOT}/${m}"
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
  echo "==> [tts:${m}] v0.10 talker(${prec}) + code_predictor + code2wav (fp16)"
  ( cd "${UPSTREAM}"
    if [ "${prec}" = "int4" ]; then
      # v0.10 still supports Qwen3-TTS Talker in FP16 only. Keep the reviewed
      # product INT4 Talker route isolated and provenance-bearing; all other
      # components, including Base clone encoders, come from native v0.10.
      if [ -z "${EDGELLM_TTS_INT4_DRIVER:-}" ] \
          || [ ! -x "${EDGELLM_TTS_INT4_DRIVER:-}" ]; then
        echo "ERROR: Qwen3-TTS INT4 requires executable EDGELLM_TTS_INT4_DRIVER" >&2
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
      tensorrt-edgellm-export "${src}" "${out}/onnx" \
        --components talker,code_predictor,code2wav
    fi )
  for required in \
    llm/model.onnx llm/config.json llm/embedding.safetensors \
    llm/text_embedding.safetensors llm/text_projection.safetensors \
    llm/tokenizer.json llm/processed_chat_template.json \
    code_predictor/model.onnx code_predictor/config.json \
    code_predictor/codec_embeddings.safetensors \
    code_predictor/lm_heads.safetensors \
    code2wav/model.onnx code2wav/config.json; do
    [ -s "${out}/onnx/${required}" ] || {
      echo "ERROR: v0.10 exporter did not produce ${out}/onnx/${required}" >&2
      exit 12
    }
  done
  run_builder "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/talker${tts_engine_suffix}" \
      --maxBatchSize "${tts_batch}" \
      --maxInputLen "${tts_max_input}" \
      --maxKVCacheCapacity "${tts_max_kv}"
  if [ "${prec}" = "int4" ]; then
    cp -p "${out}/onnx/llm/DRIVER_REVISION" "${out}/talker${tts_engine_suffix}/DRIVER_REVISION"
    cp -p "${out}/onnx/llm/PROVENANCE.md" "${out}/talker${tts_engine_suffix}/PROVENANCE.md"
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
      --engineDir "${out}" \
      --minCodeLen "${QWEN3_TTS_CODE2WAV_MIN_CODE_LEN:-1}" \
      --optCodeLen "${QWEN3_TTS_CODE2WAV_OPT_CODE_LEN:-128}" \
      --maxCodeLen "${QWEN3_TTS_CODE2WAV_MAX_CODE_LEN:-512}"
  if [ "${m}" = "qwen3-tts-base" ]; then
    local trtexec clone_engine_dir
    for required in speaker_encoder.onnx speech_tokenizer_encoder.onnx; do
      [ -s "${out}/onnx/clone_encoders/${required}" ] || {
        echo "ERROR: v0.10 Base export did not produce clone_encoders/${required}" >&2
        exit 11
      }
    done
    trtexec="$(_find_trtexec)"
    clone_engine_dir="${out}/clone_encoders"
    mkdir -p "${clone_engine_dir}"
    echo "==> [tts:${m}] build native v0.10 speaker + speech-tokenizer clone encoders"
    "${trtexec}" --onnx="${out}/onnx/clone_encoders/speaker_encoder.onnx" \
      --fp16 \
      --minShapes=wav:1x24000 \
      --optShapes=wav:1x240000 \
      --maxShapes=wav:1x960000 \
      --saveEngine="${clone_engine_dir}/speaker_encoder.engine"
    "${trtexec}" \
      --onnx="${out}/onnx/clone_encoders/speech_tokenizer_encoder.onnx" \
      --fp16 \
      --saveEngine="${clone_engine_dir}/speech_tokenizer_encoder.engine"
    _meta "${clone_engine_dir}/speaker_encoder.engine"
    _meta "${clone_engine_dir}/speech_tokenizer_encoder.engine"
  fi
  for e in \
    "${out}/talker${tts_engine_suffix}/llm.engine" \
    "${out}/code_predictor${tts_engine_suffix}/llm.engine" \
    "${out}/code2wav/code2wav.engine"; do
    _meta "${e}"
  done
  _provenance "${out}" "${repo}" "$4" "${prec}"
}

build_llm_base() { # $1 model_id  $2 hf_repo  $3 precision  $4 immutable revision
  local m="$1" repo="$2" prec="${3:-nvfp4}" src out
  src="$(_dl "${repo}" "$4")"; out="${EXPORT_ROOT}/${m}"
  [ -x "${BUILD_GDN}/examples/llm/llm_build" ] \
      && [ -s "${BUILD_GDN}/libNvInfer_edgellm_plugin.so" ] || {
    echo "ERROR: ${m} requires a v0.10 BUILD_GDN with ENABLE_CUTE_DSL=ALL" >&2
    exit 6
  }
  echo "==> [llm:${m}] quantize ${prec}+fp8-kv -> export(--skip-visual) -> GDN engine"
  ( cd "${UPSTREAM}"
    tensorrt-edgellm-quantize llm --model_dir "${src}" --output_dir "${out}/_q" \
      --quantization "${prec}" --kv_cache_quantization fp8 --text_dataset cnn_dailymail
    tensorrt-edgellm-export "${out}/_q" "${out}/onnx" --skip-visual --skip-audio )
  for required in \
    llm/model.onnx llm/config.json llm/embedding.safetensors \
    llm/tokenizer.json llm/processed_chat_template.json; do
    [ -s "${out}/onnx/${required}" ] || {
      echo "ERROR: v0.10 LLM exporter did not produce ${out}/onnx/${required}" >&2
      exit 6
    }
  done
  # GDN needs the CuTe-enabled llm_build (build-gdn); tag auto-inferred from arch
  EDGELLM_PLUGIN_PATH="${BUILD_GDN}/libNvInfer_edgellm_plugin.so" \
  "${BUILD_GDN}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}" --maxBatchSize 1 --maxInputLen 4096 --maxKVCacheCapacity 8192
  _meta "${out}/llm.engine"
  _provenance "${out}" "${repo}" "$4" "${prec}"
}

build_llm_mtp() { # $1 model_id  $2 hf_repo  $3 precision  $4 immutable revision
  local m="$1" repo="$2" prec="$3" src export_dir spec_dir export_src
  src="$(_dl "${repo}" "$4")"
  export_dir="${EXPORT_ROOT}/${m}/onnx"
  spec_dir="${MTP_SPEC_DIR:-${EXPORT_ROOT}/gdn-mtp}"
  export_src="${src}"
  if [ "${prec}" != "fp16" ] && [ "${prec}" != "bf16" ]; then
    export_src="${EXPORT_ROOT}/${m}/_q"
    echo "==> [llm:${m}] quantize ${prec}+fp8-kv for native v0.10 MTP"
    ( cd "${UPSTREAM}"
      tensorrt-edgellm-quantize llm \
        --model_dir "${src}" \
        --output_dir "${export_src}" \
        --quantization "${prec}" \
        --kv_cache_quantization fp8 \
        --text_dataset cnn_dailymail )
  fi
  echo "==> [llm:${m}] native v0.10 export --mtp"
  ( cd "${UPSTREAM}"
    tensorrt-edgellm-export "${export_src}" "${export_dir}" --mtp )
  for required in llm/model.onnx llm/config.json mtp_draft/model.onnx mtp_draft/config.json; do
    [ -s "${export_dir}/${required}" ] || {
      echo "ERROR: v0.10 MTP export did not produce ${export_dir}/${required}" >&2
      exit 6
    }
  done
  [ -x "${BUILD_GDN}/examples/llm/llm_build" ] \
      && [ -s "${BUILD_GDN}/libNvInfer_edgellm_plugin.so" ] || {
    echo "ERROR: MTP requires a v0.10 BUILD_GDN with ENABLE_CUTE_DSL=ALL" >&2
    exit 6
  }
  mkdir -p "${spec_dir}"
  EDGELLM_PLUGIN_PATH="${BUILD_GDN}/libNvInfer_edgellm_plugin.so" \
    "${BUILD_GDN}/examples/llm/llm_build" \
      --onnxDir "${export_dir}/llm" \
      --engineDir "${spec_dir}" \
      --maxBatchSize 1 \
      --maxInputLen "${MTP_MAX_INPUT_LEN:-2048}" \
      --maxKVCacheCapacity "${MTP_MAX_KV_CACHE_CAPACITY:-4096}" \
      --maxVerifyTreeSize 4 \
      --specBase
  EDGELLM_PLUGIN_PATH="${BUILD_GDN}/libNvInfer_edgellm_plugin.so" \
    "${BUILD_GDN}/examples/llm/llm_build" \
      --onnxDir "${export_dir}/mtp_draft" \
      --engineDir "${spec_dir}" \
      --maxBatchSize 1 \
      --maxInputLen "${MTP_MAX_INPUT_LEN:-2048}" \
      --maxKVCacheCapacity "${MTP_MAX_KV_CACHE_CAPACITY:-4096}" \
      --maxDraftTreeSize 4 \
      --specDraft
  # v0.9.1 used EDGELLM_MTP_BUILD_SCRIPT here. v0.10 owns --mtp export and
  # --specBase/--specDraft builds natively, so the opaque hook is retired.
  if [ ! -s "${spec_dir}/spec_base.engine" ] \
      || [ ! -s "${spec_dir}/spec_draft.engine" ] \
      || [ ! -s "${spec_dir}/base_config.json" ] \
      || [ ! -s "${spec_dir}/draft_config.json" ]; then
    echo "ERROR: native v0.10 MTP build must produce spec_base.engine," >&2
    echo "       spec_draft.engine, base_config.json and draft_config.json under ${spec_dir}" >&2
    exit 6
  fi
  _meta "${spec_dir}/spec_base.engine"
  _meta "${spec_dir}/spec_draft.engine"
  _provenance "${spec_dir}" "${repo}" "$4" "${prec}"
}

build_llm_dflash() { # id base_repo draft_repo precision base_rev draft_rev
  local m="$1" base_repo="$2" draft_repo="$3" prec="$4"
  local base_rev="$5" draft_rev="$6" base_src draft_src root base_q draft_q spec_dir
  base_src="$(_dl "${base_repo}" "${base_rev}")"
  draft_src="$(_dl "${draft_repo}" "${draft_rev}")"
  root="${EXPORT_ROOT}/${m}"
  base_q="${root}/_q-base"
  draft_q="${root}/_q-draft"
  spec_dir="${DFLASH_SPEC_DIR:-${root}/engines}"
  [ -x "${BUILD_GDN}/examples/llm/llm_build" ] \
      && [ -s "${BUILD_GDN}/libNvInfer_edgellm_plugin.so" ] || {
    echo "ERROR: DFlash requires a v0.10 BUILD_GDN with ENABLE_CUTE_DSL=ALL" >&2
    exit 15
  }
  case "${prec}" in
    fp16|bf16)
      base_q="${base_src}"
      draft_q="${draft_src}"
      ;;
    *)
      echo "==> [llm:${m}] quantize DFlash base + draft (${prec})"
      ( cd "${UPSTREAM}"
        tensorrt-edgellm-quantize llm \
          --model_dir "${base_src}" \
          --output_dir "${base_q}" \
          --quantization "${prec}" \
          --kv_cache_quantization fp8 \
          --text_dataset cnn_dailymail
        tensorrt-edgellm-quantize draft \
          --base_model_dir "${base_src}" \
          --draft_model_dir "${draft_src}" \
          --output_dir "${draft_q}" \
          --quantization "${prec}" \
          --lm_head_quantization "${prec}" \
          --text_dataset cnn_dailymail )
      ;;
  esac
  echo "==> [llm:${m}] v0.10 DFlash base + draft export"
  ( cd "${UPSTREAM}"
    tensorrt-edgellm-export \
      "${base_q}" "${root}/base-export" \
      --dflash-base --dflash-draft-dir "${draft_q}"
    tensorrt-edgellm-export \
      "${base_q}" "${root}/draft-export" \
      --dflash-draft --dflash-draft-dir "${draft_q}" )
  for required in \
    base-export/llm/model.onnx base-export/llm/config.json \
    base-export/llm/embedding.safetensors \
    draft-export/dflash_draft/model.onnx \
    draft-export/dflash_draft/config.json; do
    [ -s "${root}/${required}" ] || {
      echo "ERROR: v0.10 DFlash exporter did not produce ${root}/${required}" >&2
      exit 15
    }
  done
  mkdir -p "${spec_dir}"
  EDGELLM_PLUGIN_PATH="${BUILD_GDN}/libNvInfer_edgellm_plugin.so" \
    "${BUILD_GDN}/examples/llm/llm_build" \
      --onnxDir "${root}/base-export/llm" \
      --engineDir "${spec_dir}" \
      --maxBatchSize 1 \
      --maxInputLen "${DFLASH_MAX_INPUT_LEN:-1024}" \
      --maxKVCacheCapacity "${DFLASH_MAX_KV_CACHE_CAPACITY:-2048}" \
      --maxVerifyTreeSize 16 \
      --specBase
  EDGELLM_PLUGIN_PATH="${BUILD_GDN}/libNvInfer_edgellm_plugin.so" \
    "${BUILD_GDN}/examples/llm/llm_build" \
      --onnxDir "${root}/draft-export/dflash_draft" \
      --engineDir "${spec_dir}" \
      --maxBatchSize 1 \
      --maxInputLen "${DFLASH_MAX_INPUT_LEN:-1024}" \
      --maxKVCacheCapacity "${DFLASH_MAX_KV_CACHE_CAPACITY:-2048}" \
      --maxDraftTreeSize 16 \
      --specDraft
  for required in \
    spec_base.engine spec_draft.engine base_config.json draft_config.json \
    embedding.safetensors; do
    [ -s "${spec_dir}/${required}" ] || {
      echo "ERROR: v0.10 DFlash build did not produce ${spec_dir}/${required}" >&2
      exit 15
    }
  done
  _meta "${spec_dir}/spec_base.engine"
  _meta "${spec_dir}/spec_draft.engine"
  _provenance "${spec_dir}" \
    "${base_repo}+${draft_repo}" "${base_rev}+${draft_rev}" "${prec}"
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
    --minShapes=semantic_tokens:1x1,d_vector:1x1024 \
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
declare -A REPO REV DRAFT_REPO DRAFT_REV KIND PREC
REPO[qwen3-asr]="Qwen/Qwen3-ASR-0.6B";                  REV[qwen3-asr]="5eb144179a02acc5e5ba31e748d22b0cf3e303b0"; KIND[qwen3-asr]=asr; PREC[qwen3-asr]=int4_awq
REPO[qwen3-tts]="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"; REV[qwen3-tts]="85e237c12c027371202489a0ec509ded67b5e4b5"; KIND[qwen3-tts]=tts; PREC[qwen3-tts]=fp16
REPO[qwen3-tts-int4]="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"; REV[qwen3-tts-int4]="85e237c12c027371202489a0ec509ded67b5e4b5"; KIND[qwen3-tts-int4]=tts; PREC[qwen3-tts-int4]=int4
REPO[qwen3-tts-base]="Qwen/Qwen3-TTS-12Hz-0.6B-Base";   REV[qwen3-tts-base]="5d83992436eae1d760afd27aff78a71d676296fc"; KIND[qwen3-tts-base]=tts; PREC[qwen3-tts-base]=fp16
REPO[qwen3-tts-voicedesign]="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"; REV[qwen3-tts-voicedesign]="5ecdb67327fd37bb2e042aab12ff7391903235d3"; KIND[qwen3-tts-voicedesign]=tts; PREC[qwen3-tts-voicedesign]=fp16
REPO[qwen3.5-4b]="Qwen/Qwen3.5-4B";                     REV[qwen3.5-4b]="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"; KIND[qwen3.5-4b]=llm_base; PREC[qwen3.5-4b]=nvfp4
REPO[qwen3.5-4b-base]="Qwen/Qwen3.5-4B";                REV[qwen3.5-4b-base]="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"; KIND[qwen3.5-4b-base]=llm_base; PREC[qwen3.5-4b-base]=nvfp4
REPO[qwen3.5-4b-mtp]="Qwen/Qwen3.5-4B";                 REV[qwen3.5-4b-mtp]="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"; KIND[qwen3.5-4b-mtp]=llm_mtp; PREC[qwen3.5-4b-mtp]=nvfp4
REPO[qwen3.5-4b-dflash]="Qwen/Qwen3.5-4B";              REV[qwen3.5-4b-dflash]="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"; DRAFT_REPO[qwen3.5-4b-dflash]="z-lab/Qwen3.5-4B-DFlash"; DRAFT_REV[qwen3.5-4b-dflash]="9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf"; KIND[qwen3.5-4b-dflash]=llm_dflash; PREC[qwen3.5-4b-dflash]=nvfp4
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
    asr)      build_asr "$m" "${REPO[$m]}" "${prec}" "${REV[$m]}" ;;
    tts)      build_tts "$m" "${REPO[$m]}" "${prec}" "${REV[$m]}" ;;
    llm_base) build_llm_base "$m" "${REPO[$m]}" "${prec}" "${REV[$m]}" ;;
    llm_mtp)  build_llm_mtp "$m" "${REPO[$m]}" "${prec}" "${REV[$m]}" ;;
    llm_dflash) build_llm_dflash "$m" "${REPO[$m]}" "${DRAFT_REPO[$m]}" "${prec}" "${REV[$m]}" "${DRAFT_REV[$m]}" ;;
    moss)     build_moss "$m" "${REPO[$m]}" "${prec}" ;;
    sparktts) build_sparktts "$m" "${REPO[$m]}" "${prec}" ;;
  esac
done
echo "==> done. engines under ${EXPORT_ROOT}/<model>/  (host-matched .meta.json written)"
