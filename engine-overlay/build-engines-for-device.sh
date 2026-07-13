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
#   qwen3-asr   | Qwen/Qwen3-ASR-0.6B                    | asr | int4_awq
#   qwen3-tts   | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice   | tts | int4      (talker int4 via driver scripts; cp/code2wav fp16)
#   qwen3.5-4b  | Qwen/Qwen3.5-4B                        | llm | nvfp4     (+ fp8 kv, GDN)
#   sparktts    | <SparkTTS-0.5B repo>                   | tts | bf16_hybrid  (same tts pipeline — add + validate)
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

# --- single source of truth for arch/platform ---
eval "$(bash "${HERE}/detect-target.sh")"
echo "==> target: SM=${TARGET_SM} platform=${TARGET_PLATFORM} arch=${CMAKE_CUDA_ARCH}"

export LD_LIBRARY_PATH="${BUILD}:${BUILD_GDN}:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export EDGELLM_PLUGIN_PATH="${PLUGIN}"

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
  local m="$1" repo="$2" prec="${3:-int4_awq}" src out
  src="$(_dl "${repo}")"; out="${EXPORT_ROOT}/${m}"
  echo "==> [asr:${m}] quantize ${prec} -> export(--fp8-embedding) -> thinker + audio encoder"
  ( cd "${UPSTREAM}"
    tensorrt-edgellm-quantize llm --model_dir "${src}" --output_dir "${out}/_q" --quantization "${prec}"
    tensorrt-edgellm-export "${out}/_q" "${out}/onnx" --fp8-embedding )
  "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/thinker" --maxBatchSize 2 --maxInputLen 4096 --maxKVCacheCapacity 4096
  "${BUILD}/examples/multimodal/audio_build" --onnxDir "${out}/onnx/audio" --engineDir "${out}/audio_encoder"
  _meta "${out}/thinker/llm.engine"; _meta "${out}/audio_encoder/audio/audio_encoder.engine"
}

build_tts() { # $1 model_id  $2 hf_repo  $3 precision(int4|fp16)
  local m="$1" repo="$2" prec="${3:-int4}" src out
  src="$(_dl "${repo}")"; out="${EXPORT_ROOT}/${m}"
  echo "==> [tts:${m}] talker(${prec}) + code_predictor + code2wav (fp16)"
  ( cd "${UPSTREAM}"
    if [ "${prec}" = "int4" ]; then
      # multimodal qwen3_tts can't load via standard quantize; extract talker as
      # vanilla Qwen3ForCausalLM (driver scripts), int4-AWQ, re-assemble ONNX.
      python3 tts-int4-drivers/quantize_talker_stage1.py --model_dir "${src}" --output_dir "${out}/_s1"
      python3 tts-int4-drivers/stage2_export.py --orig_model_dir "${src}" \
        --unified_dir "${out}/_s1/_hf_unified" --stage2_ckpt "${out}/_s1_ckpt" --onnx_out "${out}/onnx/llm"
      tensorrt-edgellm-export "${src}" "${out}/onnx" --components code_predictor,code2wav
    else
      tensorrt-edgellm-export "${src}" "${out}/onnx" --components talker,code_predictor,code2wav
    fi )
  "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/talker" --maxBatchSize 2 --maxInputLen 4096 --maxKVCacheCapacity 4096
  "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/code_predictor" \
      --engineDir "${out}/code_predictor" --maxBatchSize 2 --maxInputLen 4096 --maxKVCacheCapacity 4096
  "${BUILD}/examples/multimodal/audio_build" --onnxDir "${out}/onnx/code2wav" --engineDir "${out}/code2wav"
  # tokenizer dir the C++ worker loadFromHF() needs: tokenizer.json + config + chat template
  cp -n "${out}/talker/tokenizer.json" "${out}/talker/processed_chat_template.json" "${src}/" 2>/dev/null || true
  for e in "${out}/talker/llm.engine" "${out}/code_predictor/llm.engine" "${out}/code2wav/code2wav/code2wav.engine"; do _meta "${e}"; done
}

build_llm() { # $1 model_id  $2 hf_repo  $3 precision(nvfp4|fp16)
  local m="$1" repo="$2" prec="${3:-nvfp4}" src out
  src="$(_dl "${repo}")"; out="${EXPORT_ROOT}/${m}"
  echo "==> [llm:${m}] quantize ${prec}+fp8-kv -> export(--skip-visual) -> GDN engine"
  ( cd "${UPSTREAM}"
    tensorrt-edgellm-quantize llm --model_dir "${src}" --output_dir "${out}/_q" \
      --quantization "${prec}" --kv_cache_quantization fp8 --dataset abisee/cnn_dailymail
    tensorrt-edgellm-export "${out}/_q" "${out}/onnx" --skip-visual --skip-audio )
  # GDN needs the CuTe-enabled llm_build (build-gdn); tag auto-inferred from arch
  EDGELLM_PLUGIN_PATH="${BUILD_GDN}/libNvInfer_edgellm_plugin.so" \
  "${BUILD_GDN}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/engine" --maxBatchSize 1 --maxInputLen 4096 --maxKVCacheCapacity 8192
  _meta "${out}/engine/llm.engine"
}

# --- manifest: model_id | hf_repo | kind | default_precision ---
declare -A REPO KIND PREC
REPO[qwen3-asr]="Qwen/Qwen3-ASR-0.6B";                  KIND[qwen3-asr]=asr;  PREC[qwen3-asr]=int4_awq
REPO[qwen3-tts]="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"; KIND[qwen3-tts]=tts;  PREC[qwen3-tts]=int4
REPO[qwen3.5-4b]="Qwen/Qwen3.5-4B";                     KIND[qwen3.5-4b]=llm; PREC[qwen3.5-4b]=nvfp4

for m in "$@"; do
  [ -n "${REPO[$m]:-}" ] || { echo "unknown model '${m}' (add a manifest line)"; exit 2; }
  # per-model precision override: PRECISION_qwen3_5_4b=fp16 bash ...
  ov="PRECISION_${m//[.-]/_}"; prec="${!ov:-${PREC[$m]}}"
  case "${KIND[$m]}" in
    asr) build_asr "$m" "${REPO[$m]}" "${prec}" ;;
    tts) build_tts "$m" "${REPO[$m]}" "${prec}" ;;
    llm) build_llm "$m" "${REPO[$m]}" "${prec}" ;;
  esac
done
echo "==> done. engines under ${EXPORT_ROOT}/<model>/  (host-matched .meta.json written)"
