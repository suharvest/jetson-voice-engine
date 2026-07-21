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

build_sparktts() { # $1 model_id  $2 (unused)  $3 precision(int4_awq|w4a16_bf16)
  # SparkTTS-0.5B = Qwen2.5-0.5B LLM + BiCodec vocoder + speaker decoder. The
  # vocoder/speaker ONNX are exported by the (arch-independent) committed scripts
  # from the SparkTTS checkpoint; provide it via
  #   SPARKTTS_MODEL_DIR=<dir with BiCodec/>  SPARKTTS_REPO=<Spark-TTS repo root>
  # then trtexec (bicodec fp16 dyn-T, speaker fp32). The LLM goes through the
  # standard edge-llm quantize->export->llm_build.
  # NOTE: engine-level export is validated, but end-to-end synthesis is currently
  # BLOCKED by a v0.9.0 SparkTTS mixed-precision export regression (LLM emits text
  # instead of 32 global tokens). This function builds the pieces; synthesis needs
  # the regression fixed (or a v0.8.0 LLM export). See the reproducibility audit.
  local m="$1" prec="${3:-int4_awq}" out="${EXPORT_ROOT}/$1" trtexec
  local model="${SPARKTTS_MODEL_DIR:?set SPARKTTS_MODEL_DIR=<dir with BiCodec/> for sparktts}"
  local repo="${SPARKTTS_REPO:?set SPARKTTS_REPO=<Spark-TTS repo root> for sparktts}"
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
  echo "==> [sparktts:${m}] LLM quantize ${prec} -> export -> llm_build"
  ( cd "${UPSTREAM}"
    tensorrt-edgellm-quantize llm --model_dir "${model}/LLM" --output_dir "${out}/_q" --quantization "${prec}"
    tensorrt-edgellm-export "${out}/_q" "${out}/onnx" )
  "${BUILD}/examples/llm/llm_build" --onnxDir "${out}/onnx/llm" \
      --engineDir "${out}/sparktts-llm/engine" --maxBatchSize 2 --maxInputLen 4096 --maxKVCacheCapacity 4096
  _meta "${eng}/bicodec_decoder_dynT.fp16.engine"
  echo "==> [sparktts:${m}] WARNING: engines built, but e2e synthesis is blocked by the"
  echo "    v0.9.0 SparkTTS mixed-precision export regression (LLM emits text, not 32 global"
  echo "    tokens). Track the reproducibility audit before wiring into a live profile."
}

# --- manifest: model_id | hf_repo | kind | default_precision ---
declare -A REPO KIND PREC
REPO[qwen3-asr]="Qwen/Qwen3-ASR-0.6B";                  KIND[qwen3-asr]=asr;  PREC[qwen3-asr]=int4_awq
REPO[qwen3-tts]="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"; KIND[qwen3-tts]=tts;  PREC[qwen3-tts]=int4
REPO[qwen3.5-4b]="Qwen/Qwen3.5-4B";                     KIND[qwen3.5-4b]=llm; PREC[qwen3.5-4b]=nvfp4
# moss/sparktts are not plain HF-repo pipelines — sources come from env
# (MOSS_ONNX_BUNDLE / SPARKTTS_MODEL_DIR+SPARKTTS_REPO); REPO holds a hint only.
REPO[moss]="env:MOSS_ONNX_BUNDLE";                      KIND[moss]=moss;     PREC[moss]=mix1
REPO[sparktts]="env:SPARKTTS_MODEL_DIR";                KIND[sparktts]=sparktts; PREC[sparktts]=int4_awq

for m in "$@"; do
  [ -n "${REPO[$m]:-}" ] || { echo "unknown model '${m}' (add a manifest line)"; exit 2; }
  # per-model precision override: PRECISION_qwen3_5_4b=fp16 bash ...
  ov="PRECISION_${m//[.-]/_}"; prec="${!ov:-${PREC[$m]}}"
  case "${KIND[$m]}" in
    asr)      build_asr "$m" "${REPO[$m]}" "${prec}" ;;
    tts)      build_tts "$m" "${REPO[$m]}" "${prec}" ;;
    llm)      build_llm "$m" "${REPO[$m]}" "${prec}" ;;
    moss)     build_moss "$m" "${REPO[$m]}" "${prec}" ;;
    sparktts) build_sparktts "$m" "${REPO[$m]}" "${prec}" ;;
  esac
done
echo "==> done. engines under ${EXPORT_ROOT}/<model>/  (host-matched .meta.json written)"
