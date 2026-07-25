#!/usr/bin/env bash
# Export Qwen3-ASR official Hugging Face weights to TensorRT-Edge-LLM ONNX.
#
# This is the reproducible entry point for users who downloaded the official
# Qwen3-ASR weights themselves. It does not build TensorRT engines.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR=""
OUT_DIR=""
EXPORT_PROJECT="${TRT_EXPORT_PROJECT:-/tmp/trt-export}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
EXPORT_MODELS="${ASR_EXPORT_MODELS:-}"
REDUCED_VOCAB_DIR="${REDUCED_VOCAB_DIR:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
FP8_EMBEDDING="${FP8_EMBEDDING:-0}"
TRT_NATIVE_OPS="${TRT_NATIVE_OPS:-0}"
AUDIO_QUANTIZATION="${AUDIO_QUANTIZATION:-}"
SETUP_ENV=0
DRY_RUN=0
EXTRA_LLM_ARGS=()
EXTRA_AUDIO_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/export_qwen3_asr_onnx.sh --model-dir <Qwen3-ASR HF snapshot> --out <output-dir> [options]

Required:
  --model-dir PATH       Official Qwen3-ASR Hugging Face snapshot/local model directory.
  --out PATH             Output directory for EdgeLLM ONNX export.

Options:
  --export-project PATH  uv export env created by setup_trt_export_env.sh. Default: /tmp/trt-export
  --setup-env            Run scripts/setup_trt_export_env.sh before exporting.
  --device DEVICE        Compatibility option; v0.9.1 supports only cuda here.
  --dtype DTYPE          Audio export dtype. Default: fp16
  --export-models LIST   Compatibility option for LLM components. Default: thinker.
  --reduced-vocab-dir P  Optional EdgeLLM reduced vocab dir.
  --chat-template P      Removed by v0.9.1 unified export; using it fails.
  --fp8-embedding        Export LLM embedding sidecar as FP8 when supported.
  --trt-native-ops       Removed by v0.9.1 unified export; using it fails.
  --audio-quantization Q Removed by v0.9.1 unified export; using it fails.
  --extra-llm-arg ARG    Compatibility alias: append to tensorrt-edgellm-export.
  --extra-audio-arg ARG  Compatibility alias: append to tensorrt-edgellm-export.
  --dry-run              Print commands without executing.

Output layout:
  <out>/llm      Qwen3-ASR thinker ONNX + sidecars
  <out>/audio    Qwen3-ASR audio encoder ONNX + config
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --export-project) EXPORT_PROJECT="$2"; shift 2 ;;
    --setup-env) SETUP_ENV=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --export-models) EXPORT_MODELS="$2"; shift 2 ;;
    --reduced-vocab-dir) REDUCED_VOCAB_DIR="$2"; shift 2 ;;
    --chat-template) CHAT_TEMPLATE="$2"; shift 2 ;;
    --fp8-embedding) FP8_EMBEDDING=1; shift ;;
    --trt-native-ops) TRT_NATIVE_OPS=1; shift ;;
    --audio-quantization) AUDIO_QUANTIZATION="$2"; shift 2 ;;
    --extra-llm-arg) EXTRA_LLM_ARGS+=("$2"); shift 2 ;;
    --extra-audio-arg) EXTRA_AUDIO_ARGS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_DIR" || -z "$OUT_DIR" ]]; then
  usage >&2
  exit 2
fi

if [[ "$SETUP_ENV" == "1" ]]; then
  bash "$SCRIPT_DIR/setup_trt_export_env.sh"
fi

if [[ ! -d "$EXPORT_PROJECT" && "$DRY_RUN" != "1" ]]; then
  echo "Export project not found: $EXPORT_PROJECT" >&2
  echo "Run: bash $SCRIPT_DIR/setup_trt_export_env.sh" >&2
  exit 3
fi

MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

LLM_OUT="$OUT_DIR/llm"
AUDIO_OUT="$OUT_DIR/audio"

if [[ "$DEVICE" != "cuda" ]]; then
  echo "TensorRT Edge-LLM v0.9.1 unified export no longer accepts --device; this wrapper supports cuda only." >&2
  exit 6
fi
if [[ -n "$CHAT_TEMPLATE" || "$TRT_NATIVE_OPS" == "1" || -n "$AUDIO_QUANTIZATION" ]]; then
  echo "One or more requested legacy export options do not exist in TensorRT Edge-LLM v0.9.1." >&2
  exit 6
fi
LLM_COMPONENTS="${EXPORT_MODELS:-thinker}"
EXPORT_CMD=(uv run tensorrt-edgellm-export "$MODEL_DIR" "$OUT_DIR" \
  --components "${LLM_COMPONENTS},audio" --dtype "$DTYPE")
if [[ -n "$REDUCED_VOCAB_DIR" ]]; then EXPORT_CMD+=(--reduced-vocab-dir "$REDUCED_VOCAB_DIR"); fi
if [[ "$FP8_EMBEDDING" == "1" ]]; then EXPORT_CMD+=(--fp8-embedding); fi
if [[ ${#EXTRA_LLM_ARGS[@]} -gt 0 ]]; then EXPORT_CMD+=("${EXTRA_LLM_ARGS[@]}"); fi
if [[ ${#EXTRA_AUDIO_ARGS[@]} -gt 0 ]]; then EXPORT_CMD+=("${EXTRA_AUDIO_ARGS[@]}"); fi

if [[ "$DRY_RUN" == "1" ]]; then
  run_cmd "${EXPORT_CMD[@]}"
else
  (
    cd "$EXPORT_PROJECT"
    run_cmd "${EXPORT_CMD[@]}"
  )
fi

cat > "$OUT_DIR/export_manifest.json" <<EOF
{
  "schema_version": 1,
  "model_type": "qwen3-asr",
  "model_dir": "$MODEL_DIR",
  "device": "$DEVICE",
  "dtype": "$DTYPE",
  "llm_dir": "$LLM_OUT",
  "audio_dir": "$AUDIO_OUT"
}
EOF

echo "Qwen3-ASR ONNX export complete: $OUT_DIR"
