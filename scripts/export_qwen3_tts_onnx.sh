#!/usr/bin/env bash
# Export Qwen3-TTS official Hugging Face weights to TensorRT-Edge-LLM ONNX.
#
# The default produces both official ONNX exports and the high-performance ONNX
# variants used by the current Orin runtime path when the required inputs exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR=""
OUT_DIR=""
EXPORT_PROJECT="${TRT_EXPORT_PROJECT:-/tmp/trt-export}"
TRT_SRC="${TRT_SRC:-$HOME/project/tensorrt-edge-llm}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
SETUP_ENV=0
DRY_RUN=0
HIGH_PERF=1
FP8_TEXT_EMBEDDING=1
W8A16_TALKER=1
STATEFUL_CODE2WAV=1
QWEN_TTS_ROOT="${QWEN_TTS_ROOT:-}"
CODE2WAV_CODES="${CODE2WAV_CODES:-}"
EXTRA_LLM_ARGS=()
EXTRA_AUDIO_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/export_qwen3_tts_onnx.sh --model-dir <Qwen3-TTS HF snapshot> --out <output-dir> [options]

Required:
  --model-dir PATH       Official Qwen3-TTS Hugging Face snapshot/local model directory.
  --out PATH             Output directory for ONNX exports.

Options:
  --export-project PATH  uv export env created by setup_trt_export_env.sh. Default: /tmp/trt-export
  --trt-src PATH         TensorRT-Edge-LLM fork checkout. Default: ~/project/tensorrt-edge-llm
  --setup-env            Run scripts/setup_trt_export_env.sh before exporting.
  --device DEVICE        cuda, cuda:0, or cpu. Default: cuda
  --dtype DTYPE          Audio export dtype. Default: fp16
  --official-only        Skip high-performance post-processing.
  --no-fp8-text-embedding
  --no-w8a16-talker
  --no-stateful-code2wav
  --qwen-tts-root PATH   qwen_tts Python package root for stateful Code2Wav export.
  --code2wav-codes PATH  safetensors file containing rvq_codes [T,Q] for Code2Wav validation.
                         If omitted, a tiny synthetic file is generated.
  --extra-llm-arg ARG    Append one raw argument to tensorrt-edgellm-export-llm. Repeatable.
  --extra-audio-arg ARG  Append one raw argument to tensorrt-edgellm-export-audio. Repeatable.
  --dry-run              Print commands without executing.

Output layout:
  <out>/official/llm                 EdgeLLM talker + code_predictor official ONNX export
  <out>/official/audio               EdgeLLM tokenizer_decoder/code2wav official ONNX export
  <out>/highperf/talker_w8a16        Optional W8A16 + output-k talker ONNX
  <out>/highperf/code_predictor      Optional optimized/pretransposed CP ONNX
  <out>/highperf/code2wav_stateful   Optional stateful Code2Wav ONNX
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --export-project) EXPORT_PROJECT="$2"; shift 2 ;;
    --trt-src) TRT_SRC="$2"; shift 2 ;;
    --setup-env) SETUP_ENV=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --official-only) HIGH_PERF=0; shift ;;
    --no-fp8-text-embedding) FP8_TEXT_EMBEDDING=0; shift ;;
    --no-w8a16-talker) W8A16_TALKER=0; shift ;;
    --no-stateful-code2wav) STATEFUL_CODE2WAV=0; shift ;;
    --qwen-tts-root) QWEN_TTS_ROOT="$2"; shift 2 ;;
    --code2wav-codes) CODE2WAV_CODES="$2"; shift 2 ;;
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

find_first() {
  local label="$1"
  shift
  local path
  for path in "$@"; do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  echo "Could not find $label. Checked:" >&2
  for path in "$@"; do echo "  $path" >&2; done
  return 1
}

find_first_dir() {
  local label="$1"
  shift
  local path
  for path in "$@"; do
    if [[ -d "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  echo "Could not find $label. Checked:" >&2
  for path in "$@"; do echo "  $path" >&2; done
  return 1
}

detect_qwen_tts_root() {
  if [[ -n "$QWEN_TTS_ROOT" ]]; then
    echo "$QWEN_TTS_ROOT"
    return 0
  fi
  (
    cd "$EXPORT_PROJECT"
    uv run python - <<'PY'
import pathlib
import qwen_tts
print(pathlib.Path(qwen_tts.__file__).resolve().parent)
PY
  )
}

create_synthetic_codes() {
  local output="$1"
  (
    cd "$EXPORT_PROJECT"
    uv run python - "$output" <<'PY'
import sys
import torch
from safetensors.torch import save_file
save_file({"rvq_codes": torch.zeros((16, 16), dtype=torch.int64)}, sys.argv[1])
PY
  )
}

OFFICIAL_DIR="$OUT_DIR/official"
LLM_OUT="$OFFICIAL_DIR/llm"
AUDIO_OUT="$OFFICIAL_DIR/audio"
mkdir -p "$LLM_OUT" "$AUDIO_OUT"

LLM_CMD=(uv run tensorrt-edgellm-export-llm --model_dir "$MODEL_DIR" --output_dir "$LLM_OUT" --export_models talker,code_predictor --device "$DEVICE")
if [[ ${#EXTRA_LLM_ARGS[@]} -gt 0 ]]; then LLM_CMD+=("${EXTRA_LLM_ARGS[@]}"); fi
AUDIO_CMD=(uv run tensorrt-edgellm-export-audio --model_dir "$MODEL_DIR" --output_dir "$AUDIO_OUT" --export_models tokenizer_decoder --dtype "$DTYPE" --device "$DEVICE")
if [[ ${#EXTRA_AUDIO_ARGS[@]} -gt 0 ]]; then AUDIO_CMD+=("${EXTRA_AUDIO_ARGS[@]}"); fi

if [[ "$DRY_RUN" == "1" ]]; then
  run_cmd "${LLM_CMD[@]}"
  run_cmd "${AUDIO_CMD[@]}"
else
  (
    cd "$EXPORT_PROJECT"
    run_cmd "${LLM_CMD[@]}"
    run_cmd "${AUDIO_CMD[@]}"
  )
fi

if [[ "$HIGH_PERF" == "1" ]]; then
  HIGHPERF_DIR="$OUT_DIR/highperf"
  mkdir -p "$HIGHPERF_DIR"

  TALKER_ONNX="$(find_first "talker ONNX" \
    "$LLM_OUT/talker/model.onnx" \
    "$LLM_OUT/talker/talker_decode.onnx" \
    "$LLM_OUT/onnx/talker/model.onnx" \
    "$LLM_OUT/onnx/talker_decode.onnx")" || exit 4
  CP_ONNX="$(find_first "code_predictor ONNX" \
    "$LLM_OUT/code_predictor/model.onnx" \
    "$LLM_OUT/code_predictor/cp_single_head_nopast.onnx" \
    "$LLM_OUT/onnx/code_predictor/model.onnx")" || exit 4

  if [[ "$W8A16_TALKER" == "1" ]]; then
    QUANT_SCRIPT="$TRT_SRC/scripts/quantize_onnx_matmul_w8a16.py"
    if [[ ! -f "$QUANT_SCRIPT" ]]; then
      echo "Missing W8A16 quantizer: $QUANT_SCRIPT" >&2
      exit 5
    fi
    TALKER_W8="$HIGHPERF_DIR/talker_w8a16/talker_decode_w8a16.onnx"
    TALKER_W8_OUTPUTK="$HIGHPERF_DIR/talker_w8a16/talker_decode_w8a16_outputk.onnx"
    run_cmd python3 "$QUANT_SCRIPT" \
      --input "$TALKER_ONNX" \
      --output "$TALKER_W8" \
      --external-data \
      --external-data-file "$(basename "$TALKER_W8").data" \
      --cast-plugin-inputs-to-fp16 \
      --cast-plugin-outputs-to-fp32
    run_cmd python3 "$SCRIPT_DIR/transpose_w8a16_onnx_layout.py" \
      --input "$TALKER_W8" \
      --output "$TALKER_W8_OUTPUTK" \
      --external-data-file "$(basename "$TALKER_W8_OUTPUTK").data"
  fi

  CP_OPT="$HIGHPERF_DIR/code_predictor/cp_single_head_nopast.onnx"
  CP_PRETRANSPOSE="$HIGHPERF_DIR/code_predictor/cp_single_head_nopast_lmhead_pretranspose.onnx"
  run_cmd python3 "$SCRIPT_DIR/optimize_qwen3_tts_cp_onnx.py" "$CP_ONNX" "$CP_OPT"
  run_cmd python3 "$SCRIPT_DIR/optimize_qwen3_tts_cp_lm_head_transpose.py" "$CP_OPT" "$CP_PRETRANSPOSE"

  if [[ "$FP8_TEXT_EMBEDDING" == "1" ]]; then
    TEXT_EMBED="$(find_first "text embedding safetensors" \
      "$LLM_OUT/talker/text_embedding.safetensors" \
      "$LLM_OUT/text_embedding.safetensors" \
      "$LLM_OUT/talker/embedding.safetensors")" || exit 4
    run_cmd python3 "$SCRIPT_DIR/quantize_embedding_safetensors_fp8.py" \
      "$TEXT_EMBED" "$HIGHPERF_DIR/talker/text_embedding.safetensors" \
      --tensor-name text_embedding \
      --scale-name text_embedding_scale
  fi

  if [[ "$STATEFUL_CODE2WAV" == "1" ]]; then
    QWEN_TTS_ROOT="$(detect_qwen_tts_root)"
    if [[ -z "$CODE2WAV_CODES" ]]; then
      CODE2WAV_CODES="$OUT_DIR/synthetic_rvq_codes.safetensors"
      if [[ "$DRY_RUN" != "1" ]]; then
        create_synthetic_codes "$CODE2WAV_CODES"
      else
        echo "+ create synthetic RVQ codes -> $CODE2WAV_CODES"
      fi
    fi
    CODE2WAV_DIR="$(find_first_dir "tokenizer decoder ONNX dir" \
      "$AUDIO_OUT/tokenizer_decoder" \
      "$AUDIO_OUT/onnx/tokenizer_decoder" \
      "$AUDIO_OUT/code2wav")" || exit 4
    run_cmd python3 "$SCRIPT_DIR/qwen3_tts_code2wav_stateful_export.py" \
      --onnx-dir "$CODE2WAV_DIR" \
      --codes "$CODE2WAV_CODES" \
      --qwen-tts-root "$QWEN_TTS_ROOT" \
      --max-frames 16 \
      --chunk-frames 4 \
      --export \
      --output "$HIGHPERF_DIR/code2wav_stateful/code2wav_stateful.onnx"
  fi
fi

if [[ "$HIGH_PERF" == "1" ]]; then
  HIGHPERF_JSON=true
else
  HIGHPERF_JSON=false
fi

cat > "$OUT_DIR/export_manifest.json" <<EOF
{
  "schema_version": 1,
  "model_type": "qwen3-tts",
  "model_dir": "$MODEL_DIR",
  "device": "$DEVICE",
  "dtype": "$DTYPE",
  "official_llm_dir": "$LLM_OUT",
  "official_audio_dir": "$AUDIO_OUT",
  "highperf_enabled": $HIGHPERF_JSON
}
EOF

echo "Qwen3-TTS ONNX export complete: $OUT_DIR"
