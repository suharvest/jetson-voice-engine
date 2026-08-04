#!/usr/bin/env bash
# Build MOSS-TTS-Nano TensorRT engines from ONNX on a Jetson device.
#
# Inputs (env vars or defaults):
#   ONNX_DIR        — dir containing moss_tts_*.onnx + .data + tts_browser_onnx_meta.json
#                     and browser_poc_manifest.json + tokenizer.model
#   CODEC_ONNX_DIR  — dir containing moss_audio_tokenizer_*.onnx + codec_browser_onnx_meta.json
#   OUT_DIR         — root of /opt/models/moss-tts-nano (engines/ + codec_onnx/ subdirs)
#   TRTEXEC         — path to trtexec (default /usr/src/tensorrt/bin/trtexec)
#
# Usage:
#   ONNX_DIR=$HOME/moss-onnx \
#   CODEC_ONNX_DIR=$HOME/moss-codec-onnx \
#   OUT_DIR=/opt/models/moss-tts-nano \
#   bash scripts/build_moss_tts_engines.sh

set -euo pipefail

# ---- Precision recipe ("mix1") ---------------------------------------------
# VALIDATED on DGX Spark (GB10, sm_121, TRT 10.14) 2026-07-13. A naive all-fp16
# build compiles cleanly but produces SILENT audio (RMS=0): the global
# transformer (prefill/decode_step) AND the codec attention caches are fp32
# precision-critical — fp16 collapses them to zeros. The sampler feeds valid
# non-zero frame_tokens either way, so the silence is downstream in fp16 globals
# + fp16 codec. Working recipe:
#   * prefill / decode_step  → FP32  (global transformer, precision-critical)
#   * local_*                → BF16 on GB10/sm_121, where fp16 collapses.
#                                NOT on Orin NX (sm_87, TRT 10.3): --bf16 there
#                                selects no bf16 kernel at all, so every weight
#                                stays fp32 and the three local engines cost
#                                +439 MB (227/230/231 MB instead of 65/115/69).
#                                --fp16 on sm_87 produces frame-identical audio
#                                — verified byte-for-byte on the three longest
#                                inputs. Set LOCAL_PREC=--fp16 on Orin.
#   * codec_decode_step      → FP32  (codec attn caches are float32 in meta;
#                                     runtime binds fp32 buffers, --fp16 → silence)
# NOTE: build the FP32 globals from the CLEAN full-precision ONNX (the non-paged
# moss-tts-nano-onnx export). A fresh FP32 build from the paged-fp16 ONNX still
# collapses — its weights were already truncated to fp16 precision.
# Override with GLOBAL_PREC / LOCAL_PREC / CODEC_PREC if a device needs different.
GLOBAL_PREC=${GLOBAL_PREC:-}          # empty = fp32 (no trtexec precision flag)
LOCAL_PREC=${LOCAL_PREC:---bf16}
CODEC_PREC=${CODEC_PREC:-}            # empty = fp32

ONNX_DIR=${ONNX_DIR:-/opt/models/moss-tts-nano/onnx}
CODEC_ONNX_DIR=${CODEC_ONNX_DIR:-/opt/models/moss-tts-nano/codec_onnx}
OUT_DIR=${OUT_DIR:-/opt/models/moss-tts-nano}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

ENGINES_DIR="${OUT_DIR}/engines"
CODEC_DIR="${OUT_DIR}/codec_onnx"

mkdir -p "${ENGINES_DIR}" "${CODEC_DIR}"

echo "[build] ONNX_DIR=${ONNX_DIR}"
echo "[build] CODEC_ONNX_DIR=${CODEC_ONNX_DIR}"
echo "[build] OUT_DIR=${OUT_DIR}"
echo "[build] trtexec: $(${TRTEXEC} --version 2>&1 | head -1)"

# ---- 1) Prefill engine ------------------------------------------------------
echo "[build] (1/6) moss_tts_prefill.plan ..."
# NOTE: prefill ONNX inputs are ONLY input_ids + attention_mask (no
# past_valid_lengths — that is a decode-step input). Shapes updated to match the
# current export (the old profile referenced a non-existent input and failed
# with "Cannot find input tensor past_valid_lengths").
# maxShapes raised to 1024: the MOSS prompt includes ~218 voice-conditioning
# code rows + text, so real prefill seq len exceeds the old 256 cap.
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_prefill.onnx" ${GLOBAL_PREC} \
  --minShapes=input_ids:1x1x17,attention_mask:1x1 \
  --optShapes=input_ids:1x256x17,attention_mask:1x256 \
  --maxShapes=input_ids:1x1024x17,attention_mask:1x1024 \
  --saveEngine="${ENGINES_DIR}/moss_tts_prefill.plan" 2>&1 | tail -8

# ---- 2) Decode step (12 layers KV cache; all past_key_*/past_value_* dynamic) ---
echo "[build] (2/6) moss_tts_decode_step.plan ..."
PAST_MIN="" PAST_OPT="" PAST_MAX=""
for i in $(seq 0 11); do
  PAST_MIN+="past_key_${i}:1x1x12x64,past_value_${i}:1x1x12x64,"
  PAST_OPT+="past_key_${i}:1x64x12x64,past_value_${i}:1x64x12x64,"
  PAST_MAX+="past_key_${i}:1x512x12x64,past_value_${i}:1x512x12x64,"
done
# NOTE: decode_step inputs are input_ids + past_valid_lengths + past_key_*/
# past_value_* (12 layers) — NO attention_mask (removed; the old profile listed
# a non-existent attention_mask input).
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_decode_step.onnx" ${GLOBAL_PREC} \
  --minShapes="input_ids:1x1x17,past_valid_lengths:1,${PAST_MIN%,}" \
  --optShapes="input_ids:1x1x17,past_valid_lengths:1,${PAST_OPT%,}" \
  --maxShapes="input_ids:1x1x17,past_valid_lengths:1,${PAST_MAX%,}" \
  --saveEngine="${ENGINES_DIR}/moss_tts_decode_step.plan" 2>&1 | tail -8

# ---- 3) Local decoder (one-shot, static shapes) ----------------------------
echo "[build] (3/6) moss_tts_local_decoder.plan ..."
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_local_decoder.onnx" ${LOCAL_PREC} \
  --saveEngine="${ENGINES_DIR}/moss_tts_local_decoder.plan" 2>&1 | tail -5

# ---- 4) Local cached step (required by the v0.9.1 release manifest) ---------
echo "[build] (4/6) moss_tts_local_cached_step.plan ..."
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_local_cached_step.onnx" ${LOCAL_PREC} \
  --saveEngine="${ENGINES_DIR}/moss_tts_local_cached_step.plan" 2>&1 | tail -5

# ---- 5) Local fixed sampled frame (production sampler, static) -------------
echo "[build] (5/6) moss_tts_local_fixed_sampled_frame.plan ..."
${TRTEXEC} --onnx="${ONNX_DIR}/moss_tts_local_fixed_sampled_frame.onnx" ${LOCAL_PREC} \
  --saveEngine="${ENGINES_DIR}/moss_tts_local_fixed_sampled_frame.plan" 2>&1 | tail -5

# ---- 6) Codec decode_step (audio_codes dynamic on frame count) -------------
echo "[build] (6/6) codec_decode_step.plan ..."
${TRTEXEC} --onnx="${CODEC_ONNX_DIR}/moss_audio_tokenizer_decode_step.onnx" ${CODEC_PREC} \
  --minShapes=audio_codes:1x1x16,audio_code_lengths:1 \
  --optShapes=audio_codes:1x4x16,audio_code_lengths:1 \
  --maxShapes=audio_codes:1x8x16,audio_code_lengths:1 \
  --saveEngine="${CODEC_DIR}/codec_decode_step.plan" 2>&1 | tail -8

# ---- Stage sidecar files (metadata, tokenizer, data files) -----------------
echo "[build] staging metadata + tokenizer + .data files ..."
for f in tts_browser_onnx_meta.json browser_poc_manifest.json tokenizer.model \
         moss_tts_global_shared.data moss_tts_local_shared.data; do
  [[ -f "${ONNX_DIR}/${f}" ]] && cp "${ONNX_DIR}/${f}" "${ENGINES_DIR}/${f}"
done
for f in codec_browser_onnx_meta.json moss_audio_tokenizer_encode.onnx \
         moss_audio_tokenizer_encode.data moss_audio_tokenizer_decode_shared.data; do
  [[ -f "${CODEC_ONNX_DIR}/${f}" ]] && cp "${CODEC_ONNX_DIR}/${f}" "${CODEC_DIR}/${f}"
done

for required in \
  "${ENGINES_DIR}/moss_tts_prefill.plan" \
  "${ENGINES_DIR}/moss_tts_decode_step.plan" \
  "${ENGINES_DIR}/moss_tts_local_decoder.plan" \
  "${ENGINES_DIR}/moss_tts_local_cached_step.plan" \
  "${ENGINES_DIR}/moss_tts_local_fixed_sampled_frame.plan" \
  "${ENGINES_DIR}/tokenizer.model" \
  "${CODEC_DIR}/codec_decode_step.plan" \
  "${CODEC_DIR}/codec_browser_onnx_meta.json" \
  "${CODEC_DIR}/moss_audio_tokenizer_decode_shared.data" \
  "${CODEC_DIR}/moss_audio_tokenizer_encode.onnx" \
  "${CODEC_DIR}/moss_audio_tokenizer_encode.data"; do
  if [[ ! -s "${required}" ]]; then
    echo "[build] ERROR: required v0.9.1 MOSS artifact missing or empty: ${required}" >&2
    exit 6
  fi
done

# Symlink codec assets into engines/ (worker hardcodes codec_*.{plan,json} lookup under engineDir).
ln -sf "${CODEC_DIR}/codec_decode_step.plan" "${ENGINES_DIR}/codec_decode_step.plan"
ln -sf "${CODEC_DIR}/codec_browser_onnx_meta.json" "${ENGINES_DIR}/codec_browser_onnx_meta.json"

echo "[build] DONE -> ${OUT_DIR}"
ls -lh "${ENGINES_DIR}" "${CODEC_DIR}"
