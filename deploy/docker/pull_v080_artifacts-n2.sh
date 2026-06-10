#!/usr/bin/env bash
#
# pull_v080_artifacts.sh — first-boot HF pull for the v0.8.0 edgellm stack.
#
# Slim-image pattern: the v0.8.0 Docker image bakes ONLY the 3 serving worker
# binaries + the edgellm plugin .so. The multi-GB TensorRT engines (ASR thinker,
# audio encoder, qwen3-tts talker/code_predictor/code2wav, plugin) are pulled
# from HuggingFace on first boot into fixed on-disk dirs, then the deploy_paths
# env (already exported by the Dockerfile) points the service at them.
#
# Source of truth: harvestsu/seeed-local-voice-artifacts
#                  sm87-trt10.3-jp6.2/v0.8.0/engines/{asr,qwen3-tts}
#                  sm87-trt10.3-jp6.2/v0.8.0/plugins/libNvInfer_edgellm_plugin.so
# (MOSS engines/codec are pulled separately by server.core.moss_artifacts from
#  models/moss-tts-nano/ — that path is left to the service's own provisioner.)
#
# Idempotent: a file already present is left untouched. HF_ENDPOINT mirror is
# honoured (hf-mirror.com on-device). CuTe DSL is statically linked into the
# plugin — the cutedsl tarball is NOT a runtime dependency and is NOT pulled.
set -euo pipefail

# ── Config (overridable) ────────────────────────────────────────────────────
EDGELLM_V080_AUTO_PULL="${EDGELLM_V080_AUTO_PULL:-1}"
HF_REPO="${EDGELLM_V080_HF_REPO:-harvestsu/seeed-local-voice-artifacts}"
HF_REV="${EDGELLM_V080_HF_REVISION:-main}"
HF_PREFIX="${EDGELLM_V080_HF_PREFIX:-sm87-trt10.3-jp6.2/v0.8.0}"
ROOT="${EDGELLM_V080_ROOT:-/opt/edgellm-v080}"

if [[ "${EDGELLM_V080_AUTO_PULL}" != "1" ]]; then
  echo "[v080-pull] EDGELLM_V080_AUTO_PULL=${EDGELLM_V080_AUTO_PULL} → skipping HF pull (artifacts must be pre-staged)."
  exit 0
fi

# Engine root layout produced here (matches the Dockerfile's deploy_paths env):
#   $ROOT/engines/asr/llm/*                     → EDGE_LLM_ASR_ENGINE_DIR
#   $ROOT/engines/asr/audio/audio/*             → EDGE_LLM_ASR_AUDIO_ENC_DIR=$ROOT/engines/asr/audio
#                                                 (the ASR adapter appends /audio,
#                                                  loading audio/audio_encoder.engine)
#   $ROOT/engines/qwen3-tts/talker/*            → EDGE_LLM_TTS_TALKER_DIR (+ tokenizer)
#   $ROOT/engines/qwen3-tts/code_predictor/*    → EDGE_LLM_TTS_CP_DIR
#   $ROOT/engines/qwen3-tts/code2wav/*          → EDGE_LLM_TTS_CODE2WAV_DIR
#   $ROOT/plugins/libNvInfer_edgellm_plugin.so  → EDGELLM_PLUGIN_PATH

# Sentinel: the N>1 b2 ASR thinker engine is the largest + last-needed file. Its
# presence means a previous pull completed (v080-0022 uses asr-b2/llm as the
# active EDGE_LLM_ASR_ENGINE_DIR).
ASR_ENGINE="${ROOT}/engines/asr-b2/llm/llm.engine"
if [[ -f "${ASR_ENGINE}" ]]; then
  echo "[v080-pull] v0.8.0 engines already present (${ASR_ENGINE}) — skipping HF pull."
  exit 0
fi

echo "[v080-pull] Pulling v0.8.0 engines from ${HF_REPO}@${HF_REV} (${HF_PREFIX}) into ${ROOT} ..."
echo "[v080-pull] HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}  (this may take 5-15 min on first boot)"

mkdir -p "${ROOT}"

# Pull only the v0.8.0 ASR + qwen3-tts engines + plugin (NOT gdn — that is
# served by edge-llm-chat-service; NOT toolchain/ — build-time only).
python3 - "$HF_REPO" "$HF_REV" "$HF_PREFIX" "$ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, rev, prefix, root = sys.argv[1:5]

# Allow-patterns are repo-relative (prefix-qualified). Pull ASR + qwen3-tts
# engines + the plugin .so only. gdn/, workers/ (baked), toolchain/ excluded.
allow = [
    f"{prefix}/engines/asr/**",
    f"{prefix}/engines/asr-b2/**",
    f"{prefix}/engines/qwen3-tts/**",
    f"{prefix}/plugins/libNvInfer_edgellm_plugin.so",
]
local = snapshot_download(
    repo_id=repo,
    revision=rev,
    local_dir="/tmp/v080-hf",
    allow_patterns=allow,
    max_workers=4,
)
print(f"[v080-pull] snapshot_download → {local}")
PY

# snapshot_download lands files under /tmp/v080-hf/<prefix>/...  Move them into
# the fixed ROOT layout (with the nested audio/audio/ the ASR adapter expects).
SRC="/tmp/v080-hf/${HF_PREFIX}"

mkdir -p \
  "${ROOT}/engines/asr/llm" \
  "${ROOT}/engines/asr-b2/llm" \
  "${ROOT}/engines/asr/audio/audio" \
  "${ROOT}/engines/qwen3-tts" \
  "${ROOT}/plugins"

# ASR thinker (llm/) — flat copy.
cp -an "${SRC}/engines/asr/llm/." "${ROOT}/engines/asr/llm/"
# v080-0022: N>1 b2 thinker (maxBatch=2). Only llm.engine + config.json differ
# from b1; the tokenizer/embedding/chat-template sidecars are b1/b2-identical, so
# stage the b1 sidecars first, then overlay the b2 engine + config on top.
cp -an "${SRC}/engines/asr/llm/." "${ROOT}/engines/asr-b2/llm/"
cp -af "${SRC}/engines/asr-b2/llm/llm.engine"   "${ROOT}/engines/asr-b2/llm/llm.engine"
cp -af "${SRC}/engines/asr-b2/llm/config.json"  "${ROOT}/engines/asr-b2/llm/config.json"
# ASR audio encoder — the adapter checks <dir>/audio/audio_encoder.engine, and
# EDGE_LLM_ASR_AUDIO_ENC_DIR is set to .../engines/asr/audio (one level up).
cp -an "${SRC}/engines/asr/audio/." "${ROOT}/engines/asr/audio/audio/"
# qwen3-tts talker / code_predictor / code2wav — keep subdirs as-is.
cp -an "${SRC}/engines/qwen3-tts/." "${ROOT}/engines/qwen3-tts/"
# plugin .so
cp -an "${SRC}/plugins/libNvInfer_edgellm_plugin.so" "${ROOT}/plugins/libNvInfer_edgellm_plugin.so"

# Reclaim the HF snapshot scratch (engines are now in ROOT). Only the throwaway
# /tmp/v080-hf cache is removed — never a real artifact dir.
rm -rf /tmp/v080-hf

echo "[v080-pull] v0.8.0 engines staged under ${ROOT}:"
ls -la "${ROOT}/engines/asr/llm" | head -6
ls -la "${ROOT}/engines/asr/audio/audio" | head -6
ls -la "${ROOT}/engines/qwen3-tts" | head
ls -la "${ROOT}/plugins"
echo "[v080-pull] done."
