#!/usr/bin/env bash
#
# entrypoint.jetson.v080.sh — v0.8.0 edgellm overlay entrypoint.
#
# Runs the first-boot HF pull of the v0.8.0 ASR/TTS engines + plugin, then
# chains to the canonical profile-resolving entrypoint (which sources
# /etc/openvoicestream.env, resolves OVS_PROFILE, and exec's the CMD).
#
# The deploy_paths env (EDGE_LLM_ASR_*, EDGE_LLM_TTS_*, MOSS_*, EDGELLM_PLUGIN_PATH)
# is baked as image ENV by the Dockerfile and points at the fixed pull dirs.
set -euo pipefail

# First-boot pull of the v0.8.0 engines + plugin (idempotent; HF_ENDPOINT mirror
# honoured). Fail-open: a pull error is logged but does not block startup — the
# backend preload's own FileNotFoundError remains the correctness gate.
if ! /opt/speech/pull_v080_artifacts.sh; then
  echo "[v080-entrypoint] WARNING: v0.8.0 artifact pull returned non-zero; backend preload will re-check." >&2
fi

# Chain to the canonical profile-resolving entrypoint.
exec /opt/speech/entrypoint.jetson.sh "$@"
