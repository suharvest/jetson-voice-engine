#!/usr/bin/env bash
# detect-target.sh — single source of truth for (SM arch, platform) of the
# edge-llm voice build. Replaces the ad-hoc, mutually-inconsistent hardcodes
# that assumed "aarch64 == Jetson Orin sm_87":
#   - build.sh main build defaulted aarch64 -> sm_87
#   - voice-workers cmake passed NO arch -> CMake default sm_75 (silent!)
#   - setup_pybind assumed AARCH64 -> jetson-thor
#
# Emits shell-eval'able assignments so every sub-build (main / voice-workers /
# pybind) shares ONE arch + platform. Two ORTHOGONAL axes:
#   TARGET_SM        e.g. 87 (Orin), 110 (Thor), 120 (RTX50), 121 (GB10/Spark)
#   TARGET_PLATFORM  tegra | sbsa | x86   (Tegra != arch — GB10 is Blackwell-on-sbsa)
#
# Usage:  eval "$(bash detect-target.sh)"     # or override via env before calling
#   TARGET_SM / TARGET_PLATFORM  — explicit override (skips autodetect)
set -euo pipefail

# --- SM arch: explicit env > nvidia-smi compute_cap > fail loud (never guess) ---
_detect_sm() {
  if [ -n "${TARGET_SM:-}" ]; then printf '%s' "${TARGET_SM}"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    # compute_cap "12.1" -> "121"; "8.7" -> "87"; "11.0" -> "110"
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    if [ -n "${cc}" ]; then printf '%s' "${cc}"; return; fi
  fi
  echo "detect-target: cannot detect SM arch (no nvidia-smi). Set TARGET_SM (e.g. 121a, 87, 110)." >&2
  exit 3
}

# --- platform: explicit env > Tegra markers > arch heuristic ---
_detect_platform() {
  if [ -n "${TARGET_PLATFORM:-}" ]; then printf '%s' "${TARGET_PLATFORM}"; return; fi
  case "$(uname -m)" in
    aarch64)
      # Jetson (Orin/Thor) is a Tegra SoC (l4t/JetPack); GB10/Spark is sbsa (DGX OS).
      if [ -f /etc/nv_tegra_release ] \
         || grep -qiE 'orin|thor|tegra' /proc/device-tree/model 2>/dev/null; then
        printf 'tegra'
      else
        printf 'sbsa'
      fi ;;
    x86_64) printf 'x86' ;;
    *)      printf 'unknown' ;;
  esac
}

SM="$(_detect_sm)"
PLATFORM="$(_detect_platform)"

# Blackwell (sm >= 100) links arch-specific SASS with the 'a' suffix; older
# arches (Orin sm_87, Ampere/Ada) do not. Callers may override via TARGET_SM.
ARCH_FLAG="${SM%a}"
case "${ARCH_FLAG}" in
  100|101|103|110|120|121) ARCH_FLAG="${ARCH_FLAG}a" ;;
esac
CUTE_TAG="sm_${SM%a}"

# EDGELLM_EMBEDDED_TARGET: only meaningful for Tegra (drives 0001 overlay Tegra
# link shims + CuTe tag inference). sbsa/x86 leave it empty (desktop path).
EMBEDDED_TARGET=""
if [ "${PLATFORM}" = "tegra" ]; then
  case "${SM%a}" in
    87)  EMBEDDED_TARGET="jetson-orin" ;;
    110) EMBEDDED_TARGET="auto-thor" ;;
  esac
fi

cat <<EOF
TARGET_SM=${SM%a}
TARGET_PLATFORM=${PLATFORM}
CMAKE_CUDA_ARCH=${ARCH_FLAG}
CUTE_DSL_ARTIFACT_TAG=${CUTE_TAG}
EDGELLM_EMBEDDED_TARGET=${EMBEDDED_TARGET}
EOF
