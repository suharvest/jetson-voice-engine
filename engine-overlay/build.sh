#!/usr/bin/env bash
# voxedge-engine build wrapper (overlay reproduction contract)
#
# CONTRACT
#   inputs : UPSTREAM_PIN, upstream.remote, addon/, patches/0001..0008, a build manifest
#            (manifests/*.toml), target sm (e.g. sm_87 Orin), CUDA/TRT version, model src ref.
#   outputs: worker binaries  (qwen3_asr_worker / qwen3_tts_streaming_worker / moss_tts_nano_worker)
#            plugin .so       (libNvInfer_edgellm_plugin.so)
#            .engine artifacts (per manifest)
#            sidecar checksums (md5) for each produced artifact.
#
# REPRODUCTION FLOW
#   1. clone/fetch upstream.remote @ UPSTREAM_PIN into a clean workdir
#   2. copy addon/  over the checkout (new files, exec bits preserved)
#   3. apply patches/*.patch in NNNN order (git apply --check then git apply)
#   4. configure + build via the upstream CMake entry for the target sm
#   5. emit + verify artifact checksums against the chosen manifest
#
# ============================================================================
#  BUILD-VERIFY IS DEFERRED — REQUIRES A JETSON CUDA/TRT HOST.
#  This script is DOCUMENTATION + a dry-run harness. The actual compile steps
#  (cmake/make, trtexec, plugin build) MUST run on an Orin (sm_87) build host
#  with CUDA/TensorRT toolchain. macOS dev box has NO CUDA/TRT and MUST NOT build.
#  Run with --apply-only on any host to materialize the patched source tree
#  without compiling.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="$(grep -vE '^[[:space:]]*#' "${HERE}/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')"
REMOTE="$(grep -vE '^[[:space:]]*#' "${HERE}/upstream.remote" | head -1 | tr -d '[:space:]')"
WORKDIR="${VOXEDGE_WORKDIR:-${HERE}/.build/upstream}"
MANIFEST="${1:-}"
APPLY_ONLY=0
[ "${MANIFEST}" = "--apply-only" ] && { APPLY_ONLY=1; MANIFEST=""; }

echo "==> UPSTREAM_PIN : ${PIN}"
echo "==> upstream     : ${REMOTE}"
echo "==> workdir      : ${WORKDIR}"

# --- 1. fetch upstream @ pin --------------------------------------------------
if [ ! -d "${WORKDIR}/.git" ]; then
  echo "==> cloning upstream @ pin"
  git clone --no-checkout "${REMOTE}" "${WORKDIR}"
fi
git -C "${WORKDIR}" fetch --depth 1 origin "${PIN}" || git -C "${WORKDIR}" fetch origin
git -C "${WORKDIR}" checkout -q "${PIN}"
git -C "${WORKDIR}" clean -fdq

# --- 2. copy addon/ over the checkout ----------------------------------------
echo "==> copying addon/ (new files)"
# -a preserves exec bits; addon mirrors upstream relative paths
( cd "${HERE}/addon" && find . -type f -print0 | while IFS= read -r -d '' f; do
    dst="${WORKDIR}/${f#./}"
    mkdir -p "$(dirname "${dst}")"
    cp -p "${f}" "${dst}"
  done )

# --- 3. apply patches in order -----------------------------------------------
# AUTHORITATIVE SERVING BUILD CHAIN (v0.8.0, C2b).
#
# Ground truth: the v0.8.0 image that passed the real serve gate
# (docs/plans/v080-0019 in the feat tree) is a THIN binary overlay. Its 3 worker
# binaries + edgellm plugin were built from EXACTLY:
#
#     f9cc746  (NVIDIA release/0.8.0 HEAD)
#   + addon/   (new files: MOSS runtime+worker, w8a16 kernels, statefulCode2Wav,
#               spikes, scripts — all additive; copied in step 2 above)
#   + 0001-orin-tegra-build-compat   (Orin/Tegra CUDA-12.6 build-host compat)
#   + v080-0007-customvoice-language-conditioning (Qwen3OmniTTSRuntime struct port)
#   + v080-0008-tts-cutedsl-wrap     (CuTe-DSL --wrap shim, load-bearing toolchain)
#
# (HF MANIFEST v080-0017 §header: "edgellm base: f9cc746 + port patches
#  v080-0007 / v080-0008". Workers: native/edgellm_voice_worker/*.cpp +
#  addon/cpp/workers/moss_tts_nano_worker.cpp — compiled, NOT in this patch chain.)
#
# The shipped TTS worker is the GENERIC qwen3_tts_worker on Qwen3OmniTTSRuntime;
# ASR is one-shot (stream_mode=accumulate). The fork-port streaming-worker stack,
# the slot-pool, and all v080-NNNN ASR-streaming / TTS-batch / N>1 experiments are
# NOT part of the serving build and are DROPPED here (see PATCH-STATE-v080.md §4).
echo "==> applying patches"
apply_one() {
  local p="$1"
  echo "    - $(basename "${p}")"
  git -C "${WORKDIR}" apply --check "${p}"
  git -C "${WORKDIR}" apply "${p}"
}
# Authoritative serving chain — explicit ordered allow-list (NOT a glob).
# Verified: full chain git-apply --check CLEAN on f9cc746 + addon/ (C2b).
for n in \
    0001-orin-tegra-build-compat \
    v080-0007-customvoice-language-conditioning \
    v080-0008-tts-cutedsl-wrap \
; do
  apply_one "${HERE}/patches/${n}.patch"
done
echo "==> patched source tree ready at ${WORKDIR}"
echo "    Authoritative serving chain applied (0001 + v080-0007 + v080-0008)."
echo "    MOSS delivered via addon/. Fork-port stack, legacy 0002-0008, and"
echo "    v080-NNNN streaming/batch experiments are intentionally NOT applied —"
echo "    they were not in the serve-gated build. See patches/PATCH-STATE-v080.md."

if [ "${APPLY_ONLY}" -eq 1 ]; then
  echo "==> --apply-only: stopping before compile (no CUDA/TRT needed)."
  exit 0
fi

# --- 4. build (JETSON ONLY) ---------------------------------------------------
if [ -z "${MANIFEST}" ]; then
  echo "ERROR: build manifest required (manifests/*.toml). Usage: build.sh manifests/<name>.toml" >&2
  exit 2
fi
case "$(uname -m)" in
  aarch64) ;;  # Jetson/Orin OK
  *) echo "ERROR: compile step requires aarch64 Jetson host (sm_87 + CUDA/TRT). Aborting on $(uname -m)." >&2
     echo "       Use 'build.sh --apply-only' to just materialize the patched tree." >&2
     exit 3 ;;
esac

# Real CMake build — verified on Orin NX (CUDA 12.6 + TRT 10.3.0.30, recon
# 2026-05-31). The upstream entry is plain CMake (no build.sh upstream).
# CRITICAL: CMakeLists defaults CUDA_CTK_VERSION=12.8 and resolves
#   CUDA_DIR=/usr/local/cuda-${CUDA_CTK_VERSION}; on Jetson that path is 12.6,
#   so this MUST be overridden or cmake cannot find CUDA. TensorRT ships as a
#   system package → TRT_PACKAGE_DIR=/usr. Release build type matters
#   (empty type ≈ 2x slower runtime).
CUDA_CTK="${CUDA_CTK_VERSION:-12.6}"
TRT_PKG="${TRT_PACKAGE_DIR:-/usr}"
echo "==> cmake configure (CUDA_CTK_VERSION=${CUDA_CTK}, TRT_PACKAGE_DIR=${TRT_PKG}, Release)"
cmake -S "${WORKDIR}" -B "${WORKDIR}/build" \
      -DCUDA_CTK_VERSION="${CUDA_CTK}" \
      -DTRT_PACKAGE_DIR="${TRT_PKG}" \
      -DCMAKE_BUILD_TYPE=Release
echo "==> make -j$(nproc) (plugin .so + workers)"
cmake --build "${WORKDIR}/build" -j"$(nproc)"
# MOSS worker has its own helper (addon/cpp/workers/build_moss_worker.sh):
if [ -x "${WORKDIR}/cpp/workers/build_moss_worker.sh" ]; then
  echo "==> building MOSS worker"
  ( cd "${WORKDIR}/cpp/workers" && bash build_moss_worker.sh )
fi
echo "==> build done. Artifacts under ${WORKDIR}/build/ :"
echo "      libNvInfer_edgellm_plugin.so*  examples/omni/qwen3_tts_streaming_worker"
echo "    Collect worker binaries + plugin .so + .engine, write md5 sidecars,"
echo "    and reconcile against ${MANIFEST}. Engine build uses build_engine_bundle.py."
