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
# Apply ORDER (v0.8.0 re-pin, C2a):
#   1. legacy 000N theme patches that still clean-apply on v0.8.0 base: 0001, 0008
#      (0002/0003/0004/0005/0006/0007 are SUPERSEDED on v0.8.0 — see
#       patches/PATCH-STATE-v080.md; they are NOT applied here)
#   2. fork-port TTS-runtime stack v080-port-000N (canonical source =
#      fork port/qwen3-tts-base-v080; streaming worker, Base speaker-encoder,
#      cutedsl shim, cuBLAS-free GEMM, M=1 GEMV, fp8 text_embedding)
#   3. v080-NNNN ASR-streaming / MOSS / CustomVoice / TTS-batch feat patches
#      *** PENDING REBASE — several do NOT clean-apply from base yet ***
#      (double-source vs addon/, corrupt v080-0008, non-applicable
#       v080-0024..0027 worker-diff snapshots). See PATCH-STATE-v080.md.
echo "==> applying patches"
apply_one() {
  local p="$1"
  echo "    - $(basename "${p}")"
  git -C "${WORKDIR}" apply --check "${p}"
  git -C "${WORKDIR}" apply "${p}"
}
# 1) legacy clean-on-v080 theme patches (explicit allow-list, NOT a glob)
for n in 0001-orin-tegra-build-compat; do
  apply_one "${HERE}/patches/${n}.patch"
done
# 2) fork-port TTS-runtime stack (canonical) — verified clean-applies as a stack
for p in "${HERE}"/patches/v080-port-[0-9][0-9][0-9][0-9]-*.patch; do
  apply_one "${p}"
done
# 3) legacy 0008 example-registration:
#    omni-CMakeLists hunk now DUPLICATES the fork-port streaming-worker target
#    registration (v080-port-0001) -> conflicts. Pending hunk-split (drop omni
#    hunk, keep .gitignore + examples/llm/CMakeLists.txt). NOT applied until
#    rebased. See PATCH-STATE-v080.md.
# for n in 0008-build-misc-example-registration; do apply_one "${HERE}/patches/${n}.patch"; done
# 4) v080-NNNN ASR/MOSS/CV/TTS-batch feat patches — PENDING REBASE, NOT applied:
# for p in "${HERE}"/patches/v080-[0-9][0-9][0-9][0-9]-*.patch; do apply_one "${p}"; done
echo "==> patched source tree ready at ${WORKDIR}"
echo "    NOTE: legacy 0008 + v080-NNNN feat patches are PENDING REBASE (C2a)."
echo "          See patches/PATCH-STATE-v080.md before the next build."

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
