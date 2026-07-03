#!/usr/bin/env bash
# voxedge-engine build wrapper (overlay reproduction contract)
#
# CONTRACT
#   inputs : UPSTREAM_PIN, upstream.remote, addon/, patches/ (v080-sparktts series
#            + 0001 build-compat), a build manifest
#            (manifests/*.toml), target sm (e.g. sm_87 Orin), CUDA/TRT version, model src ref.
#   outputs: worker binaries  (qwen3_asr_worker [N>1] / qwen3_tts_worker / moss_tts_nano_worker)
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
# REMOTE source of the upstream fork. Default = upstream.remote file (the github fork).
# Override via EDGELLM_UPSTREAM_REMOTE for cross-wall / offline devices that cannot
# reach github.com — point it at a LOCAL fork checkout or git bundle, e.g.
#   EDGELLM_UPSTREAM_REMOTE=/home/harvest/project/edgellm-v080 bash build.sh
# (any path/URL `git clone` accepts; the PIN must be fetchable from it).
REMOTE="${EDGELLM_UPSTREAM_REMOTE:-$(grep -vE '^[[:space:]]*#' "${HERE}/upstream.remote" | head -1 | tr -d '[:space:]')}"
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
# Integration branch carries 3rdParty submodules (googletest, nlohmannJson, NVTX).
# Without this the CMake configure fails on the missing 3rdParty/* trees.
echo "==> submodule update --init --recursive (3rdParty: googletest, nlohmannJson, NVTX)"
git -C "${WORKDIR}" submodule update --init --recursive --depth 1

# --- 2. copy addon/ over the checkout ----------------------------------------
echo "==> copying addon/ (new files)"
# -a preserves exec bits; addon mirrors upstream relative paths
( cd "${HERE}/addon" && find . -type f -print0 | while IFS= read -r -d '' f; do
    dst="${WORKDIR}/${f#./}"
    mkdir -p "$(dirname "${dst}")"
    cp -p "${f}" "${dst}"
  done )

# --- 3. apply patches in order -----------------------------------------------
# C2 SPARKTTS RE-PIN (2026-07-03): UPSTREAM_PIN is the PURE NVIDIA v0.8.0 base
# (f9cc7462, release/0.8.0 HEAD). ALL fork content now travels as patches:
#
#   patches/v080-sparktts-0001..0030
#     = git format-patch f9cc7462..integration/v080-sparktts (fork HEAD 8437f027)
#     A strict superset of the previously-pinned branch (its 12 commits =
#     v080-sparktts-0001..0012: streaming worker + slot-pool + Base speaker-
#     encoder + sm_87 GEMM/GEMV + fp8 text_embedding + shared-engine ctor +
#     N<=0 guard + SessionLaneManager + CV 9-row runtime-if on langId) PLUS the
#     SparkTTS consolidation set (v080-sparktts-0013..0030: bf16/fp16
#     mixed-precision, INT4-AWQ/W4A16 mixed-precision export + plugin + GEMM,
#     borrowed/shared-engine ctor for voice-clone). All opt-in: default paths
#     byte-identical to upstream v0.8.0 behaviour.
#
# APPLY CHAIN on top of f9cc7462 + addon/:
#   + addon/   (new files: MOSS runtime+worker, w8a16 kernels, statefulCode2Wav,
#               spikes, scripts — all additive; copied in step 2 above)
#   + v080-sparktts-0001..0030   (in numeric order)
#   + 0001-orin-tegra-build-compat  (Orin/Tegra CUDA-12.6 build-host compat;
#               static-lib PUBLIC/INTERFACE shim + --wrap=_cudaLaunchKernelEx
#               propagation. NOT in the fork branch — verified git-apply CLEAN
#               on top of the full v080-sparktts chain.)
#
# Dry-run verified 2026-07-03: full chain git-apply --check CLEAN on f9cc7462;
# resulting tracked tree == fork integration/v080-sparktts exactly.
#
# NOT APPLIED — kept on disk (see PATCH-STATE-v080.md §13 disposition table):
#   - 0002-weight-streaming-budget-v080-OPTIN : rebased onto v0.8.0
#       builderUtils.cpp (engine-build memory budget). OPT-IN only — every
#       serve-gated v0.8.0 build was produced WITHOUT it.
#   - 0006/0007 server SSE-disconnect + OpenAI API (v0.7.1) : the v0.8.0 server
#       was REWRITTEN upstream and already ships its own tool_calling; the
#       SSE-disconnect watcher is still absent upstream but needs a
#       re-implementation against the new server (backlog), not a rebase.
#       PR-pending status of the SSE fix unchanged — do NOT auto-submit.
#   - 0008-build-misc-example-registration : omni streaming-worker registration
#       superseded by v080-sparktts-0001 (10b338d); its examples/llm spike
#       registrations target the deferred v0.7.1 spike API and would break the
#       v0.8.0 build. Archival.
#   - v080-0007 / v080-0008 (pre-runtime-if CV patches) : superseded by
#       v080-sparktts-0011/0012 (runtime-if on langId). Archival.
#   - v080-NNNN ASR-streaming / TTS-batch incremental-KV experiments : DEFERRED
#       (C3 backlog). N>1 ASR is delivered by the vendored worker
#       native/edgellm_voice_worker/qwen3_asr_worker.cpp on the vanilla one-shot
#       core + the asr-b2 engine (export artifact), selected via the n2 profile —
#       it needs NONE of these engine patches.
echo "==> applying patches"
apply_one() {
  local p="$1"
  echo "    - $(basename "${p}")"
  git -C "${WORKDIR}" apply --check "${p}"
  git -C "${WORKDIR}" apply "${p}"
}
# v080-sparktts series in numeric order (deterministic glob), then build-compat.
# Verified: full chain git-apply --check CLEAN on f9cc7462 + addon/.
for p in "${HERE}"/patches/v080-sparktts-00*.patch; do
  apply_one "${p}"
done
apply_one "${HERE}/patches/0001-orin-tegra-build-compat.patch"
echo "==> patched source tree ready at ${WORKDIR}"
echo "    v0.8.0 base (f9cc7462) + v080-sparktts-0001..0030 + 0001 build-compat."
echo "    Tracked tree == fork integration/v080-sparktts (HEAD 8437f027):"
echo "    streaming worker + slot-pool + shared-engine ctor + Base speaker-encoder"
echo "    + 9-row CV runtime-if (langId) + SparkTTS mixed-precision/int4 opt-ins."
echo "    MOSS via addon/. See patches/PATCH-STATE-v080.md §13 for dispositions."

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
# CUDA target SM. On aarch64/Tegra (Orin) the upstream CMakeLists, hardened by
# 0001, SKIPS the desktop "set(CMAKE_CUDA_ARCHITECTURES 80;86;89)" block — so
# nothing sets the arch and the int4 WOQ GEMM fails to compile with
# "cp.async requires sm_80+". Orin is sm_87, so default to 87 there (matching
# the EMBEDDED_TARGET=jetson-orin Tegra detection in 0001). Override via
# CMAKE_CUDA_ARCHITECTURES for other targets.
case "$(uname -m)" in
  aarch64) CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-87}" ;;
  *)       CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-}" ;;
esac
echo "==> cmake configure (CUDA_CTK_VERSION=${CUDA_CTK}, TRT_PACKAGE_DIR=${TRT_PKG}, CUDA_ARCH=${CUDA_ARCH:-<cmake default>}, Release)"
cmake -S "${WORKDIR}" -B "${WORKDIR}/build" \
      -DCUDA_CTK_VERSION="${CUDA_CTK}" \
      -DTRT_PACKAGE_DIR="${TRT_PKG}" \
      ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"} \
      -DCMAKE_BUILD_TYPE=Release
echo "==> make -j$(nproc) (engine core libedgellmCore.a + plugin .so + Base TTS N>1 streaming worker)"
cmake --build "${WORKDIR}/build" -j"$(nproc)"
# Explicitly ensure the Base N>1 streaming worker target is built. It lives in
# examples/omni/ (add_subdirectory(examples)->omni) and carries the slot-pool +
# shared-engine ctor (a361221) + Base speaker-encoder (external embedding) path.
# Output: ${WORKDIR}/build/examples/omni/qwen3_tts_streaming_worker
echo "==> ensure Base TTS N>1 streaming worker (qwen3_tts_streaming_worker)"
cmake --build "${WORKDIR}/build" -j"$(nproc)" --target qwen3_tts_streaming_worker

# --- 4b. ASR voice worker (N>1) — VERIFIED reproducible path -----------------
# Base TTS N>1 is the streaming worker built in step 4 above (examples/omni/
# qwen3_tts_streaming_worker — slot-pool + shared-engine ctor). This step builds
# the ASR side only: native/edgellm_voice_worker/ is a SEPARATE CMake project
# that links qwen3_asr_worker (N>1 lane-pool + streaming PARTIALs, v080-0023,
# binary md5 5ebd436b) against the libedgellmCore.a just built above. It is the
# formalization of the previously hand-built worker (v080-0021 "Box build dir
# ~/project/v080-worker-build"): now reproducible from THIS overlay.
# Source of truth = ${HERE}/../native/edgellm_voice_worker (vendored at feat HEAD).
VOICE_WORKER_SRC="${VOICE_WORKER_SRC:-${HERE}/../native/edgellm_voice_worker}"
if [ -f "${VOICE_WORKER_SRC}/CMakeLists.txt" ]; then
  echo "==> building ASR voice worker (qwen3_asr_worker N>1)"
  echo "    src=${VOICE_WORKER_SRC}  EDGE_LLM_BASE=${WORKDIR}  EDGE_LLM_BUILD=${WORKDIR}/build"
  cmake -S "${VOICE_WORKER_SRC}" -B "${WORKDIR}/build/voice-workers" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_CTK_VERSION="${CUDA_CTK}" \
        -DEDGE_LLM_SOURCE_DIR="${WORKDIR}" \
        -DEDGE_LLM_BUILD_DIR="${WORKDIR}/build"
  cmake --build "${WORKDIR}/build/voice-workers" -j"$(nproc)" \
        --target qwen3_asr_worker
else
  echo "WARN: ${VOICE_WORKER_SRC}/CMakeLists.txt not found — ASR worker NOT built." >&2
fi

# --- 4c. MOSS worker (own helper) --------------------------------------------
# Pin EDGELLM_SRC to THIS overlay's freshly-built clone (WORKDIR). Without it the
# helper defaults EDGELLM_SRC=/home/harvest/TensorRT-Edge-LLM — a stale per-dev
# checkout — so on a clean build the edgellmCore .o files it links are MISSING and
# the MOSS worker fails (BUILD_EXIT=1) even though TTS+ASR built fine from WORKDIR.
# (ORT_ROOT/SP_ROOT/CUDA_ROOT remain build-host env per the helper's defaults.)
if [ -x "${WORKDIR}/cpp/workers/build_moss_worker.sh" ]; then
  echo "==> building MOSS worker"
  ( cd "${WORKDIR}/cpp/workers" && EDGELLM_SRC="${WORKDIR}" bash build_moss_worker.sh )
fi
# --- 4d. plugin unversioned symlink -----------------------------------------
# The plugin builds as libNvInfer_edgellm_plugin.so.1.0 (VERSION 1.0/SOVERSION 1).
# The workers default to the UNVERSIONED relative path build/libNvInfer_edgellm_plugin.so.
# Create the unversioned symlink so the workers load it without an explicit
# EDGELLM_PLUGIN_PATH override. (Override still honored if exported.)
PLUGIN_VERSIONED="$(ls "${WORKDIR}/build"/libNvInfer_edgellm_plugin.so.* 2>/dev/null | head -1 || true)"
if [ -n "${PLUGIN_VERSIONED}" ]; then
  ln -sf "$(basename "${PLUGIN_VERSIONED}")" "${WORKDIR}/build/libNvInfer_edgellm_plugin.so"
  echo "==> plugin symlink: ${WORKDIR}/build/libNvInfer_edgellm_plugin.so -> $(basename "${PLUGIN_VERSIONED}")"
  echo "    (export EDGELLM_PLUGIN_PATH=${WORKDIR}/build/libNvInfer_edgellm_plugin.so to override)"
else
  echo "WARN: libNvInfer_edgellm_plugin.so.* not found in ${WORKDIR}/build — no symlink created." >&2
fi

echo "==> build done. Artifacts:"
echo "      ${WORKDIR}/build/                       libNvInfer_edgellm_plugin.so*"
echo "      ${WORKDIR}/build/examples/omni/         qwen3_tts_streaming_worker (Base N>1, slot-pool + shared-engine ctor)"
echo "      ${WORKDIR}/build/voice-workers/workers/ qwen3_asr_worker (N>1, 5ebd436b)"
echo "    Collect worker binaries + plugin .so + .engine, write md5 sidecars,"
echo "    and reconcile against ${MANIFEST}. Engine build uses build_engine_bundle.py."
echo "    ONE worker handles BOTH via runtime-if on langId:"
echo "      langId<0  -> Base: talker + code-predictor + speaker-encoder (external emb)"
echo "      langId>=0 -> CustomVoice: int4/fp8 CV talker engine"
echo "        (harvestsu/qwen3-tts-0.6b-customvoice-jetson-trtllm-int4fp8)."
echo "    The engine BUNDLE selected at runtime decides which path runs."
echo "    N=2 runtime: select profile jetson-edgellm-v080-n2 (stream_mode=worker,"
echo "    asr-b2 engine 4122dfcc, session-gate triplet LAZY_TTS=1 +"
echo "    OVS_TTS_WORKER_CONCURRENCY=2 + OVS_MAX_CONCURRENT_SESSIONS=2)."
