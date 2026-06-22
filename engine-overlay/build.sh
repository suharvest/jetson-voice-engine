#!/usr/bin/env bash
# voxedge-engine build wrapper (overlay reproduction contract)
#
# CONTRACT
#   inputs : UPSTREAM_PIN, upstream.remote, addon/, patches/0001..0008, a build manifest
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
# BASE + CUSTOMVOICE N>1 SERVING BUILD CHAIN (v0.8.0, CV re-pin / Option A).
#
# Ground truth: this overlay reproduces the Qwen3-TTS N>1 stack (streaming
# worker + slot-pool + shared-engine ctor) with BOTH Base AND CustomVoice in ONE
# binary via a RUNTIME-IF on langId (langId<0 => Base 8-row speaker-encoder
# prefill, UNCHANGED; langId>=0 => CustomVoice 9-row langId prefill). The single
# source for the FULL N>1 stack (TTS streaming worker + ASR voice worker + CV
# runtime-if) is the fork integration branch pinned in UPSTREAM_PIN
# (c48c0de = suharvest/wip/cv-9row-v080-n1n2), which is:
#
#     f9cc746  (NVIDIA release/0.8.0 HEAD)
#   + 10b338d  port streaming worker (base, N>1 slot-pool) onto v0.8.0
#   + ba9ecdb  backport Base speaker-encoder (external-embedding) path
#   + 867b74d  link cutedsl cudart shim into omni exes (CUDA 12.6 sm_87)
#   + 26a4a69  cuBLAS-free tiled FP16 GEMM fallback (talker MLP/linear, sm_87)
#   + 50b8670  warp-per-column M=1 GEMV for talker decode hot path (sm_87)
#   + 873ca22  fp8 text_embedding (native kernel path)
#   + a361221  D2-1 shared-engine ctor for slot-pool (Base N>1)
#   + 8de933f  C3 N<=0 prefill guard (defensive; N>0 paths unchanged)
#   + a099544  add minimal SessionLaneManager (lane allocator)
#   + 7142a30  SessionLaneManager accessor maxSessionBatchSize() —
#              the ASR voice worker's #include target (lane allocator ONLY, no
#              deferred streaming wrapper). Makes step 4b self-contained.
#   + 12ee383  port 9-row CustomVoice language conditioning onto v0.8.0 N>1
#              (RUNTIME-IF on langId; Base path unchanged when langId<0)
#   + c48c0de  wire per-request language into streaming worker buildRequest  <- HEAD
#
# Because all of these commits are IN the pinned branch, the old
# engine-overlay/patches/v080-port-0001..0006 (fork-port) are REDUNDANT and are
# NOT applied (kept on disk for archival only). See PATCH-STATE-v080.md §4 / §12.
#
# APPLY CHAIN = exactly ONE patch on top of the pinned branch + addon/:
#   + addon/   (new files: MOSS runtime+worker, w8a16 kernels, statefulCode2Wav,
#               spikes, scripts — all additive; copied in step 2 above)
#   + 0001-orin-tegra-build-compat  (Orin/Tegra CUDA-12.6 build-host compat;
#               static-lib PUBLIC/INTERFACE shim + --wrap=_cudaLaunchKernelEx
#               propagation. NOT in the fork branch — verified git-apply CLEAN.)
#
# DROPPED / SUPERSEDED — NOT applied (see PATCH-STATE-v080.md §4 / §10 / §12):
#   - v080-port-0001..0006  : redundant, now in the pinned branch.
#   - v080-0007-customvoice-language-conditioning  AND
#     0005-customvoice-language-conditioning : the OLD pre-runtime-if CV patches.
#       They forced a SEPARATE CV-only binary whose 9-row prefill could not
#       coexist with the Base 8-row path. SUPERSEDED by fork commit 12ee383,
#       which puts the 9-row CV prefill behind a RUNTIME-IF on langId so ONE
#       binary serves both Base (langId<0) and CV (langId>=0). These two patches
#       are now ARCHIVAL ONLY (kept on disk, never applied).
#   - v080-0008-tts-cutedsl-wrap : built on top of v080-0007's shim block, FAILS
#       git-apply on this branch (examples/omni/CMakeLists.txt context drift).
#       Superseded by the fork's 867b74d cutedsl shim + the
#       --wrap=_cudaLaunchKernelEx already in cmake/CuteDsl.cmake (CUDA<12.8) and
#       hardened by 0001. Archival only.
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
# Base+CV N>1 serving chain — explicit ordered allow-list (NOT a glob).
# Verified: full chain git-apply --check CLEAN on the pinned branch + addon/.
for n in \
    0001-orin-tegra-build-compat \
; do
  apply_one "${HERE}/patches/${n}.patch"
done
echo "==> patched source tree ready at ${WORKDIR}"
echo "    Base+CustomVoice N>1 chain applied (pinned branch c48c0de + 0001)."
echo "    Streaming worker + slot-pool + shared-engine ctor + Base speaker-encoder"
echo "    + 9-row CV runtime-if (langId) come from the pinned branch. MOSS via addon/."
echo "    v080-port-0001..0006 are redundant (in branch); the OLD CV patches"
echo "    (v080-0007 / 0005 / v080-0008) are SUPERSEDED by the in-branch runtime-if"
echo "    (12ee383/c48c0de) and intentionally NOT applied; v080-NNNN streaming/batch"
echo "    experiments are NOT applied either."
echo "    See patches/PATCH-STATE-v080.md (§4 chain, §10/§12 CustomVoice runtime-if)."

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
