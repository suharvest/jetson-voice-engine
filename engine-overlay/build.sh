#!/usr/bin/env bash
# voxedge-engine build wrapper (overlay reproduction contract)
#
# CONTRACT
#   inputs : UPSTREAM_PIN, upstream.remote, addon/, patches/v091-candidate/,
#            a build manifest
#            (manifests/*.toml), target sm (e.g. sm_87 Orin), CUDA/TRT version, model src ref.
#   outputs: worker binaries  (qwen3_asr_worker [N>1] / qwen3_tts_worker / moss_tts_nano_worker)
#            plugin .so       (libNvInfer_edgellm_plugin.so)
#            .engine artifacts (per manifest)
#            sidecar checksums (md5) for each produced artifact.
#
# REPRODUCTION FLOW
#   1. clone/fetch upstream.remote @ UPSTREAM_PIN into a clean workdir
#   2. copy addon/  over the checkout (new files, exec bits preserved)
#   3. validate + apply v091-candidate/0001..0041 in numeric order
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
#   EDGELLM_UPSTREAM_REMOTE=/home/harvest/project/edgellm-v091 bash build.sh
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
# Patches are applied with `git apply` onto tracked files; a plain checkout of
# the same PIN does NOT reset that dirty tree, so a re-run would fail every
# `git apply --check`. Hard-reset to the pin to make the script re-runnable.
git -C "${WORKDIR}" reset --hard -q "${PIN}"
git -C "${WORKDIR}" clean -fdq
# The full build requires the exact pinned 3rdParty submodules. Apply-only
# verifies only the overlay source delta and deliberately avoids network-heavy
# submodule initialization.
if [ "${APPLY_ONLY}" -eq 0 ]; then
  echo "==> submodule update --init --recursive (3rdParty: googletest, nlohmannJson, NVTX)"
  git -C "${WORKDIR}" submodule update --init --recursive --depth 1
else
  echo "==> --apply-only: skipping 3rdParty submodule initialization"
fi

# --- 2. copy addon/ over the checkout ----------------------------------------
echo "==> copying addon/ (new files)"
# -a preserves exec bits; addon mirrors upstream relative paths
( cd "${HERE}/addon" && find . -type f -print0 | while IFS= read -r -d '' f; do
    dst="${WORKDIR}/${f#./}"
    mkdir -p "$(dirname "${dst}")"
    cp -p "${f}" "${dst}"
  done )

# --- 3. apply the active v0.9.1 patch series in order -------------------------
# v0.8/v0.9.0 patch files remain on disk solely as rollback/history.
echo "==> applying patches"
apply_one() {
  local p="$1"
  echo "    - $(basename "${p}")"
  # Idempotent apply: if the patch already applies cleanly, apply it; if it is
  # ALREADY PRESENT in the pinned branch (forward --check fails but reverse
  # --check succeeds), skip it — the integration branch may have folded it in.
  # This keeps the overlay reproducible across future re-pins that absorb a
  # former patch. Only a genuine context conflict is fatal.
  if git -C "${WORKDIR}" apply --check "${p}" 2>/dev/null; then
    git -C "${WORKDIR}" apply "${p}"
  elif git -C "${WORKDIR}" apply --reverse --check "${p}" 2>/dev/null; then
    echo "      (already present in pinned branch — skipping)"
  else
    echo "ERROR: patch $(basename "${p}") does not apply and is not already present" >&2
    git -C "${WORKDIR}" apply --check "${p}"  # re-run to surface the real conflict
    return 1
  fi
}
PATCH_DIR="${HERE}/patches/v091-candidate"
shopt -s nullglob
PATCHES=("${PATCH_DIR}"/[0-9][0-9][0-9][0-9]-*.patch)
shopt -u nullglob
if [ "${#PATCHES[@]}" -eq 0 ]; then
  echo "ERROR: v0.9.1 patch series is empty: ${PATCH_DIR}" >&2
  exit 5
fi
expected=1
for p in "${PATCHES[@]}"; do
  actual="$(basename "${p}" | cut -c1-4)"
  printf -v wanted '%04d' "${expected}"
  if [ "${actual}" != "${wanted}" ]; then
    echo "ERROR: non-contiguous v0.9.1 patch series: expected ${wanted}, found ${actual}" >&2
    exit 5
  fi
  expected=$((expected + 1))
done
if [ "${#PATCHES[@]}" -ne 41 ]; then
  echo "ERROR: expected 41 v0.9.1 patches, found ${#PATCHES[@]}" >&2
  exit 5
fi
for p in "${PATCHES[@]}"; do
  apply_one "${p}"
done
echo "==> patched source tree ready at ${WORKDIR}"
echo "    v0.9.1 base (7f061f21) + v091-candidate/0001..0041."
echo "    streaming worker (v0.9.1 native streaming API) + slot-pool +"
echo "    shared-engine ctors + external speaker-embedding + 9-row CV runtime-if"
echo "    (langId) + SparkTTS mixed-precision/int4 opt-ins + MOSS (in-series)."
echo "    See patches/v091-candidate/PATCH-STATE.md for dispositions."

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
# Non-interactive shells (ssh/fleet exec) don't have nvcc on PATH → cmake dies
# with CMAKE_CUDA_COMPILER-NOTFOUND. Point PATH/CUDACXX at the toolkit.
if ! command -v nvcc >/dev/null 2>&1; then
  export PATH="/usr/local/cuda-${CUDA_CTK}/bin:${PATH}"
fi
export CUDACXX="${CUDACXX:-/usr/local/cuda-${CUDA_CTK}/bin/nvcc}"
TRT_PKG="${TRT_PACKAGE_DIR:-/usr}"
# Ensure nvcc is discoverable. Non-login shells (ssh exec / nohup / CI) often
# lack /usr/local/cuda-*/bin on PATH, so a CLEAN cmake configure fails with
# "CMAKE_CUDA_COMPILER-NOTFOUND" even though nvcc is installed. (Incremental
# rebuilds masked this by reusing a cached compiler path.) Prepend the CTK bin
# dir and export CUDACXX so the build entry is self-contained.
CUDA_BIN="/usr/local/cuda-${CUDA_CTK}/bin"
if ! command -v nvcc >/dev/null 2>&1; then
  if [ -x "${CUDA_BIN}/nvcc" ]; then
    export PATH="${CUDA_BIN}:${PATH}"
    export CUDACXX="${CUDA_BIN}/nvcc"
    echo "==> nvcc not on PATH; using ${CUDACXX}"
  else
    echo "ERROR: nvcc not found on PATH nor at ${CUDA_BIN}/nvcc. Install CUDA ${CUDA_CTK} toolkit or set PATH/CUDACXX." >&2
    exit 4
  fi
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-${CUDA_CTK}}"
# CUDA target SM + platform — SINGLE SOURCE OF TRUTH via detect-target.sh.
# The upstream CMakeLists (hardened by 0001) SKIPS the desktop
# "set(CMAKE_CUDA_ARCHITECTURES 80;86;89)" block on aarch64, so we MUST set the
# arch explicitly or int4 WOQ GEMM fails ("cp.async requires sm_80+").
# The old code hardcoded aarch64->sm_87 (assumed Jetson Orin); that silently
# mis-built on non-Orin aarch64 (GB10/Spark sm_121). detect-target.sh emits
# (TARGET_SM, TARGET_PLATFORM, CMAKE_CUDA_ARCH, CUTE_DSL_ARTIFACT_TAG,
# EDGELLM_EMBEDDED_TARGET) from nvidia-smi + platform markers. Override with
# CMAKE_CUDA_ARCHITECTURES or TARGET_SM/TARGET_PLATFORM.
_TGT="$(bash "${HERE}/detect-target.sh")" || { echo "ERROR: target detection failed; set TARGET_SM (e.g. 121a/87/110)." >&2; exit 3; }
eval "${_TGT}"
CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-${CMAKE_CUDA_ARCH}}"
echo "==> target: SM=${TARGET_SM} platform=${TARGET_PLATFORM} arch=${CUDA_ARCH} cute_tag=${CUTE_DSL_ARTIFACT_TAG} embedded='${EDGELLM_EMBEDDED_TARGET}'"
# ============================================================================
# DUAL BUILD CONFIGURATION (v0.9.1 migration) — pick per artifact family:
#
# (A) VOICE WORKER BUILDS (this script's default): ENABLE_CUTE_DSL=OFF.
#     The voice stack (qwen3_tts_streaming_worker / qwen3_asr_worker /
#     moss_tts_nano_worker + plugin) does NOT need CuTe DSL kernels — the
#     talker hot path uses our own sm_87 kernels from the patch series
#     (candidate 0002 tiled FP16 GEMM + 0003 warp-per-column M=1 GEMV).
#     Keep it OFF for the fallback release variant.
#
# (B) GDN LLM ENGINE BUILDS (Qwen3.5 GDN/MTP, SEPARATE build dir — not this
#     script): need ENABLE_CUTE_DSL=ALL (the GDN group is CuTe-DSL-only).
#     On Jetson Orin (sm_87, CUDA 12.6/JP6.2) that additionally requires:
#       1. regenerating the sm_87 CuTe DSL artifact ON the device with
#          the device-qualified cutlass-dsl 4.5.1 toolchain:
#          pip install nvidia-cutlass-dsl==4.5.1 &&
#          python kernelSrcs/build_cutedsl.py --gpu_arch sm_87
#          (the packaged v0.9.1 SM87 archive was built with CUDA 13.2 and is
#          incompatible with JP6.2 CUDA 12.6);
#       2. candidate 0039's shim/driver/wrap propagation;
#       3. -DAARCH64_BUILD=ON -DEMBEDDED_TARGET=jetson-orin and
#          -DCMAKE_CUDA_ARCHITECTURES=87.
#     Do NOT mix (A) and (B) artifacts in one build dir.
# ============================================================================
CUTE_DSL="${ENABLE_CUTE_DSL:-OFF}"
echo "==> cmake configure (CUDA_CTK_VERSION=${CUDA_CTK}, TRT_PACKAGE_DIR=${TRT_PKG}, CUDA_ARCH=${CUDA_ARCH:-<cmake default>}, ENABLE_CUTE_DSL=${CUTE_DSL}, Release)"
cmake -S "${WORKDIR}" -B "${WORKDIR}/build" \
      -DCUDA_CTK_VERSION="${CUDA_CTK}" \
      -DTRT_PACKAGE_DIR="${TRT_PKG}" \
      -DAARCH64_BUILD=ON \
      ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"} \
      ${CUTE_DSL_ARTIFACT_TAG:+-DCUTE_DSL_ARTIFACT_TAG="${CUTE_DSL_ARTIFACT_TAG}"} \
      ${EDGELLM_EMBEDDED_TARGET:+-DEMBEDDED_TARGET="${EDGELLM_EMBEDDED_TARGET}"} \
      -DENABLE_CUTE_DSL="${CUTE_DSL}" \
      -DCMAKE_BUILD_TYPE=Release
echo "==> make -j$(nproc) (engine core libedgellmCore.a + plugin .so + Base TTS N>1 streaming worker)"
cmake --build "${WORKDIR}/build" -j"$(nproc)"
# Explicitly ensure the Base N>1 streaming worker target is built. It lives in
# examples/omni/ (add_subdirectory(examples)->omni) and carries the slot-pool +
# shared-engine ctor (a361221) + Base speaker-encoder (external embedding) path.
# Output: ${WORKDIR}/build/examples/omni/qwen3_tts_streaming_worker
echo "==> ensure Base TTS N>1 streaming worker (qwen3_tts_streaming_worker)"
cmake --build "${WORKDIR}/build" -j"$(nproc)" --target qwen3_tts_streaming_worker
echo "==> ensure audio engine builder (audio_build)"
cmake --build "${WORKDIR}/build" -j"$(nproc)" --target audio_build
if [ ! -x "${WORKDIR}/build/examples/multimodal/audio_build" ]; then
  echo "ERROR: audio_build target completed without an executable artifact" >&2
  exit 4
fi

# --- 4b. ASR voice worker (N>1) — VERIFIED reproducible path -----------------
# Base TTS N>1 is the streaming worker built in step 4 above (examples/omni/
# qwen3_tts_streaming_worker — slot-pool + shared-engine ctor). This step builds
# the ASR side only: native/edgellm_voice_worker/ is a SEPARATE CMake project
# that links qwen3_asr_worker (N>1 lane-pool + streaming PARTIALs) against the
# libedgellmCore.a just built above. The worker sources were first adapted by
# port/v090-workers and are now compiled against the active v0.9.1 tree.
# Source of truth = ${HERE}/../native/edgellm_voice_worker (vendored).
VOICE_WORKER_SRC="${VOICE_WORKER_SRC:-${HERE}/../native/edgellm_voice_worker}"
if [ -f "${VOICE_WORKER_SRC}/CMakeLists.txt" ]; then
  echo "==> building ASR voice worker (qwen3_asr_worker N>1)"
  echo "    src=${VOICE_WORKER_SRC}  EDGE_LLM_BASE=${WORKDIR}  EDGE_LLM_BUILD=${WORKDIR}/build"
  # CRITICAL: propagate the target arch. Without -DCMAKE_CUDA_ARCHITECTURES this
  # sub-build inherited CMake's compiler default (sm_75) — SILENTLY, so the main
  # build was sm_121a but qwen3_asr_worker was sm_75 → "no kernel image is
  # available for execution on the device" at runtime on GB10. Must match the
  # main build's ${CUDA_ARCH}.
  cmake -S "${VOICE_WORKER_SRC}" -B "${WORKDIR}/build/voice-workers" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_CTK_VERSION="${CUDA_CTK}" \
        ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"} \
        -DEDGE_LLM_SOURCE_DIR="${WORKDIR}" \
        -DEDGE_LLM_BUILD_DIR="${WORKDIR}/build"
  cmake --build "${WORKDIR}/build/voice-workers" -j"$(nproc)" \
        --target qwen3_asr_worker spark_tts_worker
else
  echo "WARN: ${VOICE_WORKER_SRC}/CMakeLists.txt not found — ASR worker NOT built." >&2
fi

# --- 4c. MOSS worker ----------------------------------------------------------
# Candidate 0031/0032 carries the worker in examples/omni/moss_tts_nano_worker.cpp
# and registered the CMake target `moss_tts_nano_worker` (needs onnxruntime via
# ORT_ROOT env / /usr/local/onnxruntime / ~/ort-from-container + SentencePiece;
# the target is SKIPPED with a STATUS message when deps are missing, so this is
# best-effort). The old cpp/workers/build_moss_worker.sh helper is LEGACY
# reference only — do not use it on the v0.9.1 chain.
echo "==> building MOSS worker (cmake target moss_tts_nano_worker, best-effort)"
cmake --build "${WORKDIR}/build" -j"$(nproc)" --target moss_tts_nano_worker \
  || echo "WARN: moss_tts_nano_worker target unavailable (ORT/SentencePiece missing?) — skipped." >&2
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
echo "      ${WORKDIR}/build/examples/omni/         moss_tts_nano_worker (if ORT/SP present)"
echo "      ${WORKDIR}/build/voice-workers/workers/ qwen3_asr_worker (N>1)"
echo "    Collect worker binaries + plugin .so + .engine, write md5 sidecars,"
echo "    and reconcile against ${MANIFEST}. Engine build uses build_engine_bundle.py."
echo "    ONE worker handles BOTH via runtime-if on langId:"
echo "      langId<0  -> Base: talker + code-predictor + speaker-encoder (external emb)"
echo "      langId>=0 -> CustomVoice: int4/fp8 CV talker engine"
echo "        (harvestsu/qwen3-tts-0.6b-customvoice-jetson-trtllm-int4fp8)."
echo "    The engine BUNDLE selected at runtime decides which path runs."
echo "    N=2 runtime: select profile jetson-edgellm-v091-n2 (stream_mode=worker,"
echo "    asr-b2 engine 4122dfcc, session-gate triplet LAZY_TTS=1 +"
echo "    OVS_TTS_WORKER_CONCURRENCY=2 + OVS_MAX_CONCURRENT_SESSIONS=2)."
