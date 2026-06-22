# Patch state — v0.8.0 Base TTS N>1 serving chain (C2-repin)

> **CURRENT AUTHORITATIVE STATE = §C2-repin below (this section).**
> Produced by **C2-repin** (Mac, git/source only — no engine build, no deploy).
> Finalizes the overlay to reproduce the **Base Qwen3-TTS N>1** stack (streaming
> worker + slot-pool + shared-engine ctor, NO CustomVoice) by repinning to the
> fork integration branch that single-sources it.
>
> The historical C2a / C2b sections (§0–§9 below) are kept for provenance but are
> **superseded** by C2-repin on three points: (1) the pin is now the fork
> integration branch a361221, not NVIDIA `f9cc746`; (2) the canonical TTS path is
> the fork-port **streaming worker** (N>1), not the generic `qwen3_tts_worker`;
> (3) the apply chain is **0001 only** — `v080-0007`/`v080-0008` move to the
> CustomVoice variant. Read §C2-repin and §10 first.

---

## §C2-repin — Base TTS N>1 (CURRENT, 2026-06-21)

### Decision (user, 2026-06-21)
Ship **Base N>1** now; **CustomVoice becomes an independent first-class variant**
(later, see §10). Reason: the CustomVoice 9-row langId patch (`v080-0007`)
conflicts with the fork's Base speaker-encoder kernel in talker prefill (8-row vs
9-row — cannot coexist in one binary). The verified N>1 path uses the **Base**
talker. → Base and CV are **two build variants**; this overlay builds **Base only**.

### Pin (supersedes §1)
`UPSTREAM_PIN = a3612217d167772744aacac8be8df8ecd226dc1c`
= fork integration branch **`suharvest/port/qwen3-tts-base-v080-n1n2`** HEAD.
This branch IS the single source for Base N>1:
```
f9cc746  NVIDIA release/0.8.0 HEAD (PR #101)
10b338d  port streaming worker (base, N>1 slot-pool) onto v0.8.0
ba9ecdb  backport Base speaker-encoder (external-embedding) path onto v0.8.0
867b74d  link cutedsl cudart shim into omni exes (CUDA 12.6 sm_87)
26a4a69  cuBLAS-free tiled FP16 GEMM fallback (talker MLP/linear, sm_87)
50b8670  warp-per-column M=1 GEMV for talker decode hot path (sm_87)
873ca22  fp8 text_embedding (native kernel path)
a361221  D2-1 shared-engine ctor for slot-pool (Base N>1)   ← HEAD
```
`upstream.remote` repointed to the suharvest fork (NVIDIA HEAD is not reachable
to a fork-only SHA).

### Apply chain (supersedes §2 / §9.4)
On top of the pinned branch + `addon/`, build.sh applies **exactly ONE** patch:

| # | patch | disposition on a361221 | reason |
|---|---|---|---|
| 1 | `0001-orin-tegra-build-compat` | **KEEP** — git-apply --check CLEAN | NOT in the fork branch. Orin/Tegra CUDA-12.6 build-host compat: `CMakeLists.txt` + `cmake/CuteDsl.cmake` (static-lib PUBLIC/INTERFACE shim + `--wrap=_cudaLaunchKernelEx` propagation) + `cpp/CMakeLists.txt`. The fork's `cute_dsl_setup` foreach region is unchanged from f9cc746, so 0001's pre-image still matches. |

Removed / re-homed (NOT in the Base apply chain):

| patch(es) | disposition | reason (vs the pinned fork branch) |
|---|---|---|
| `v080-port-0001..0006` | **REMOVED — redundant** | These six patches ARE the six fork-port commits (10b338d…873ca22), now IN the pinned branch. Symbols confirmed present in checkout: `examples/omni/qwen3_tts_streaming_worker.cpp` (md5 `0d752b42`), `cpp/runtime/slotPool.h`, shared-engine ctor `Qwen3OmniTTSRuntime(ICudaEngine*, ICudaEngine*, …)` + `initializeEngineRunnersShared`, Base speaker-encoder external-embedding at `qwen3OmniTTSRuntime.cpp:949/988`. Forward-apply no-ops; reverse-apply drifts (branch final state ≠ intermediate-patch state). Kept on disk for archival. |
| `v080-0007-customvoice-language-conditioning` | **MOVED → CV variant (§10)** | CustomVoice 9-row langId conditioning. Conflicts with Base 8-row speaker-encoder talker prefill. NOT applied to Base. |
| `v080-0008-tts-cutedsl-wrap` | **REMOVED — superseded + CV-coupled** | Built on top of v080-0007's shim block; **FAILS git-apply** on the Base branch (`examples/omni/CMakeLists.txt:17` context drift — the fork's 867b74d rewrote that area). Its `--wrap=_cudaLaunchKernelEx` is already provided for Base by `cmake/CuteDsl.cmake` (CUDA<12.8, applied to LINK_TARGETS incl. the static `edgellmCore`) and hardened by 0001. Belongs to the CV variant. |
| `v080-NNNN` ASR-streaming / TTS-batch incremental-KV | **DEFERRED (C3)** | Unchanged from §5/§9.1. N>1 ASR is the vendored worker on the vanilla one-shot core + asr-b2 engine, selected via the n2 profile. Needs no engine patch. |

### apply-check verdict (temp worktree @ a361221 + addon/)
- `0001-orin-tegra-build-compat`: **CHECK CLEAN → APPLY OK** (full ordered chain, 0 conflicts).
- `v080-0008-tts-cutedsl-wrap`: **FAIL** (`examples/omni/CMakeLists.txt:17` — confirms it is CV-chain-coupled, correctly dropped).
- `v080-port-0001..0006`: do not forward-apply (content already in branch) → confirmed redundant.

### build.sh (Base N>1)
- step 4 builds the engine core + plugin **and** the Base N>1 streaming worker
  (`examples/omni/qwen3_tts_streaming_worker`, via `add_subdirectory(examples)→omni`;
  explicit `--target qwen3_tts_streaming_worker`). This carries the slot-pool +
  shared-engine ctor + Base speaker-encoder.
- step 4b builds the ASR side only (`qwen3_asr_worker` N>1, from
  `native/edgellm_voice_worker/`). The generic `qwen3_tts_worker` target is no
  longer the TTS path — the streaming worker is.
- Base TTS engines = talker + code-predictor + **speaker-encoder** (external
  embedding); NOT the CustomVoice int4 engine.

### n2 profile — UNCHANGED, retained
`deploy/docker/jetson-edgellm-v080-n2.json` + `Dockerfile.jetson.v080-edgellm-n2`
keep the session-gate triplet (`LAZY_TTS=1` + `OVS_TTS_WORKER_CONCURRENCY=2` +
`OVS_MAX_CONCURRENT_SESSIONS=2`), asr-b2 engine dir, and prefix flags. See §9.3.

### §C2b — build.sh self-sufficiency + ASR-worker header gap (2026-06-21)

C2b real-run found build.sh built the TTS streaming worker (Base N>1, smoke
PASS) but was NOT self-sufficient. Four gaps, all resolved here:

1. **submodules** — checkout did not init `3rdParty/{googletest,nlohmannJson,NVTX}`.
   → build.sh now runs `git submodule update --init --recursive --depth 1` after
   checkout + clean.
2. **sm_87 arch** — the cmake configure passed no `CMAKE_CUDA_ARCHITECTURES`. On
   aarch64 the upstream `set(CMAKE_CUDA_ARCHITECTURES 80;86;89)` block is SKIPPED
   (0001 guards it with `NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"`), so the
   arch was left unset → int4 WOQ GEMM `cp.async requires sm_80+`. → build.sh now
   defaults `-DCMAKE_CUDA_ARCHITECTURES=87` on aarch64 (override via env), matching
   the EMBEDDED_TARGET=jetson-orin Tegra detection in 0001.
3. **plugin path** — the plugin builds as `libNvInfer_edgellm_plugin.so.1.0`
   (VERSION 1.0/SOVERSION 1) but workers default to the unversioned relative
   `build/libNvInfer_edgellm_plugin.so`. → build.sh step 4d now `ln -sf` the
   unversioned symlink and prints the `EDGELLM_PLUGIN_PATH` override.
4. **ASR worker header gap (KEY)** — `qwen3_asr_worker.cpp` does
   `#include "runtime/asrStreamingSessionRuntime.h"` for `rt::SessionLaneManager`
   + `rt::LaneOwnerKind::kAsr` (lane RESERVATION only; the worker runs the vanilla
   one-shot `LLMInferenceRuntime` core, NOT the chunked-streaming wrapper). That
   header did NOT exist on the Base port branch — it was only ever created by the
   DEFERRED ASR-streaming spike `v080-0002` (full `AsrStreamingSessionRuntime`
   incremental-prefill runtime) and extended by `v080-0005` (the lane manager).

   **Solution chosen = 甲 (single source):** add a MINIMAL
   `cpp/runtime/asrStreamingSessionRuntime.{h,cpp}` to the fork integration branch
   containing ONLY `LaneOwnerKind` + `SessionLaneManager` (verbatim from the
   v080-0005 definitions). The full `AsrStreamingSessionRuntime` streaming wrapper
   (appendChunk / beginAsrSession / per-lane KV reset + its `LLMInferenceRuntime`
   hooks) is intentionally EXCLUDED — so the deferred streaming spike and any
   talker-prefill changes are NOT dragged into the Base build. Rationale: the
   lane manager is fully self-contained (`<cstdint>/<mutex>/<vector>/<stdexcept>`
   + `LOG_ERROR`); it has no dependency on the streaming wrapper. The `.cpp` is
   auto-globbed into `edgellmCore` by the existing
   `file(GLOB_RECURSE RUNTIME_CPP_SRCS "runtime/*.cpp")` — no CMake edit needed.

   A SECOND, smaller gap surfaced once the include resolved: the worker also calls
   `runtime->maxSessionBatchSize()` (to size the lane manager to the engine's
   physical batch). That accessor existed only on the deferred spike's modified
   runtime. Added as a trivial read-only inline accessor on `LLMInferenceRuntime`
   (`return mDeployment.base.maxSupportedBatchSize;` — same field already used at
   `llmInferenceRuntime.cpp:293`). No streaming state; deferred wrapper still out.

   Fork commits (on suharvest fork branch, on top of `8de933f`):
   - `a099544` add minimal SessionLaneManager (`asrStreamingSessionRuntime.{h,cpp}`)
   - `7142a30` expose `maxSessionBatchSize()` on LLMInferenceRuntime   ← new HEAD

**Repin:** `UPSTREAM_PIN a361221 → 7142a30` (= new branch HEAD). Three commits
added on top of the previous pin: `8de933f` (C3 N<=0 prefill guard — defensive
early-return; N>0 paths unchanged), `a099544` (lane-manager header), `7142a30`
(maxSessionBatchSize accessor). Apply chain UNCHANGED (still exactly
`0001-orin-tegra-build-compat` on top of branch + addon/).

**Targeted verify (orin-nx, jve-c2b/.build/upstream):** dropped the new files in,
reconfigured (refresh source glob), `cmake --build build --target edgellmCore`
(compiles asrStreamingSessionRuntime.cpp; `SessionLaneManager::acquire` now in
libedgellmCore.a) then `cmake --build build/voice-workers --target qwen3_asr_worker`
→ **RC=0**, ARM aarch64 ELF `qwen3_asr_worker` (17.9 MB, md5 d9c863e2). No full
rebuild. MOSS worker (step 4c) layout is a separate follow-up; not blocking Base
N>1 + ASR.

---

## (HISTORICAL) Patch state — v0.8.0 authoritative serving chain (C2b)

> **SUPERSEDED by §C2-repin above.** Kept for provenance.
> Produced by **C2b — apply-chain cleanup** (Mac, no engine build). Reconciles the
> overlay patch chain to the **authoritative serving build** that actually passed
> the v0.8.0 real serve gate, then proves it git-apply --check clean from base.
>
> **This supersedes the C2a state**, which mistakenly designated the fork-port TTS
> streaming-worker stack as the canonical TTS source. Ground truth (below) shows
> the shipped/serve-gated build uses the GENERIC `qwen3_tts_worker` on
> `Qwen3OmniTTSRuntime` and does **not** build the fork-port streaming worker,
> the slot-pool, or any of the v080-NNNN ASR-streaming / TTS-batch experiments.

## 0. Ground truth — the authoritative build set

Source of truth = the feat tree `~/project/edgellm-v080-migration`:

| evidence | says |
|---|---|
| `docs/plans/v080-0017-hf-artifacts-manifest.md` (header) | **"edgellm base: `f9cc746` (Merge #101 dev-release/0.8.0) + port patches v080-0007 / v080-0008."** |
| `docs/plans/v080-0016-server-wiring-workers-acceptance.md` | 3 serving workers built; **TTS = `qwen3_tts_worker` (generic LLMEngineRunner, Qwen3OmniTTSRuntime struct port)**, ASR = `qwen3_asr_worker` one-shot, MOSS = `moss_tts_nano_worker` (no engine change). Worker `.cpp` live in `native/edgellm_voice_worker/` + `addon/cpp/workers/` — **compiled, not in the patch chain**. |
| `docs/plans/v080-0019-serve-gate-acceptance.md` | v0.8.0 image (thin binary overlay) **transcribes + synthesizes e2e**; ASR `stream_mode=accumulate` (one-shot path); streaming `begin/chunk/end` is a **501 stub** (no streaming worker). |
| `deploy/docker/Dockerfile.jetson.v080-edgellm` | THIN overlay: `COPY`s pre-built binaries `qwen3_asr_worker`/`qwen3_tts_worker`/`moss_tts_nano_worker` + `libNvInfer_edgellm_plugin.so`; profile `jetson-edgellm-v080` **omits** all explicit-KV / highperf-streaming TTS keys (worker rejects `--qwen3TtsTalkerEngine`/`--max_slots`). |

### Shipped binary md5 (Dockerfile + HF MANIFEST v080-0017)
```
qwen3_asr_worker             be7bee91728c63253e5926a5933896c0
qwen3_tts_worker             22216e8dc724bd8619d4fca26b0c2d5b
moss_tts_nano_worker         6a03bdf5c7a26b09f60597b95008ebfe
libNvInfer_edgellm_plugin.so 8f004bb4c9ddcce30ae4eecf2f410624
```
Worker `.cpp` sources (parent repo `native/edgellm_voice_worker/`, compiled — NOT
overlay patches): `qwen3_asr_worker.cpp` = `ab09b992`, `qwen3_tts_worker.cpp` =
`2e2de126` (matches the C2a §6 inventory).

## 1. UPSTREAM_PIN
- `f9cc74623d95d7acf1addab6026b9d410ba81f52` (NVIDIA `origin/release/0.8.0` HEAD,
  "Merge #101 from JCalafato/dev-release/0.8.0"). Unchanged from C2a.

## 2. AUTHORITATIVE APPLY ORDER (what build.sh applies)

build.sh: clone `f9cc746` → copy `addon/` (new files) → apply, in order:

| # | patch | result on base+addon | disposition |
|---|---|---|---|
| 1 | `0001-orin-tegra-build-compat` | **CHECK-CLEAN** | **keep** — Orin/Tegra CUDA-12.6 build-host compat (CMakeLists / CuteDsl.cmake / cpp/CMakeLists). |
| 2 | `v080-0007-customvoice-language-conditioning` | **CHECK-CLEAN** | **keep** — Qwen3OmniTTSRuntime struct port + talkerMLP kernels + omni CMake + export.py. The canonical TTS source. |
| 3 | `v080-0008-tts-cutedsl-wrap` | **CHECK-CLEAN** (after corruption fix, §3) | **keep / rebased** — load-bearing CuTe-DSL `--wrap=_cudaLaunchKernelEx` shim for the statically-linked exe on CUDA <12.8. Incremental on v080-0007. |

**Full chain `git apply --check` on `f9cc746` + `addon/`: 0 conflicts** (EVIDENCE
in the C2b report). MOSS is delivered entirely by `addon/` (new files; the engine
globs them into `edgellmCore`; no patch needed — see §4).

Patch file md5 (post-cleanup):
- `0001-orin-tegra-build-compat.patch` = `01a41ce1`
- `v080-0007-customvoice-language-conditioning.patch` = `74b31e98`
- `v080-0008-tts-cutedsl-wrap.patch` = `67336efb` (was corrupt; regenerated header+context, §3)

## 3. Fix applied this round — corrupt `v080-0008`
`git apply` reported "corrupt patch at line 53". Two defects in the single
`examples/omni/CMakeLists.txt` hunk:
1. The blank **context** line between `endif()` and `# Set output directory` had
   lost its leading space (was empty `""`, must be `" "`).
2. The hunk header undercounted the new side: the added comment block is **14**
   lines, so the header must be `@@ -17,6 +17,20 @@` (was `+17,19`).
Both fixed → applies CHECK-CLEAN on `f9cc746 + v080-0007`. No content change to
the actual CMake addition (same `-Wl,--wrap` block, same proven effect).

## 4. DROPPED from the serving chain (with reason)

| patch(es) | why dropped |
|---|---|
| `0002`-`0007` legacy theme, `0008` legacy example-registration | superseded on v0.8.0. `0002` (weight-streaming) references deleted `eagleDraftEngineRunner/llmEngineRunner*`; `0003`/`0006` reference rewritten/removed files; legacy `0008` registers the v0.7.1 streaming-worker omni targets that the serving build does not compile. **= P2/P6 decision, see §6.** Legacy `0005`/`0007` already re-authored as `v080-0007` / docs. |
| `v080-port-0001..0006` (fork-port TTS-runtime stack) | **NOT in the authoritative build.** These build `qwen3_tts_streaming_worker` + slot-pool (N>1) on the fork's `port/qwen3-tts-base-v080`. The serve-gated image ships the GENERIC `qwen3_tts_worker` (Qwen3OmniTTSRuntime, N=1). This was C2a's mis-designation; reversed. Kept on disk for the (future) N>1 streaming track. |
| `v080-0001/0002/0003/0004/0005/0006/0010/0012` | ASR-streaming + TTS-batch-lane **experiments** (audioRunner hooks, asrContinuousBatcher, hybridCacheManager per-lane KV, streaming decode hook, batch-lane maxBatchSize=2). The serve-gated ASR is one-shot (`stream_mode=accumulate`; streaming = 501 stub); TTS is N=1 generate-then-chunk. None compiled into the shipped binaries. **Deferred / experimental.** |
| `v080-0011-moss-tts-nano-port` | **MOSS delivered via `addon/`** (`cpp/runtime/mossTtsNanoRuntime.{cpp,h}`, `cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.{cu,h}`, `cpp/workers/{moss_tts_nano_worker.cpp,build_moss_worker.sh}`). The engine globs the new files into `edgellmCore`; `build_moss_worker.sh` links the resulting `.o`s. The only patch-only hunk is a `BUILD_UNIT_TESTS`-guarded `mossTtsNanoSmokeMain.cpp` exclusion — irrelevant to the serving build (unit tests not built). Single-sourced to addon → patch dropped. |

## 5. Moved to C3 backlog (NOT in any apply order; files kept)
`v080-0024-asr-incremental-kv-streaming`, `v080-0025-prefix-cap-BROKEN`,
`v080-0026-prefix-rollback-WORKS`, `v080-0027-prefix-rollback-byteexact-final`.
These are ASR-streaming experiment snapshots; `0025` is explicitly `-BROKEN`, and
`0025/0026/0027` target a non-git path literally named
`qwen3_asr_worker.cpp(+prefix-...)`. They are **not git-applicable** and the ASR
worker is not in the patch tree. → **C3 (ASR worker → fork single source, N≤0
guard).** Files left in place; just out of the build apply list.

## 6. P2 / P6 decision (per Dockerfile / serve-gate ground truth)
- **P2 (weight-streaming-budget, legacy `0002`):** **DROP.** Not in the
  authoritative build; the GDN VRAM fit was handled at engine-export time, not via
  this patch. Patch FAILs on v0.8.0 (missing `eagleDraftEngineRunner`/
  `llmEngineRunner*`). The feat tree did not apply it for the serving build.
- **P6 (server SSE + OpenAI API, legacy `0006`/`0007`):** **DROP from this engine
  overlay.** The v0.8.0 in-tree server was rewritten (`0006` FAILs at
  `api_server.py:33`). The shipped stack reaches the LLM via a **separate**
  `edge-llm-chat-service` (URL swap only — `v080-0016` §"deploy_paths"); there is
  no patched in-tree HTTP server, and tool-call / shared-engine-ctor logic lives
  in that separate service, not in this voice-worker engine overlay. So neither
  the SSE-disconnect fix nor the tool-call additions belong in this chain. (If a
  future track builds the in-tree server, P6 must be rebased onto the v0.8.0
  server rewrite — out of C2b scope.)

## 7. Native / addon worker md5 inventory (cross-check, unchanged)
| file | md5 | source |
|---|---|---|
| `native/edgellm_voice_worker/qwen3_asr_worker.cpp` | `ab09b992` | feat `native/` (compiled) ✓ |
| `native/edgellm_voice_worker/qwen3_tts_worker.cpp` | `2e2de126` | feat `native/` (compiled) ✓ |
| `addon/cpp/workers/moss_tts_nano_worker.cpp` | `71e4c83d` | addon (compiled via build_moss_worker.sh) |
| `addon/cpp/runtime/mossTtsNanoRuntime.cpp` | `ed98b2a6` | addon |
| `deploy/docker/worker_io.voxedge-patch.py` | `a8ff1b7c` | feat (Blocker-3 voxedge overlay) ✓ |

Shipped binary md5 (built from chain §2 + workers above) — §0.

## 9. N>1 ASR streaming — formalized into overlay + n2 profile (THIS ROUND)

**Goal:** make the real-machine-verified N>1 ASR streaming path (v080-0021 / 0022 /
0023, orin-nx PASS) reproducible from `build.sh` + an out-of-the-box N=2 profile —
WITHOUT building engines/images or touching devices.

### 9.1 Decisive finding — N>1 needs ZERO additional engine patches
The recon premise was that N>1 requires re-adding runtime patches
`v080-0001..0006` to the apply chain. **Ground truth from the acceptance docs
disproves this:**
- The verified worker (`native/edgellm_voice_worker/qwen3_asr_worker.cpp`) runs
  **`runOneShotCore` on the VANILLA `rt::LLMInferenceRuntime`** for both the N>1
  lane path and per-hop streaming (worker.cpp:17 *"v0.8.0 vanilla rt::LLMInferenceRuntime
  on the ONE-SHOT path only"*; :304 *"lane reservation only, NOT
  AsrStreamingSessionRuntime KV-append"*; :435-440 the `appendChunk/decodeToTranscript`
  incremental path is the **DEFERRED** optimization, NOT called).
- v080-0021 §1 + v080-0023 §1 confirm: N>1 = lane reservation pool + per-hop
  **cumulative re-decode via the proven one-shot core**. No engine-source change.
- The `v080-0001..0006` patches build `examples/llm/spike_v080_m1..m6` experiment
  binaries + `asrStreamingSessionRuntime` / `asrContinuousBatcher` /
  `hybridCacheManager` per-lane KV = the **incremental-KV spike track** (the
  deferred optimization), irrelevant to serving.

**git-apply --check verdict (temp worktree @ f9cc746 + addon/, minimal chain
0001+v080-0007+v080-0008 applied):**
| patch | check | note |
|---|---|---|
| v080-0001-asr-audio-chunk-api | CLEAN | spike-track, not needed |
| v080-0002-asr-streaming-runtime-hooks | CLEAN | spike-track, not needed |
| v080-0003-asr-streaming-spikes | **FAIL** | `examples/llm/spike_v080_m1/m2.cpp` already exist in working dir (collide with addon spikes) |
| v080-0004-asr-single-token-chunk-guard | **FAIL** | needs `cpp/runtime/asrStreamingSessionRuntime.{h,cpp}` (created only by 0002) — order-fragile |
| v080-0005-per-lane-kv-reset-and-lane-manager | **FAIL** | missing asrStreamingSessionRuntime + `llmInferenceRuntime.{h,cpp}` / `examples/llm/CMakeLists.txt` offsets do not apply |
| v080-0006-asr-continuous-batcher | CLEAN | spike-track, not needed |

→ The spike track is **NOT clean-applicable** standalone AND **not used** by the
verified path. **DROPPED from the N>1 scope** (stay in C3 backlog, §5). The
**minimal serving chain is the complete engine basis for N>1** — full-chain
`git apply --check` CLEAN, unchanged from §2.

### 9.2 What WAS formalized this round
1. **`build.sh` voice-worker stage (NEW, step 4b):** wired the previously
   hand-built workers (v080-0021 "Box build dir `~/project/v080-worker-build`")
   into the reproduction flow. After the engine core builds `libedgellmCore.a`,
   `build.sh` now runs the `native/edgellm_voice_worker/` CMake project
   (`-DEDGE_LLM_SOURCE_DIR=${WORKDIR} -DEDGE_LLM_BUILD_DIR=${WORKDIR}/build`),
   building targets `qwen3_asr_worker` (N>1) + `qwen3_tts_worker`. MOSS unchanged
   (step 4c).
2. **n2 profile session-gate triplet (was MISSING → N=2 was NOT open):** the
   profile shipped `OVS_TTS_WORKER_CONCURRENCY=1`, so `effective_limit=min(asr=2,
   tts=1)=1` clamped admission to 1. Added the v080-0023 §"Session-gate change"
   triplet — `LAZY_TTS=1` + `OVS_TTS_WORKER_CONCURRENCY=2` +
   `OVS_MAX_CONCURRENT_SESSIONS=2` — so N=2 is OPEN OUT OF THE BOX (TTS never
   spawned with `--max_slots`, so the generic-runner worker never aborts). Also
   added `EDGE_LLM_ASR_ENGINE_DIR→engines/asr-b2/llm` + `OVS_ASR_STREAM_PREFIX=1`
   + `OVS_ASR_STREAM_PREFIX_FINAL_ONESHOT=1`.

### 9.3 n2 profile formalization — LOCATION
The profile lives in **this repo (jetson-voice-engine)**, NOT seeed-local-voice:
- `deploy/docker/jetson-edgellm-v080-n2.json` — the source-of-truth profile
  (env + asr_max_slots=2), updated with the triplet + engine_dir + prefix.
- `deploy/docker/Dockerfile.jetson.v080-edgellm-n2` — bakes the profile +
  operator-owned ENV (triplet now baked as image ENV so it wins at runtime), and
  the worker md5 comments corrected `f9bb821d` → `5ebd436b` (the pcm-fixed binary).
- seeed-local-voice `configs/profiles/` was checked — it does NOT carry an n2
  profile; **no new branch needed there.** (Its v080 profiles are baked from this
  repo's deploy/docker via the thin-overlay Dockerfile.)

### 9.4 Final apply order (minimal + N>1) — UNCHANGED engine chain
N>1 adds NO engine patch. The apply order is exactly §2:
`0001-orin-tegra-build-compat` → `v080-0007-customvoice-language-conditioning` →
`v080-0008-tts-cutedsl-wrap`. N>1 is delivered by the build.sh voice-worker stage
(§9.2.1) + the n2 profile (§9.3) + the asr-b2 engine export (md5 4122dfcc, an
artifact, not a patch).

### 9.5 Engine / worker md5 reconciliation (N>1)
| artifact | md5 | source | role |
|---|---|---|---|
| `native/edgellm_voice_worker/qwen3_asr_worker.cpp` | `ab09b992` | feat HEAD = `c6bf483` (v080-0023 pcm fix) | N>1 worker SOURCE |
| qwen3_asr_worker binary | `5ebd436b` | built from the above on orin-nx (v080-0023 §1) | N>1 worker BINARY |
| `engines/asr-b2/llm/llm.engine` (max_batch_size=2) | `4122dfcc` | export artifact (v080-0022 §1) | N>1 ASR engine |
| `engines/asr-b2/llm/config.json` (`max_batch_size:2`) | `0e6bb1f6` | export artifact | N>1 engine config |
| audio encoder `engines/asr/audio/config.json` (min_time_steps=100) | `3b9ff631` | minchunk1 export (v080-0019) | streaming encoder |

(The Dockerfile/profile previously referenced the SUPERSEDED `f9bb821d` worker =
the pre-pcm-fix v080-0022 binary that SIGABRT'd on real `pcm_b64`; corrected to
`5ebd436b` throughout this round.)

## 8. addon/ note
C2a already removed the v0.7.1-vendored `addon/examples/omni/qwen3_tts_streaming_worker.cpp`
and `addon/cpp/runtime/slotPool.h` (both belonged to the dropped fork-port path).
The remaining addon files (MOSS, w8a16 kernels, statefulCode2Wav, spikes, scripts,
experimental/server) are all **additive new files** — they do not conflict with
the apply chain and matched the v080-0016 build set.

## 10. CustomVoice variant — DEFINED, NOT BUILT HERE (future first-class variant)

> **⚠️ SUPERSEDED 2026-06-22 by §12.** This section described a *separate-binary*
> CV variant (own apply chain `0001` + `v080-0007` + `v080-0008`, a CV-only binary
> that could not coexist with Base). That plan is **obsolete.** CustomVoice is now
> folded INTO the pinned branch (`c48c0de`) as a **runtime-if on `langId`** — ONE
> binary handles BOTH Base (langId<0, 8-row) and CV (langId>=0, 9-row). The old CV
> patches (`v080-0007` / `0005` / `v080-0008`) are archival-only and NOT applied.
> Read §12 for the current state; the text below is retained for history only.

Per the 2026-06-21 decision, **CustomVoice is an independent first-class build
variant**, parallel to the Base N>1 variant this overlay builds. It is **NOT built
in this overlay / build.sh run.** Definition for the future variant:

- **Same N>1 machinery** as Base: the slot-pool + shared-engine ctor + streaming
  worker (`examples/omni/qwen3_tts_streaming_worker`) from the pinned fork branch
  (a361221). N>1 ASR/TTS concurrency, session-gate triplet, n2-style profile —
  all reused unchanged.
- **CustomVoice talker conditioning patch:** `v080-0007-customvoice-language-conditioning`
  (9-row langId prefill + talkerMLP kernel layout) — **applied for the CV variant
  only.** It conflicts with the Base 8-row speaker-encoder talker prefill, so the
  two variants are **separate binaries** (cannot coexist).
- **Companion toolchain patch:** `v080-0008-tts-cutedsl-wrap` (incremental on
  v080-0007's shim block) — re-home it to the CV variant too, since it only
  applies on top of v080-0007 and is unnecessary for Base.
- **CustomVoice int4 engine** (NOT an export the Base variant uses):
  `harvestsu/qwen3-tts-0.6b-customvoice-jetson-trtllm-int4fp8` (int4/fp8 talker;
  the verified CV engine). Base uses talker + code-predictor + speaker-encoder
  (external embedding) instead.
- **Separate build:** the CV variant would have its own apply chain
  (`0001` + `v080-0007` + `v080-0008`) on the pinned branch, its own engine bundle
  (CV int4), and its own profile/Dockerfile. NOT produced by this build.sh.
- **Known-good reference:** the CV int4 talker is verified usable end-to-end
  (ASR confirms intelligible audio) — see project memory
  `customvoice_talker_int4_eos_fail_2026_06_20.md`. The earlier "garbled/runaway"
  saga was a wrong-tokenizer test-harness bug, not an int4/engine defect.

This overlay's `UPSTREAM_PIN` + apply chain + build.sh stay **Base-only**. The CV
variant is a follow-up task (own pin chain may equal a361221 too, just with the
two CV patches applied + the CV int4 engine bundle).

---

## 11. Coexistence disambiguation — easily-confused parallel artifacts (2026-06-22)

Several look-alike artifacts coexist on disk / in HF and have repeatedly been
mistaken for each other. This section is the single authoritative "which is
production, which is alternative/deferred" map. **None of the deferred items
below are dead — they are validated artifacts kept for the alternative track;
do not delete. This is annotation only.**

### 11.1 Dual N>1 TTS worker — slot-pool (PRODUCTION) vs batch-lane (alt/deferred)
| path | what it is | status |
|---|---|---|
| **slot-pool** | `examples/omni/qwen3_tts_streaming_worker` + `cpp/runtime/slotPool.h` + shared-engine ctor (`Qwen3OmniTTSRuntime(ICudaEngine*, ICudaEngine*, …)` + `initializeEngineRunnersShared`). One **runtime per lane** over **shared read-only engine weights**; uses the ordinary `maxBatchSize=1` Talker. In the pinned fork branch `a361221`; built by `build.sh` step 4. | **PRODUCTION N>1 TTS path.** |
| **batch-lane** | patch `v080-0010-tts-batch-lane-concurrency` — rewrites `Qwen3OmniTTSRuntime` so a SINGLE runtime with a `maxBatchSize=2` Talker (the `talker-b2` engine, md5 `f7339e02`) natively batches two lanes (replacing the per-slot replication that patch-0004 used). Lockstep batched prefill/decode. | **ALTERNATIVE / DEFERRED — NOT a production path.** Not in the Base apply chain (see §4 line for `…/0010/…` and §C2-repin apply table). Build chain does **not** apply it. Validated GREEN (byte-identical per-lane audio) and kept for the alt track. |

Rule of thumb: production N>1 TTS = **replicate runtime, share weights** (slot-pool).
batch-lane = **one runtime, batched Talker engine** (the b2 engine). The
`recipes/talker-b2-engine.md` formerly called itself "slot-pool" — corrected
2026-06-22 to "batch-lane".

### 11.2 Dual TTS Talker engine — int4 (CV, prod) vs talker-b2 (batch-lane)
| engine | role | consumed by |
|---|---|---|
| **CustomVoice int4 talker** (`harvestsu/qwen3-tts-0.6b-customvoice-jetson-trtllm-int4fp8`) | the verified production CustomVoice talker (int4/fp8). | the CustomVoice variant (§10). Production CV path. |
| **talker-b2** (`engines-v080-tts-b2/talker/llm.engine`, md5 `f7339e02`) | `maxBatchSize=2` Talker, build-flag-only rebuild of the maxBatch=1 Talker ONNX. | the **batch-lane** runtime (`v080-0010`) ONLY — see §11.1. NOT the slot-pool, NOT the CV int4 path. |

(The Base N>1 slot-pool path uses neither of these — it uses the ordinary Base
maxBatch=1 talker + code-predictor + speaker-encoder; see §C2-repin "build.sh".)

### 11.3 Incremental-KV spike track — confirmed NON-serving (cross-ref §5 / §9.1)
Patch files `v080-0001..0006` (ASR-streaming runtime hooks / spikes / lane mgr /
continuous batcher) **and** `v080-0024..0027` (incremental-KV + prefix-cap /
rollback snapshots) are the **DEFERRED incremental-KV spike track**, NOT in any
serving build, NOT applied by `build.sh`. Already established in §5 (C3 backlog)
and §9.1 (decisive finding: N>1 needs ZERO additional engine patches; the spike
binaries are `examples/llm/spike_v080_m1..m6` experiments). Filename markers
confirm their non-serving nature: `v080-0025-prefix-cap-BROKEN.patch` is
explicitly `-BROKEN`, and `0025/0026/0027` target a non-git working-copy path
literally named `qwen3_asr_worker.cpp(+prefix-…)` / `(#12-state)` — they are
**not git-applicable** and must stay OUT of the apply sequence. The build chain
must never reference them.

### 11.4 prefix-multiturn image — EXPERIMENT, not the v0.8.0 n1n2-rebake production image
Any `prefix-multiturn` image depends on an **unreleased `asr-b2` dependency
chain** and the `OVS_ASR_STREAM_PREFIX*` flags wired only for the streaming
spike. It is an **experiment**, NOT the `v0.8.0-n1n2-rebake` production image.
The production N>1 image is built from `deploy/docker/Dockerfile.jetson.v080-edgellm-n2`
+ `jetson-edgellm-v080-n2.json` (see §9.3) on the **vanilla one-shot** ASR worker
+ asr-b2 engine artifact — no incremental-prefix runtime. Do not promote a
`prefix-multiturn` build to production.

---

## 12. CustomVoice re-pin — ONE binary via runtime-if on langId (CURRENT, 2026-06-22)

**Supersedes §10's "separate-binary CV variant" plan.** CustomVoice is no longer a
separate binary with its own apply chain — it is **folded into the pinned branch**
and selected at runtime by `langId`.

### Pin (supersedes §C2-repin §27 pin too)
```
UPSTREAM_PIN = c48c0de505f769e8779035ba912764c3a399b7c2
             = suharvest/wip/cv-9row-v080-n1n2 (fork integration branch HEAD)
```
Re-pinned **7142a30 → c48c0de** (Option A — full re-pin so `build.sh` reproduces the
CustomVoice worker FROM SOURCE). The branch is the previous Base N>1 chain
(`…a361221…7142a30`) **plus 2 CV commits folded in:**

| sha | what |
|---|---|
| `12ee383` | port 9-row CustomVoice language conditioning onto v0.8.0 N>1 — adds the 9-row CV talker prefill behind a **RUNTIME-IF on `langId`**. ONE binary now handles BOTH: `langId < 0` → Base 8-row speaker-encoder prefill (UNCHANGED); `langId >= 0` → CustomVoice 9-row langId prefill. |
| `c48c0de` | wire per-request `language` into the streaming worker `buildRequest` (per-request langId reaches the runtime-if path). ← branch HEAD |

### Apply chain — UNCHANGED (still ONE patch)
Because the 2 CV commits are now **IN** the pinned branch, the `build.sh` apply
chain stays exactly:
```
0001-orin-tegra-build-compat        (Tegra/Orin sm_87 build compat)
```
No CV patch is applied. The runtime-if comes from the branch, not a patch.

### Superseded → archival-only (NOT in any apply chain; files kept on disk)
The old pre-runtime-if CV patches are **superseded** by the in-branch runtime-if
(`12ee383`/`c48c0de`) and are intentionally **NOT applied**:
- `patches/v080-0007-customvoice-language-conditioning.patch`
- `patches/0005-customvoice-language-conditioning.patch`
- `patches/v080-0008-tts-cutedsl-wrap.patch` (companion shim that only sat on top of v080-0007)

They forced a SEPARATE CV-only binary that could not coexist with Base — exactly
the limitation the runtime-if removes. Kept on disk for archival/history only.

### Rebuild + gate verification (2026-06-22, orin-nx build → orin-nano gate)
- **Built on orin-nx** via `engine-overlay/build.sh manifests/qwen3-tts-highperf-sm87.toml`
  (fetched pin `c48c0de` from the suharvest TensorRT-Edge-LLM fork, applied `0001`,
  compiled sm_87 / CUDA 12.6 / TRT 10.3, Release). Built worker
  `examples/omni/qwen3_tts_streaming_worker` md5 `50d586de5b869192230f6a31ca72ac77`
  (plugin `libNvInfer_edgellm_plugin.so.1.0` md5 `e723ffc7186bf8f62ec68b2fb20fbafa`).
  A fresh orin-nx build legitimately has a **different md5** than the orin-nano
  reference (`6784031688289953699c3e89f324dd71`) — **functional equivalence**, not
  md5, is the criterion (c2-repin-spec §5).
- **Gated on orin-nano** (faster-whisper `small`, `cv_drive.py` JSONL driver):

  | request | worker log | audio frames | faster-whisper transcript |
  |---|---|---|---|
  | CV zh (`language=chinese`) | `langId=2055`, `prefixRows=9` | 46 (exit: EOS), ok=true | "你好,很高兴见到你,今天天气怎么样?" (lang zh 0.996) |
  | CV en (`language=english`) | `langId=2050`, `prefixRows=9` | 67 (exit: EOS), ok=true | "Hello, nice to meet you. How is the weather today?" (lang en 0.779) |
  | Base (no language) | `langId=-1`, `prefixRows=8` | 55 (exit: EOS), ok=true | intelligible zh (0.995) |

  Base 55-frame + CV-en 67-frame outputs match the validated reference exactly →
  ONE binary correctly routes Base (8-row) and CV (9-row) by langId, with no Base
  regression. CV gate **PASSED.**

This pin + the runtime-if is the SINGLE SOURCE for the v0.8.0 N>1 stack — Base AND
CustomVoice in ONE binary. See `UPSTREAM_PIN` header for the full commit ladder.
