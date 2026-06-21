# Patch state — v0.8.0 authoritative serving chain (C2b)

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

## 8. addon/ note
C2a already removed the v0.7.1-vendored `addon/examples/omni/qwen3_tts_streaming_worker.cpp`
and `addon/cpp/runtime/slotPool.h` (both belonged to the dropped fork-port path).
The remaining addon files (MOSS, w8a16 kernels, statefulCode2Wav, spikes, scripts,
experimental/server) are all **additive new files** — they do not conflict with
the apply chain and matched the v080-0016 build set.
