# Patch state after v0.8.0 re-pin (C2a)

> Produced by **C2a — source-level re-pin** (Mac, no engine build). Pin moved
> v0.7.1 `364769…` → v0.8.0 `f9cc746…` (`origin/release/0.8.0` HEAD). All claims
> below were verified with `git apply --check` against a worktree of the fork at
> `f9cc746` plus `addon/` copied on top (build.sh step-2 order).
>
> Scope note: C2a re-pins + regenerates the **fork-port** TTS-runtime stack and
> swaps the streaming worker. The **v080-NNNN feat patch series** (ASR streaming,
> MOSS, CustomVoice, TTS-batch) is pre-existing `feat/edgellm-v080-migration`
> debt; reconciling it into a clean from-base apply chain is **deferred to a
> follow-up rebase** (out of C2a scope). C2b (build-level md5 reconciliation) and
> C3 (ASR worker → fork single source, N≤0 guard) follow.

## 1. UPSTREAM_PIN
- **Now:** `f9cc74623d95d7acf1addab6026b9d410ba81f52` (NVIDIA `origin/release/0.8.0`
  HEAD, "Merge #101 from JCalafato/dev-release/0.8.0").
- fork port branch on this base: `port/qwen3-tts-base-v080` = 6 commits / 9 files.

## 2. Legacy 000N theme patches vs v0.8.0 base (apply-check evidence)

| patch | result on v0.8.0 base | disposition |
|---|---|---|
| 0001 orin-tegra-build-compat | **CLEAN** | keep, applied by build.sh |
| 0002 weight-streaming-budget | FAIL `builderUtils.cpp:266` + missing `eagleDraftEngineRunner/llmEngineRunner*` | superseded; needs rebase if still wanted |
| 0003 asr-streaming-session | FAIL `audioRunner.cpp:511` + missing `llmInferenceSpecDecodeRuntime*` | superseded by v080-NNNN ASR + fork port |
| 0004 tts-slotpool-concurrency | FAIL `qwen3OmniTTSRuntime.cpp:1059` | superseded by fork-port streaming worker (v080-port-0001) |
| 0005 customvoice-9row | FAIL — `experimental/llm_loader/export_all_cli.py` gone in v0.8.0 | superseded by v080-0007 (feat) |
| 0006 server-sse + openai-api | FAIL `api_server.py:33` / `engine.py:74` | superseded (server rewritten); rebase if wanted |
| 0007 server docs | FAIL docs path | bound to 0006 |
| 0008 build-misc-example-registration | **CLEAN in isolation**, but its `examples/omni/CMakeLists.txt` hunk DUPLICATES the streaming-worker target now registered by v080-port-0001 → conflicts in-chain | **hunk-split needed** (drop omni hunk, keep `.gitignore` + `examples/llm/CMakeLists.txt`) |

This matches the C2 spec §2 disposition table exactly (P1/P8 clean, P2/P6 rebase,
P3/P4/P5/P7 superseded).

## 3. Fork-port stack (canonical TTS runtime) — `v080-port-0001..0006`
Regenerated via `git format-patch f9cc746..port/qwen3-tts-base-v080`.

**Verified CLEAN as a stack** on `f9cc746` + `addon/` (after addon de-dup, §5):

```
OK 0001  port streaming worker (base, N>1 slot-pool)   -> creates worker + slotPool.h + omni CMakeLists
OK 0002  backport Base speaker-encoder (external-embed)
OK 0003  link cutedsl cudart shim into omni exes
OK 0004  cuBLAS-free tiled FP16 GEMM fallback (sm_87)
OK 0005  warp-per-column M=1 GEMV (talker hot path)
OK 0006  fp8 text_embedding (native kernel path)
```

Resulting file md5 after the stack:
- `examples/omni/qwen3_tts_streaming_worker.cpp` = `fe7c0736` ✓ (spec target; replaces stale `0038bbd6`)
- `cpp/runtime/slotPool.h` = `fe8263a7` ✓ (identical to the removed addon copy)
- `scripts/quantize_text_embedding_fp8.py` = `e220fce3` (created by v080-port-0006)

## 4. v080-NNNN feat patch series — PENDING REBASE (not applied by build.sh yet)
Sequential apply against `f9cc746` + addon does **not** complete. Failure classes:

1. **Double-source vs addon/** — these files are shipped BOTH by `addon/` copy
   AND created as new-file by a patch → "already exists in working directory":
   - `examples/llm/spike_v080_m1_append_prefill.cpp`, `spike_v080_m2_session_lifecycle.cpp` (v080-0003)
   - `cpp/runtime/mossTtsNanoRuntime.{cpp,h}`, `cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.{cu,h}`,
     `cpp/workers/moss_tts_nano_worker.cpp`, `cpp/workers/build_moss_worker.sh`,
     `unittests/mossLinearKvKernelsTests.cu`, `unittests/mossTtsNanoSmokeMain.cpp` (v080-0011 MOSS)
   → decide per file: deliver via addon OR via patch, not both.
2. **Sequential-offset failures** — v080-0010 (`hybridCacheManager.cpp:269`),
   v080-0012 (`asrStreamingSessionRuntime.h`), v080-0007 (`talkerMLPKernels.cu:338`):
   these expect a tree state that includes the fork-port changes and/or earlier
   v080 hunks → must be rebased onto `f9cc746` + fork-port stack.
3. **Corrupt patch** — `v080-0008-tts-cutedsl-wrap.patch` "corrupt patch at
   line 53" (blank context line lacks the leading space) → regenerate.
4. **Non-applicable diff snapshots** — `v080-0024/0025/0026/0027` target a path
   literally named `qwen3_asr_worker.cpp(#12-state)` / `(+prefix-cap-BROKEN)` etc.
   These are ASR-streaming experiment records (one explicitly `-BROKEN`), NOT
   git-applicable patches; the ASR worker is not even in the patch tree. Defer to
   **C3** (ASR worker → fork single source); they do not belong in the build chain.

## 5. addon/ reconciliation (this change set)
- **Removed** `addon/examples/omni/qwen3_tts_streaming_worker.cpp` (was md5
  `0038bbd6`, a v0.7.1 copy). Now delivered by v080-port-0001 at `fe7c0736`.
- **Removed** `addon/cpp/runtime/slotPool.h` (md5 `fe8263a7`). Now delivered by
  v080-port-0001 at the identical md5.
- MOSS / spike addon files left in place (their patch-vs-addon double-source is a
  v080-NNNN rebase decision, §4.1 — not C2a's to resolve).

## 6. Native / addon worker md5 inventory (spec §4 cross-check)
| file | md5 | source |
|---|---|---|
| `native/edgellm_voice_worker/qwen3_asr_worker.cpp` | `ab09b992` | feat (spec §4) ✓ |
| `native/edgellm_voice_worker/qwen3_tts_worker.cpp` | `2e2de126` | feat (spec §4) ✓ |
| `deploy/docker/worker_io.voxedge-patch.py` | `a8ff1b7c` | feat (spec §4) ✓ |
| `examples/omni/qwen3_tts_streaming_worker.cpp` (built) | `fe7c0736` | fork port (replaces `0038bbd6`) ✓ |
| `cpp/runtime/slotPool.h` (built) | `fe8263a7` | fork port ✓ |
