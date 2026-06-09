# Phase 1 — v0.8.0 ASR-streaming spike port notes

Status: Phase 1 scaffolding output (codex research, main-thread verified against `v0.8.0` tag 2026-06-09). The compilable-spike goal is largely deferred: every ASR session call depends on Phase 2/3 runtime hooks that do not yet exist in v0.8.0. Phase 1's real deliverable is this **confirmed API mapping + the Phase 2/3 hook work-list** below.

## Spec Section 1 line-numbers — verified against the real v0.8.0 tree
All CONFIRMED except one correction:
- ⚠️ **Correction:** `LLMInferenceRuntime` *class* starts at `cpp/runtime/llmInferenceRuntime.h:59` (spec cited `:110`, which is actually the `handleRequest` *declaration* — still correct as the handleRequest anchor).
- `handleRequest` decl: `llmInferenceRuntime.h:110` ✓
- `DecodingInferenceContext`: `state/decodingInferenceContext.h:57-76` ✓; `::initialize` `.cpp:25-55` ✓
- `runBaseModelPrefill`: `llmInferenceRuntime.cpp:1070-1136` ✓; full accumulated-token copy `.cpp:1105-1108` ✓ (this is the R2 refactor seam)
- `setUpForPrefillExecution`: `.cpp:1296-1408` ✓; cache reset call `.cpp:1403` ✓ (clears tokenIds at :1332)
- `HybridCacheManager::resetForNewSequences`: `.h:162-166`, `.cpp:247-274` ✓; `commitSequenceLength`: `.h:168-176`, `.cpp:276-289` ✓
- `EngineExecutor::prepare/execute`: `.cpp:86-155`, `158-177` ✓
- `kvcache_start_index` binding: `registryBuilder.cpp:59-67`, `pipelineIO.cpp:185-190` ✓
- `prefillDims(batch, seqLen, kvCacheAllEmpty)`: `llmEngineConfig.h:156-162` ✓

## Compile-surface assessment
**API-complete in v0.8.0 today** (spikes can reference these as-is): `LLMInferenceRuntime` construction, `DecodingInferenceContext::initialize`, `HybridCacheManager` reset/commit, `EngineExecutor::prepare/execute`, `PipelineIO` bindings.

**Blocked on Phase 2/3** (do NOT exist in v0.8.0 headers — the spikes can only stub these): all ASR session-management calls.

## Phase 2/3 runtime-hook work-list (drives the next phases)
1. **Audio chunk embedding API + session MRope init** (Phase 2). ✅ **LANDED** — patch `engine-overlay/patches/v080-0001-asr-audio-chunk-api.patch`. Added `Qwen3OmniAudioRunner::encodeMelChunk` (public, copies encoder output into a caller-allocated FP16 tensor), `encodeOneAudioImpl` (private shared forward pass, extracted bit-exact from `preprocessAudio`'s loop body), and `initializeSequentialMRopeLane(lane, maxPositions, mropeCosSin, stream)` (single-lane, NO reshape) over `PipelineIO::mropeCosSin` (spec §5 R3; old `0003:298-329,338-381`). v0.7.1→v0.8.0 adaptation: the old `initializeMRopeForSession(rt::Tensor&, ..., activeBatchSize)` reshaped the whole cache via `initializeSequentialMRopeCache`; v0.8.0's MRope cache is the shared `PipelineIO::mropeCosSin` `[maxBatch, maxKVCacheCapacity, rotaryDim]` FP32 buffer, so the new lane variant writes only `lane`'s contiguous slab (`base + lane*maxPos*rotaryDim`) with `initializeMRopeCosSin(batchSize=1)` and never reshapes (verified against `initializeCosSinCache.cu:241,264` per-batch layout). Base-class no-op virtual `MultimodalRunner::initializeSequentialMRopeLane` added so the Phase 3 runtime can dispatch polymorphically. audioRunner TU compiles clean vs v0.8.0.
2. **`SessionLaneManager` + targeted `HybridCacheManager` lane reset** (Phase 3/4, spec R1). Add `resetLanes(span<int32_t>)`, `setLaneLength`, `maxBatchSize` next to `resetForNewSequences` (`hybridCacheManager.h:162-176`) — per-lane memset, NO reshape. ✅ **LANDED** — patch `v080-0005-per-lane-kv-reset-and-lane-manager.patch` (Phase 4, 2026-06-10). Added to `HybridCacheManager`: `resetLanes(std::vector<int32_t> const&, cudaStream_t)` (per-element `cudaMemsetAsync` on `mDeviceKVCacheLengths[batchIdx]`, NO reshape — the full-batch buffer is allocated at `maxBatchSize` capacity in the ctor and never freed, so element `[batchIdx]` is always addressable regardless of the current logical `activeBatchSize`); `setLaneLength(int32_t, int32_t, cudaStream_t)` (single-element H2D memcpy / memset); `maxBatchSize() const`. `mKVCacheAllEmpty` becomes conservatively `false` after a partial reset (other lanes may hold KV). **Existing full-batch `resetForNewSequences` is byte-unchanged** (still the one-shot reshape path). New `SessionLaneManager{vector<LaneRecord>{ownerKind,ownerId,slotId,inUse,kvLength}+mutex}` (in `asrStreamingSessionRuntime.{h,cpp}`) constructed with `maxBatchSize`+static partitions (ASR `[0,asrMax)`, TTS `[asrMax,asrMax+ttsMax)`); `acquire(LaneOwnerKind,ownerId)→laneId` (partition-scoped first-free, `-1` if full) / `release(laneId)`. `AsrStreamingSessionRuntime` gained a `(runtime, SessionLaneManager&, ownerId)` ctor: `beginAsrSession` acquires an ASR lane, `endAsrSession` now calls the new `LLMInferenceRuntime::resetSessionLane(lane)` → `HybridCacheManager::resetLanes({lane})` (**NOT** full-batch) + `release(lane)`. The old `(runtime, lane)` ctor keeps the Phase-3 full-batch `resetForSessionEnd` fallback (byte-unchanged, exercised by m2). The `// TODO(phase4): per-lane reset` markers in `LLMInferenceRuntime::resetForSessionEnd`/`.h` and `AsrStreamingSessionRuntime::endAsrSession` are resolved (full-batch reset retained only as the no-lane-manager fallback). **R1 lane-isolation GATE empirically green** — see Phase 4 acceptance below.
3. **`LLMInferenceRuntime::runBaseModelPrefillChunk(...)`** factored from `runBaseModelPrefill` (`.cpp:1070-1136`) so it packs ONLY the chunk slice, not all `context.tokenIds` (spec R2; the `:1105-1108` copy is the seam). One-shot path becomes a thin wrapper → byte-identical. ✅ **LANDED** — patch `v080-0002-asr-streaming-runtime-hooks.patch`. `runBaseModelPrefill(context)` now builds full-token `rt::SpanI32` views from `context.tokenIds` and calls `runBaseModelPrefillChunk(context, fullSpans, context.audioEmbeddings, sampleAfterPrefill=true)`; the helper reshapes IO to the CHUNK length, writes `hostCtxLen[i]=chunkTokenIds[i].size()`, packs only the chunk slice (replacing the `:1105-1108` full copy), and forwards explicit per-chunk audio embeds. Since after `setUpForPrefillExecution` `tokenIds[i].size() == effectivePrefillLengths[i]`, the wrapper's `inputIdsLength`/`hostCtxLen`/packed-ids/embed inputs are bit-for-bit the originals → one-shot byte-identical (code-inspection proof; no v0.8.0 ASR engine exists on the box, so empirical single-vs-split logit-diff is **Phase 3b**). `rt::SpanI32` added as a C++17 stand-in for `std::span<int32_t const>`.
4. **`AsrStreamingSessionRuntime`** begin/append/end wrapper around `LLMInferenceRuntime` (spec §5.2). NOT a DecodingStrategy. ✅ **LANDED** — new `cpp/runtime/asrStreamingSessionRuntime.{h,cpp}` (auto-picked by `RUNTIME_CPP_SRCS` GLOB). Composition wrapper owning a session `DecodingInferenceContext` + lane + MRope position cursor + reusable FP16 per-chunk audio scratch. `beginAsrSession` inits context (batch=1), runs `setUpAsrSessionPrefill` once (reuses the exact one-shot setUpForPrefillExecution via the registry's cache-priming/vanilla strategy), and `initAsrSessionMRopeLane` (Phase 2 single-lane MRope init, no shared-buffer reshape). `appendChunk` encodes the mel via `encodeAudioChunk` (Phase 2 `encodeMelChunk`) + runs `appendPrefillChunk` (NEVER re-runs setUpForPrefillExecution); `sampleAfterPrefill` only on the final chunk; carries an overflow guard against `maxPositions`. `endAsrSession` does the Phase-3 full-batch reset + clears session state. Hooks are `private` on `LLMInferenceRuntime` with `friend class AsrStreamingSessionRuntime`.
5. **Session introspection**: append status, `peekKvCacheLength`, session status query. ✅ **LANDED** (v080-0002). `LLMInferenceRuntime::peekKvCacheLength(lane, stream)` copies one element of the device KV-length tensor D2H; wrapper exposes `status()` (`kIdle`/`kActive`/`kFinished`), `appendedTokens()`, `chunkCount()`, `peekKvCacheLength()`.
6. **Testing surface**: `getLogitsForTesting` (so the spikes' R2 single-vs-split argmax/logit parity is observable). ✅ **LANDED** (v080-0002). `LLMInferenceRuntime::getLogitsForTesting(lane, stream)` returns the host FP32 logits row for the lane from `mPipelineIO->outputLogits`; wrapper forwards it. Enables the Phase-3b single-vs-split parity assertion once a v0.8.0 ASR engine exists.
7. **Sys-prompt mismatch fallback** BEFORE cached-KV restore in `setUpForPrefillExecution` (spec R4; move validation ahead of `:1339` restore, fresh-prefill on mismatch). ✅ **LANDED** (v080-0002). The cached-KV validation (`shapeOk` + `lengthOk` + exact `std::equal` token-prefix `matchIds`) now runs BEFORE `restoreKVCache`; only a full match installs base/strategy/recurrent KV + reuses the prefix. ANY mismatch falls through to a fresh prefill (`tokenIds=full input`, `reuse=0`, zeroed recurrent) — the cache-miss path — instead of v0.8.0's install-then-LOG_WARNING. Prevents a stale/altered tokenization (with MRope positions baked into cached KV) from silently corrupting the run.

## Phase 3 build/validation status (LANDED 2026-06-09)
- All five files compile clean into `edgellmCore` against v0.8.0 (orin-nx, Release, `cmake --build build --target edgellmCore`); 0 compile errors. New `asrStreamingSessionRuntime.cpp.o` produced; all hooks present as defined `T` symbols in `libedgellmCore.a`.
- Link proven: `llm_inference` + `audio_build` examples relink against the modified core with 0 errors.
- One-shot byte-identity: **code-inspection proof only** (see #3) — no v0.8.0 qwen3-asr TRT engine exists on the box (only Python modeling templates in `tensorrt_edgellm/models/qwen3_asr/`). The runtime R2 single-vs-split argmax/logit-parity test is **Phase 3b**, needing a built v0.8.0 ASR engine + the deferred spike drivers (`spike_v080_m1_append_prefill.cpp` / `spike_v080_m2_session_lifecycle.cpp`).
- Next dependency: **Phase 4** (R1 per-lane `HybridCacheManager` reset + `SessionLaneManager`) for concurrent-session/TTS-slot co-residency, and **Phase 3b** (engine-backed runtime validation).

## Spike files — LANDED + RUN (Phase 3b, 2026-06-09)
`spike_v080_m1_append_prefill.cpp` (R2/R3) and `spike_v080_m2_session_lifecycle.cpp`
(lifecycle + KV-overflow) are written, registered in `examples/llm/CMakeLists.txt`, and
tracked as patch `engine-overlay/patches/v080-0003-asr-streaming-spikes.patch`. They drive
the new `rt::AsrStreamingSessionRuntime` over a real `LLMInferenceRuntime` using text token
IDs (no mel) to isolate the prefill-chunk seam. Needed `loadEdgellmPluginLib()` in `main()`
to register `AttentionPlugin` before deserialize.

## Phase 3b empirical result (full evidence: `docs/plans/v080-phase3b-acceptance-results.md`)
Built a fresh v0.8.0 Qwen3-ASR-0.6B engine (the May-1 ONNX config lacked the now-required
`kv_cache_dtype` → had to re-export; `hf-xet` uninstalled to dodge a cas-bridge stall).
Engine loads in official `llm_inference` (INFER_EXIT=0, EOS) → genuine v0.8.0 compat.
Engine md5: llm `b133dff24c8aa96ac1679b95e2f97153`, audio `5c877cfe58b8fcb7914679c6fe274f90`.
- **R2: byte-identical (max-abs logit diff = 0.0, argmax MATCH) for every split whose final
  chunk is ≥ 2 tokens** (N=40/64/100, splits 1..N-2). KV continuity (R3) holds.
- **m2 PASS**: session-pair teardown byte-identical, KV-overflow refused without silent
  advance, append-after-end refused.
- **R4** structurally verified in the compiled binary (`restoreKVCache` now gated inside
  `if(matchIds)`; mismatch → fresh prefill).
- ⚠️ **One regression for Phase 4**: a **1-token FINAL prefill chunk** diverges (logit diff
  ~20, argmax can flip; KV length still correct). Root cause points at the last-token gather
  / `selectTokenIndices` using the per-chunk length (=1) instead of the accumulated position;
  the one-shot path never hits `inputIdsLength==1` so it was invisible to code inspection.
  Fix: gather at the accumulated sequence position when chunkLen==1, or never emit a 1-token
  final chunk. Re-run `spike_v080_m1_append_prefill $ENG N N-1` to confirm.

The old spikes ported FROM: `spike_m1_append_prefill_embeds.cpp`, `spike_m2_session_lifecycle.cpp` (this dir).

## Phase 4 — per-lane KV reset + SessionLaneManager (LANDED 2026-06-10, patch `v080-0005`)
Full evidence: `docs/plans/v080-phase4-acceptance-results.md`. Summary:
- **R1 lane-isolation GATE — GREEN.** New `spike_v080_m3_lane_isolation.cpp` drives a hand-built
  2-lane `HybridCacheManager` directly (validation OPTION **(a)** — mechanism-level proof; the
  on-box ASR engine is `maxBatchSize=1` so it cannot host 2 live lanes, and a full 2-session run
  is Phase 5). Seed lane0 KV=0xAA len=5, lane1 KV=0xBB len=7 → `resetLanes({0})` → re-capture:
  **lane1 KV-length UNCHANGED (7), lane1 KV bytes UNCHANGED (byte-equal, all 0xBB); lane0 length
  now 0; lane0 KV bytes intact (length-only reset); allEmpty=false.** `SessionLaneManager` ASR/TTS
  partition + release/re-acquire verified. Exit 0.
- **No regression.** Re-run against the real v0.8.0 ASR engine: `m1 64 63` and `m1 64 32` →
  argmax MATCH, max-abs logit diff **0.0**; `m2` → all 3 scenarios PASS (teardown byte-identical,
  KV-overflow refused, append-after-end refused). The Phase-3 full-batch `resetForSessionEnd` path
  is retained as the no-lane-manager fallback (exercised by m2) → byte-unchanged.
- **Memset-not-reshape** confirmed: `resetLanes`/`setLaneLength` operate per-element on
  `mDeviceKVCacheLengths[batchIdx]`; `resetForNewSequences` (the sole reshaping path) is untouched.
- Note: the v080-0004 single-token-append guard means the Phase-4 1-token-final case is no longer
  a divergence (the carry-over coalesces N-1 splits) — m1 64/63 (final chunk = 1 raw token, carried)
  is diff 0.0. The ⚠️ Phase 3b "1-token FINAL" caveat above is therefore CLOSED by v080-0004.
- **Phase 5 (batch-lane concurrency over a maxBatchSize=2 engine) is UNBLOCKED** — the R1 isolation
  mechanism it depends on is proven.

## Phase 5a — batch-lane N=2 concurrency + isolation gate (LANDED 2026-06-10, patch `v080-0006`)
Full evidence: `docs/plans/v080-phase5a-acceptance-results.md`. Summary:
- **maxBatchSize=2 engine** built from the SAME on-box `onnx-v080/llm` (no re-export — `--maxBatchSize`
  is a `llm_build` flag) → `engines-v080-b2/llm/llm.engine` md5 `4122dfcc666fe82b8b0cae4b93c97b70`,
  config `"max_batch_size": 2`. The maxBatch=1 engine (`b133dff…`) is untouched.
- **New net-new layer `AsrContinuousBatcher`** (`cpp/runtime/asrContinuousBatcher.{h,cpp}`, GLOB'd
  into `edgellmCore`): owns ONE shared `DecodingInferenceContext` (row i == KV lane i == MRope row
  i) over `SessionLaneManager`; `admit`/`enqueueChunk`/`tick`/`evict`. `tick()` packs ONLY the lanes
  with a pending chunk into ONE batched `appendPrefillChunk` (reuses `runBaseModelPrefillChunk`
  across the active batch), samples per-lane on final, evicts via per-lane `resetSessionLane`. Idle
  lanes are never packed (empty span → `selectTokenIndices=-1` corrupts the lane). Batch-wide
  single-token guard (refuse iff `max(stepLen)==1 && anyPriorKv` — the AttentionPlugin keys
  prefill-vs-decode on the batch-wide `qInputTensor.shape[1]`, so mixed `[1,5]` ticks are safe).
- **Codex design verdict (file:line)**: row<->KV-row<->MRope-row alignment confirmed — no indirection
  in `runBaseModelPrefillChunk` (`llmInferenceRuntime.cpp:1098-1141`); `StepPreparer::prepare`
  rowwise (`stepPreparer.cpp:37-59`); `kvcache_start_index` bound `[batch]`
  (`pipelineIO.cpp:185-190`, `registryBuilder.cpp:59-67`, `llmEngineConfig.cpp:458-481`); RoPE kernel
  indexes `cosSinCacheBatchIdx=batchIdx` when batch != 1 (`applyRopeWriteKV.cu:161-168`); mode
  decision batch-wide from `qInputTensor.shape[1]` (`attentionPlugin.cpp:93-109`), per-lane absolute
  KV-end = `kvCacheStartIdx + runtimeSeqLen` (`utilKernels.cu:54-58`). Idle lanes corrupt
  (`stepPreparer.cpp:46-53`). `resetLanes` per-lane memset (`hybridCacheManager.cpp:276-298`);
  non-prefix/lone-survivor lane needs `performBatchEvict`-style compaction
  (`llmInferenceRuntime.cpp:1610-1685`).
- **THE N=2 ISOLATION GATE — GREEN.** New `spike_v080_m4_concurrent_lanes.cpp`: two DISTINCT token
  streams (seeds 1234/9876), solo baseline per session, then concurrent lockstep activeBatchSize=2.
  Per-lane concurrent-vs-solo **max-abs logit diff = 0.000000e+00** (argmax MATCH) for BOTH lanes,
  **0 CUDA errors**, across splits 64/32, 64/63 (1-tok-final carry-over), 64/17, 100/50. Zero
  cross-talk proven byte-identical.
- **No regression** on the maxBatch=2 engine (single-lane subset): m1 64/63 + 64/32 → argmax MATCH,
  diff 0.0; m2 → all 3 scenarios PASS. Raising the builder batch dim is accuracy-neutral.
- **Remaining ASR track**: async (non-lockstep) eviction needs lone-survivor lane compaction; mel
  chunks through the batcher; worker/server-loop wiring; live ASR+TTS lane co-residency. **TTS track
  is separate** (Talker-batched + sequential CodePredictor/Code2Wav, replacing patch-0004's runtime
  replication with one `maxBatchSize=N` Talker runtime).
