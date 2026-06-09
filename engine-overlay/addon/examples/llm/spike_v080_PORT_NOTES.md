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
1. **Audio chunk embedding API + session MRope init** (Phase 2). New `Qwen3OmniAudioRunner::encodeMelChunk` + `initializeSequentialMRopeLane` over `PipelineIO::mropeCosSin` (spec §5 R3; old `0003:298-329,338-381`).
2. **`SessionLaneManager` + targeted `HybridCacheManager` lane reset** (Phase 3/4, spec R1). Add `resetLanes(span<int32_t>)`, `setLaneLength`, `maxBatchSize` next to `resetForNewSequences` (`hybridCacheManager.h:162-176`) — per-lane memset, NO reshape.
3. **`LLMInferenceRuntime::runBaseModelPrefillChunk(...)`** factored from `runBaseModelPrefill` (`.cpp:1070-1136`) so it packs ONLY the chunk slice, not all `context.tokenIds` (spec R2; the `:1105-1108` copy is the seam). One-shot path becomes a thin wrapper → byte-identical.
4. **`AsrStreamingSessionRuntime`** begin/append/end wrapper around `LLMInferenceRuntime` (spec §5.2). NOT a DecodingStrategy.
5. **Session introspection**: append status, `peekKvCacheLength`, session status query.
6. **Testing surface**: `getLogitsForTesting` (so the spikes' R2 single-vs-split argmax/logit parity is observable).
7. **Sys-prompt mismatch fallback** BEFORE cached-KV restore in `setUpForPrefillExecution` (spec R4; move validation ahead of `:1339` restore, fresh-prefill on mismatch).

## Spike files (deferred)
`spike_v080_m1_append_prefill.cpp` (R2 one-chunk-vs-split parity) and `spike_v080_m2_session_lifecycle.cpp` (begin/append/end + KV-overflow refusal) will be written alongside Phase 3 when hooks #3/#4/#6 land — porting them now yields only TODO-stubs against non-existent APIs. The old spikes to port FROM: `spike_m1_append_prefill_embeds.cpp`, `spike_m2_session_lifecycle.cpp` (this dir).
