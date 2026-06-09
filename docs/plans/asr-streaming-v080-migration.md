# Streaming-ASR Session Migration Spec: v0.7.1 → v0.8.0

Status: DESIGN (codex spec, main-thread verified against `v0.8.0` tag 2026-06-09)
Scope: re-implement our chunked-prefill streaming-ASR session (overlay `patches/0003-asr-streaming-session.patch`) on TensorRT-Edge-LLM **v0.8.0**'s re-architected runtime.

## Why this is a re-implementation, not a rebase
v0.8.0 **deleted** `cpp/runtime/llmInferenceSpecDecodeRuntime.{cpp,h}` — the host of ~77% (~960 lines) of patch 0003 — and split the C++ runtime into composable execution / decoding / preprocessing / state-management components. v0.8.0's Qwen3-ASR is **one-shot only** (full audio → transcribe); there is no native streaming/chunked-prefill session to adopt. The streaming session must be re-built against the new component model.

## Verification status (main thread, against `v0.8.0`)
Key anchors confirmed real:
- `cpp/runtime/llmInferenceRuntime.cpp` is the new request runtime. `runBaseModelPrefill` at **:1070**; line **:1108** `std::copy(context.tokenIds[i]...)` confirms it packs the ENTIRE accumulated `context.tokenIds[i]` → refactor into a chunk-only helper is mandatory, not optional.
- `setUpForPrefillExecution` at **:1296**; **:1332** `context.tokenIds.clear()` → appendChunk must NOT call it.
- **:1403** `mSharedResources->cacheManagers[0]->resetForNewSequences(...)` → process-lifetime cache manager, full-batch reset.
- `DecodingInferenceContext` fields at `cpp/runtime/state/decodingInferenceContext.h:61-74` (tokenIds, effectivePrefillLengths, slotStreams, audioEmbeddings, activeBatchSize).
- `cpp/runtime/state/sharedResources.h` confirmed.

**CORRECTION to codex paths:** `HybridCacheManager` lives at **`cpp/runtime/hybridCacheManager.{cpp,h}`** (NOT `cpp/runtime/state/hybridCacheManager.*`). Methods `commitSequenceLength` / `resetForNewSequences` are real (used in `cpp/runtime/decoding/eagleDecoder.cpp:270,544,645`). Apply this path fix wherever the table below says `state/hybridCacheManager`.

---

## Section 1: Architecture Mapping Table

| Old touchpoint | Old ref | v0.8.0 owner | v0.8.0 ref |
|---|---|---|---|
| Deleted runtime home `LLMInferenceSpecDecodeRuntime` | `DIVERGENCE.md:48-49`; file absent in v0.8.0 | `LLMInferenceRuntime` is now the request-level runtime | `cpp/runtime/llmInferenceRuntime.h:110`, `llmInferenceRuntime.cpp:402` |
| Request entry + context construction | patch `0003:564-599` (`handleRequest`) | `handleRequest` builds `DecodingInferenceContext`, selects strategy, preprocesses, calls setup+prefill | `llmInferenceRuntime.cpp:431-455`, `542-580` |
| Per-request/session state | old `SpecDecodeInferenceContext` patch `0003:775-807`, `946-963` | `DecodingInferenceContext` | `state/decodingInferenceContext.h:57-76`; `decodingInferenceContext.cpp:25-55` |
| One-shot prefill setup (LoRA, sys-prompt KV restore, reuse lengths, cache reset) | `setUpForPrefillExecutionOneShot` patch `0003:604-721`, `.h:1630-1638` | `LLMInferenceRuntime::setUpForPrefillExecution`; strategy cache hooks | `llmInferenceRuntime.cpp:1296-1408`; `decoding/decodingStrategy.h:114-124` |
| Chunk-only setup (must NOT reset KV or clear tokens) | `setUpForPrefillExecutionForChunk` patch `0003:725-763`, `.h:1640-1649` | No public equivalent — add new ASR append path over lower components | `runBaseModelPrefill` shape/pack/embed at `llmInferenceRuntime.cpp:1070-1137`; `HybridCacheManager::commitSequenceLength` at `hybridCacheManager.cpp:276-289` |
| Base prefill execution | old `executePrefillStep` patch `0003:1182-1195` | `runBaseModelPrefill` packs ids, embeds, lengths, calls `EngineExecutor::prepare/execute`, commits KV | `llmInferenceRuntime.cpp:1076-1137`; `exec/engineExecutor.cpp:86-155`, `158-177` |
| KV start-index / empty-vs-chunked prefill | active cache lengths patch `0003:1001-1028`, `1417-1420` | registry binds `kvcache_start_index`; shape `[0]` only when all empty, else `[batch]` | `exec/registryBuilder.cpp:59-67`; `state/pipelineIO.cpp:185-190`; `config/llmEngineConfig.h:156-162` |
| KV slot ownership / reset / commit / capture-restore | begin/end + peek patch `0003:803-840`, `1208-1223` | `SharedResources` owns one `HybridCacheManager` per engine | `state/sharedResources.h:40-49`; `hybridCacheManager.h:158-176`, `210-221`; `hybridCacheManager.cpp:247-289`, `367-451` |
| System-prompt cache generation | patch `0003:1225-1267` | base cache in runtime; draft cache in strategy | `llmInferenceRuntime.cpp:1411-1515`; `state/systemPromptKVCache.h:34-42` |
| Audio one-shot preprocessing | split into `encodeOneAudioImpl`/`encodeMelChunk` patch `0003:11-18`, `298-329` | v0.8.0 has only `preprocess()` + private `preprocessAudio()` — no chunk/session API | `multimodal/audioRunner.h:70-72`, `111-112`; `audioRunner.cpp:191-235`, `340-492` |
| MRope session init | `initializeMRopeForSession` patch `0003:338-381`, `458-476` | v0.8.0 allocates MRope in `PipelineIO`; text-only in ctor, one-shot audio in preprocess | `pipelineIO.cpp:75-79`, `274-280`; `audioRunner.cpp:224-233`, `506-524`, `527-587` |
| Streaming output | `PerBatchAsrState` + `StreamChannel` patch `0003:1445-1476` | token-output streaming attached in `handleRequest`; `StreamChannel`/`slotStreams` reusable | `streaming.h:82-111`, `138-230`; `llmInferenceRuntime.cpp:548-564` |

---

## Section 2: Lifecycle Design

**Do NOT implement as a new `DecodingStrategy`** — strategies own post-prefill decode + draft-cache (`decoding/decodingStrategy.h:95-124`), but ASR streaming needs incremental prefill *before* decode. Correct insertion: a new composition component **`AsrStreamingSessionRuntime`** wrapping `LLMInferenceRuntime` + narrow runtime hooks.

- **`beginAsrSession`**: init a `DecodingInferenceContext` as `handleRequest` does after multimodal formatting (`llmInferenceRuntime.cpp:431-455`) but skip one-shot audio preprocess. Call `setUpForPrefillExecution` once so LoRA / sys-prompt KV reuse / recurrent zero-restore / cache lengths are correct (`:1296-1408`). For MRope, re-apply the patch's virtual hook (`0003:458-476`) against `mPipelineIO->mropeCosSin` (MRope now in `PipelineIO`, `pipelineIO.h:50-53`, `pipelineIO.cpp:274-280`).
- **`appendPrefillChunk`**: must NOT call `setUpForPrefillExecution` (it clears `context.tokenIds` + `resetForNewSequences`, `:1332-1334`, `:1403-1407`). Factor `runBaseModelPrefill` (`:1076-1137`) into a private `runBaseModelPrefillChunk` accepting pre-encoded per-chunk token slices + explicit audio embeddings. Steps: reshape `mIdsInput`/`hostContextLengths`/`inputsEmbeds`/logits; pack **only** chunk ids (`:1105-1112`); multimodal embedding lookup (`:1114-1117`); `StepPreparer::prepare(kPrefill)` (`:1119-1121`); execute with `prefillDims(activeBatchSize, chunkLen, kvCacheAllEmpty=false)` (`:1128-1135`); `commitSequenceLength` (`:1136`; `hybridCacheManager.cpp:276-289`). Preserves cross-chunk KV accumulation.
- **`endAsrSession`**: reset occupied lanes via `HybridCacheManager::resetForNewSequences` (`hybridCacheManager.cpp:247-274`) + clear session context vectors, mirroring old teardown (`0003:803-840`). If product slots share one runtime, lane allocation must fit within a single active-batch context (old batched-append contract `0003:1478-1529`), not independent contexts.

---

## Section 3: Hard Risks

1. **KV slot ownership.** v0.8.0 has process-lifetime `SharedResources::cacheManagers` (`state/sharedResources.h:40-49`), not per-session. TTS slot-pool may share the same engine runtime. `resetForNewSequences` resets the *entire* active-batch tensor (`hybridCacheManager.cpp:257-273`) → partial-lane teardown unsupported → **a lane allocator is required**; for independent concurrent-session teardown, a targeted lane reset must be added to `HybridCacheManager`.
2. **Incremental prefill.** Non-empty-KV prefill IS possible via `prefillDims(..., kvCacheAllEmpty=false)` + `[batch]` `kvcache_start_index` (`registryBuilder.cpp:59-67`), but no public partial-prefill API. Calling `runBaseModelPrefill` directly re-embeds ALL accumulated tokens (`:1107-1108`, verified) → refactor mandatory.
3. **MRope.** v0.8.0 inits sequential MRope for one-shot audio every preprocess (`audioRunner.cpp:224-233`) + text-only at `PipelineIO` ctor (`pipelineIO.cpp:274-280`). Streaming must init once per session over max KV capacity + active lanes, then guarantee chunk appends never overwrite `mropeCosSin`. Correctness invariant.
4. **System-prompt cache.** v0.8.0 warns but does NOT fall back on token mismatch (`:1372-1379`). Restore the patch's stricter fallback (`0003:622-642`, `700-710`) — MRope positions baked into cached KV can silently misalign on mismatch.

---

## Section 4: Phased Plan (14–21 engineering days, 1 dev fluent in TRT-Edge-LLM C++)

- **Phase 1 — Scaffolding (2-3d).** Add `addon/examples/llm/spike_v080_m1_append_prefill.cpp`, `spike_v080_m2_session_lifecycle.cpp`. Port control flow + asserts from `spike_m1_append_prefill_embeds.cpp` / `spike_m2_session_lifecycle.cpp`, swapping `SpecDecodeInferenceContext` → `DecodingInferenceContext`. Goal: compiles vs v0.8.0 headers (no runtime yet).
- **Phase 2 — Audio chunk API (2d).** Touch `cpp/multimodal/audioRunner.{h,cpp}`, `multimodalRunner.h`. Reapply `encodeOneAudioImpl`/`encodeMelChunk`/`initializeMRopeForSession` (`0003:11-18`, `298-329`, `338-381`, `394-426`); wire MRope to `PipelineIO::mropeCosSin`. Test driver from `spike_m35_audio_runner_split.cpp`. Gate on confirming MRope tensor layout unchanged v0.7.1↔v0.8.0.
- **Phase 3 — Runtime hooks + chunk-prefill factor (4-6d).** Touch `cpp/runtime/llmInferenceRuntime.{h,cpp}`; add `cpp/runtime/asrStreamingSessionRuntime.{h,cpp}` (the wrapping component). Expose `beginAsrSession`/`appendPrefillEmbedsBatched`/`endAsrSession`/`peekKVLength`/`sessionStatus`. Factor `runBaseModelPrefillChunk` (private, chunk slice + explicit embeds). Reuse append logic `0003:946-1199`. Do NOT change `setUpForPrefillExecution` contract.
- **Phase 4 — Cache/scheduler correctness (3-5d).** Touch `cpp/runtime/state/sharedResources.{h,cpp}`, `cpp/runtime/hybridCacheManager.{h,cpp}`. Per-lane allocator; decide if concurrent ASR sessions + TTS slot-pool can share one `HybridCacheManager` (if not, add targeted lane reset). Validate `kvcache_start_index` binding (`registryBuilder.cpp:59-67`) picks up partial-lane lengths after each `commitSequenceLength`.
- **Phase 5 — Validation (3-5d).** Reuse `spike_m36_empirical_lcs.cpp` for audio-split LCS regression. Required cases: bit-exact single-chunk vs split-chunk; lifecycle reset; KV-overflow refusal; MRope position continuity across chunks; sys-prompt cache mismatch fallback; hardware ASR roundtrip (`DIVERGENCE.md:51`).

---

## Section 5: Risk Mitigation Design

Status: DESIGN (codex), main-thread verified against `v0.8.0` 2026-06-09. Anchors confirmed:
`resetForNewSequences` reshapes the whole batch (`hybridCacheManager.cpp:247-273`: `mActiveBatchSize=batchSize; mDeviceKVCacheLengths.reshape({mActiveBatchSize})` + global `mKVCacheAllEmpty`) → per-lane reset genuinely needs a new non-reshaping API. Capture/restore are already lane-addressed by `batchIdx` (`hybridCacheManager.h:215,221`) → adding `resetLanes` is consistent. Sys-prompt: `reuseKVCacheLengthsData[i]` is set BEFORE the `std::equal` match check that only `LOG_WARNING`s (`llmInferenceRuntime.cpp:~1366-1377`) → mismatched KV installs before the warning. `kvcache_start_index` binds directly to `cacheMgr.getKVCacheLengths()` (`pipelineIO.cpp:190`).

**Implementation order is a strict chain: Risk 4 → Risk 1 → Risk 3 → Risk 2.**

### R4 (first) — sys-prompt mismatch fallback
Move validation BEFORE `restoreKVCache` in `setUpForPrefillExecution` (`llmInferenceRuntime.cpp:1335-1380`). Add local `bool useCachedKV=false`; compute `reuseLength`+`shapeOk`, then `matchIds = shapeOk && cached.tokenizedPrompt.size() <= input.size() && std::equal(...)`. Only if both pass: restore base/strategy KV + recurrent state, set `reuseKVCacheLengthsData[i]=reuseLength-1`, assign suffix tokens. Else fall through to fresh prefill (`tokenIds[i]=batchedInputIds[i]`, `effectivePrefillLengths[i]=size`, `reuse=0`, zero recurrent — existing miss path `:1381-1390`). Mirrors old patch `0003:622-642,700-720`. In streaming, runs only in `beginAsrSession`; chunk append never revisits sys-prompt restore. Test: same prompt string, altered tokenization → assert `restoreKVCache` NOT called, reuse=0, output byte-equal to uncached full prefill.

### R1 (second) — lane allocator
New composition owner `SessionLaneManager` in `cpp/runtime/asrStreamingSessionRuntime.{h,cpp}`: `vector<LaneRecord>{ownerKind, ownerId, slotId, inUse, kvLength}` + mutex, constructed with `maxBatchSize` and static partitions ASR `[0, asrMax)` / TTS `[asrMax, asrMax+ttsMax)`. TTS gets a lane id at `SlotPool` acquire (`addon/cpp/runtime/slotPool.h:42-46`); `slotId` must NOT imply `batchIdx` unless the partition maps it. Add to `HybridCacheManager` (after `:162-176`): `resetLanes(span<int32_t const> batchIdxs, stream)`, `setLaneLength(int32_t batchIdx, int32_t length, stream)`, `maxBatchSize()`. Per-element `cudaMemsetAsync`/`cudaMemcpyAsync` on `mDeviceKVCacheLengths[batchIdx]`, **no reshape**; `mKVCacheAllEmpty` becomes conservative (false if any lane may be non-empty). Existing full-batch `resetForNewSequences` untouched (one-shot path). Test: 2 ASR + 1 TTS lane, reset lane 0 → assert only lane 0 zeroed, lane 1/TTS bytes unchanged.

### R3 (third) — session-scoped MRope by lane
`SessionMRopeState{int32_t lane; int32_t nextPosition; int32_t maxKvCapacity; bool initialized;}` owned by `AsrStreamingSessionRuntime`. `beginAsrSession`: init only the assigned lane row for `[0,maxKVCapacity)` via new `Qwen3OmniAudioRunner::initializeSequentialMRopeLane(lane, maxPositions, mropeCosSin, stream)` derived from `initializeSequentialMRopeCache` (`audioRunner.cpp:527-587`) but single-row, no sub-capacity reshape. `appendChunk`: `positionOffset=state.nextPosition`; require `positionOffset+chunkLen<=maxKvCapacity`; `state.nextPosition+=chunkLen`. Chunk path NEVER calls one-shot `preprocess()` MRope init. Lane id from R1 == MRope batch row; if ASR/TTS share `PipelineIO`, lane partition must also partition MRope rows. Test: 2 lanes, append to lane 0 → lane 1 MRope bytes unchanged; lane 0 pre-chunk-N positions unchanged after chunk N.

### R2 (last) — chunk prefill helper
Extract `llmInferenceRuntime.cpp:1076-1136` into private `runBaseModelPrefillChunk(context, vector<span<int32_t const>> chunkTokenIds, OptionalInputTensor audioEmbeds, bool sampleAfterPrefill)` (decl near `:291-294`). Reshape `mIdsInput`/`hostContextLengths`/`inputsEmbeds`/`outputLogits`/`baseHiddenStates` to `chunkLen` (NOT accumulated); write `hostCtxLenData[i]=chunkTokenIds[i].size()`; copy only the chunk slice (replaces `:1105-1108` full-token copy). Reuse embedding/deepstack/StepPreparer/executor prepare-execute + `commitSequenceLength` (`:1114-1136`). `runBaseModelPrefill(context)` becomes a thin wrapper over full-token spans → one-shot byte-identical. Test: one-shot vs 2-chunk split with same total tokens → final KV lengths equal, last-token logits within fp tolerance.

### Key conflict — shared PipelineIO serialization (IMPORTANT)
`mPipelineIO->inputsEmbeds`, `mIdsInput`, `hostContextLengths`, `outputLogits`, `baseHiddenStates`, `mropeCosSin`, `mDeviceKVCacheLengths` are runtime-owned shared mutable buffers. If ASR sessions and TTS slots share one `PipelineIO`, **concurrent engine calls with different active batch shapes corrupt these tensors**. Mitigation: a single `std::mutex engineExecMutex` on the runtime owning `PipelineIO`, held across the full prefill/decode step. Lane reservation + MRope init outside the step need no mutex. **Consequence:** ASR+TTS "N=2" becomes logical-lane reservation + *serialized engine steps*, not truly-parallel engine execution on a shared runtime. True parallelism requires separate runtime/cacheManager/PipelineIO instances (multiplies engine memory). This directly affects the 0004 TTS slot-pool concurrency moat — decide per-deployment whether serialized-step concurrency is acceptable or separate runtimes are warranted.

---

## Section 6: Concurrency Architecture

Status: DESIGN (codex), main-thread verified against `v0.8.0` 2026-06-09. Anchors confirmed:
one context/executor `createExecutionContext(kUSER_MANAGED)` (`engineExecutor.cpp:49-50`) with USER-MANAGED scratch via `setDeviceMemoryV2`/`getDeviceMemorySizeV2` (`engineExecutor.cpp:228,233`); exec-context memory is shared across engines *because they run serially* (`llmInferenceRuntime.cpp:~341`: "All engines execute serially so they can share a single buffer sized to the max requirement", `mSharedExecContextMemory = max({base,strategy,vision,audio,action})`); `PipelineIO` has an explicit POINTER STABILITY INVARIANT — must not move after `buildTensorMap`, `TensorMap` holds raw `Tensor*` into its members (`pipelineIO.h:35-38`).

### Resource ownership (what can be shared vs must be per-slot)
| Resource | v0.8.0 ownership | Concurrent-safe | Evidence |
|---|---|---|---|
| `ICudaEngine` (weights) | per `EngineExecutor`, deserialized | YES — TRT allows N contexts over 1 engine | `engineExecutor.cpp:33-50` |
| `IExecutionContext` | one per executor | NO — one context = single-threaded stateful | `engineExecutor.cpp:48-51` |
| exec-context scratch | one shared buffer, sized max over serial engines | NO if 2 engines run at once | `llmInferenceRuntime.cpp:341-370` |
| `PipelineIO` (inputsEmbeds/logits/hidden/MRoPE…) | per runtime, pointer-stable | NO | `pipelineIO.h:35-68`, `pipelineIO.cpp:35-78` |
| `HybridCacheManager` (KV) | per engine in `SharedResources` | NO without lane APIs (Section 5 R1) | `hybridCacheManager.cpp:247-273` |

> **SUPERSEDED.** Option C (per-slot context pool) below was the first cut. It is replaced by the leaner **batch-lane** design (v2) — leverage v0.8.0's native N-requests-as-N-batch-lanes path (one shared context/scratch/PipelineIO, KV per-lane) instead of replicating runtimes. Original Option C kept for record: refactor `EngineExecutor` to M contexts each with own scratch; rejected because it multiplies scratch+PipelineIO when the native batch dimension already gives true-parallel execution for free.

---

## Section 6 (v2): Batch-Lane Concurrency

Status: DESIGN (codex), main-thread verified against `v0.8.0` 2026-06-09. Supersedes Option C above.

### Decisive verdict — static-admission batching, NOT continuous
v0.8.0 fixes the admitted set at `handleRequest` entry (`activeBatchSize = request.requests.size()`, `llmInferenceRuntime.cpp:418`) and builds one request-local `DecodingInferenceContext` (`:453-455`). The decode loop **only evicts** finished lanes — verified: "Perform batch eviction if needed" → `compactBatch(mDeviceBatchMapping, oldActiveBatch, newActiveBatch)` + `setActiveBatchSize(newActiveBatch)` (`:1611-1614,1666`), shrink-only, no admit/refill in-loop. `compactBatch` also compacts MRoPE rows (baseRopeCache reshape old→new) — reusable by R3. **Consequence:** async-arriving streaming ASR/TTS sessions cannot ride native `handleRequest` continuously → we must ADD a thin continuous-batch scheduler that refills evicted lanes between steps. That scheduler is the one genuinely new layer; everything below it is native v0.8.0 batching.

### ASR — one thinker runtime, batch lanes + continuous scheduler
Native ragged batch prefill already exists: `runBaseModelPrefill` pads to `max_element(effectivePrefillLengths)` (`:1076-1078,1101-1103`), per-lane context lengths (`:1096-1108`), `StepPreparer::prepare(kPrefill, activeBatchSize)` (`:1119-1121`), per-lane `commitSequenceLength` (`:1136`). Chunk append is structurally supported: `kvcache_start_index` is `[0]` for empty prefill, `[batch]` for append (`registryBuilder.cpp:59-67`, bound at `pipelineIO.cpp:185-190`). Add an `AsrContinuousBatcher` owning persistent `AsrLane{sessionId, physicalLane, tokenIds, kvLen, decoding, finished, StreamChannel}` records over `SlotPool::acquireOrExisting` (`slotPool.h:145-149`); each `tick()` = refill free lanes → ragged `runBaseModelPrefillChunk` for lanes with pending chunks → batched `decodeStep` for ready lanes → evict+lane-reset finished. Hooks: R2 chunk-prefill helper (`:1076-1136`); R1 lane-targeted reset next to `resetForNewSequences` (`hybridCacheManager.cpp:247-273`); reuse `compactBatch` semantics (`:1548-1667`) but **allow refill after compaction**, not only shrink.

### TTS — Talker batches, CodePredictor + Code2Wav stay per-lane sequential
Talker is already batched: `handleAudioGeneration(vector<TalkerGenerationRequest>)` (`qwen3OmniTTSRuntime.h:156-177`), `activeBatchSize=requests.size()` (`:1048-1056`), batched prefill/decode/sample every frame (`:1253-1306,1473-1514`). **CodePredictor is batch=1** (verified comment "CodePredictor workspace — batch=1 (per-batch for-loop, each frame resets KV cache)", `:543-569`; frame loop `for b<activeBatchSize` with per-lane `resetForNewSequences({1})`, `:1407-1419`). Code2Wav single-sample (`code2WavRunner.h:67-73`, `.cpp:178-182`). So TTS concurrency = batched Talker + sequential CodePredictor/Code2Wav queue. **Patch-0004 slot-pool runtime-replication is REPLACEABLE** by one runtime at `maxBatchSize=N` (native `maxBS` buffers already exist `:529-598`); keep `SlotPool` only as a logical admission/cancel guard (`slotPool.h:71-89,114-149`), `slotId`→lane index, not a runtime instance. This is *less* migration work than porting 0004's per-slot ctor.

### Object graph + memory math
One runtime per engine type (ASR thinker / TTS talker / code2wav), sized to max batch, N KV lanes — scratch/PipelineIO/IExecutionContext NOT replicated (one context/executor `engineExecutor.cpp:48-51`; shared ctx-mem `llmInferenceRuntime.cpp:341-370`, `qwen3OmniTTSRuntime.cpp:122-135`). KV per-lane: `layerBytes = maxBatch × 2 × numKVHeads × maxSeqLen × headDim × elemSize` (`kvCacheManager.cpp:72-79`). vs Option C this saves `(N-1) × (context scratch + PipelineIO workspaces + duplicate non-KV tensors)` per engine, keeping only legitimate linear KV growth.

| Engine | Plan | KV/lane | Orin NX 16GB cap |
|---|---|---|---|
| ASR thinker | one runtime `maxBatchSize=2` | ~50 MB@B1 → ~100 MB@B2 (maxKV=512) | N=2 |
| TTS Talker | one runtime `maxBatchSize=2` | buffers already @maxBS | N=2 (LLM absent) / N=1 (Qwen3.5 resident) |
| CodePredictor | batch=1 frame loop | per-frame KV reset `{1}` | N=1 sequential |
| Code2Wav | sequential queue | single-sample | queue |

Realistic co-resident default: **ASR N=2, TTS Talker N=1, Code2Wav queued** (Qwen3.5-4B resident). TTS N=2 only for LLM-disabled or W8A16/AWQ-Talker deployments.

### What changes vs Section 5 / old Section 6
- **R1 SessionLaneManager / lane reset** — RETAINED; lane reset hits one lane in `hybridCacheManager.cpp:247-273`.
- **R2 chunk prefill** — RETAINED, now the primary scheduler primitive.
- **R3 MRope-per-lane** — RETAINED; `compactBatch` already MRoPE-compacts on evict (`:1611-1614`), reuse it.
- **`engineExecMutex`** — SIMPLIFIED to one mutex per runtime *step* (protects shared PipelineIO/logits/staging); no per-slot context locking.
- **`DeviceExecutionScheduler`** — RETAINED but thinner (arbitrates ASR/Talker/Code2Wav GPU turns; owns no runtime slots).
- **Option C per-slot context pool** — REPLACED by `maxBatchSize=N` single runtime.
- **NEW (only net-new layer):** `AsrContinuousBatcher` + `TtsBatchLaneManager` — lane add/evict refill between steps, since v0.8.0 never refills mid-loop.

## Note on scope decision
This migration is only warranted if a full v0.8.0 move is chosen for the **voice stack**. If the only driver is the Qwen3.5-4B memory fix, that lives on the separate `edge-llm-chat-service` LLM container and does NOT require migrating the ASR streaming session — keep the voice stack on v0.7.1 and skip this work.

**Memory driver RESOLVED (real-machine, 2026-06-09):** v0.8.0 fixes the Qwen3.5-4B GDN runtime-memory regression — measured **5.76 GB total RAM** on Orin NX 16GB (vs 12.9–13.5 GB on v0.7.1, same shape maxInputLen 4096); the 11891 MiB cuBLAS pre-alloc is gone, only a 957 MiB exec-ctx allocation remains. Model identity verified (`qwen3_5_text`, 24 GDN + 8 attn, `hybrid_mamba` engine). ⇒ **Recommended path: upgrade ONLY `edge-llm-chat-service` to v0.8.0** to unlock Qwen3.5-4B + ASR/TTS/LLM co-residency (~1 week), leaving the voice stack on v0.7.1. This Sections 1-6 voice-stack migration is the SEPARATE, larger track — execute only with an independent driver.
