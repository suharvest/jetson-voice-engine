# v0.8.0 qwen3_asr_worker Incremental-KV Streaming Partials Plan

Source note: the overlay worktree has `third_party/qwen3-edgellm-jetson` as an unpopulated submodule, so runtime citations below are grounded in the populated sibling checkout's `engine-overlay/patches/0003-asr-streaming-session.patch`. The named `v080-0012` helpers (`decodeAsrSessionToCompletion`, `decodeToCompletion`, `getAsrSessionTranscript`) were not present in readable files; helper-specific lines are marked `[inferred]`.

## Goal

Replace the current cumulative re-decode path with O(N) incremental prefill: each hop extracts only new audio, runs `encodeMelChunk`, appends the resulting audio/text-token slice to the session lane's live KV via `appendPrefillEmbeds(Batched)`, then emits a non-evicting transcript peek. The current worker still builds a full cumulative request in `runStreamingHop` and calls `runtime.handleRequest` at `native/edgellm_voice_worker/qwen3_asr_worker.cpp:612-663`; `handleChunk` invokes that on every hop at `qwen3_asr_worker.cpp:1016-1019`, so cumulative PCM/mel costs grow quadratically.

## Runtime Primitive To Add: `peekTranscript`

Add `LLMInferenceSpecDecodeRuntime::peekAsrSessionTranscript(context, lane, maxNewTokens, stream) -> DecodePeekResult`. It must decode to EOS from the current post-prefill logits without finalizing the real session. Do not call `decodeAsrSessionToCompletion` for partials because final decode/teardown is expected to reset/release state `[inferred: v080-0012]`; the visible teardown API proves the destructive boundary: `endAsrSession` zeros reuse lengths through `resetForNewSequences` and clears `context.tokenIds` / `effectivePrefillLengths` at `engine-overlay/patches/0003-asr-streaming-session.patch:807-840`.

Implementation strategy: decode-then-restore, not deep-copy all KV tensors. Before the peek, save:

- Lane KV length: read `HybridCacheManager::getKVCacheLengths()` as `peekKvCacheLength` already does at `0003-asr-streaming-session.patch:1211-1222`. Also save the full `[activeBatchSize]` KV-length vector for batched mode, because append capacity checks already D2H all N lengths at `0003-asr-streaming-session.patch:1001-1014`.
- `context.tokenIds[lane]`, because the decode loop appends first token and subsequent tokens; the test decode helper appends at `0003-asr-streaming-session.patch:875-878` and relies on `runVanillaDecoding` appending more at `0003-asr-streaming-session.patch:897-900`.
- `context.currentGenerateLengths[lane]` and `context.generationRound`, because the helper increments them at `0003-asr-streaming-session.patch:877` and `:887-900`.
- `context.finishedStates[lane]` if present in this branch `[inferred: v080-0012]`; final decode uses finished/completed bookkeeping to stop and evict completed batches `[inferred: decodeToCompletion]`.
- MRope cursor/cache-visible state. The visible ASR path initializes MRope once at begin and per-chunk encoding does not touch it (`0003-asr-streaming-session.patch:765-799`, `:338-380`), so there should be no mutable MRope cursor in worker code; if v080-0012 added one, snapshot it with the decode context `[inferred]`.

After greedy decode, copy the generated tokens/text into `DecodePeekResult`, then restore `context.tokenIds[lane]`, `currentGenerateLengths[lane]`, `generationRound`, finished state, and the KV length for that lane. The minimum runtime API needed is `restoreKvCacheLength(lane, savedLen, stream)` implemented by writing the saved int to the cache-length tensor; do not zero/copy the KV memory. Future appends use the restored length as `kvcache_start_index`, and overwritten speculative peek KV tail becomes dead. This matches the append contract: live cache length drives capacity/start index and append must not call one-shot reset (`0003-asr-streaming-session.patch:1001-1028`, `:1031-1043`, `:1417-1420`). Guard the whole peek under `gEngineExecMutex` `[inferred: v080 worker]` so no append observes temporary peek lengths.

## Delta Extraction And Append

Extend per-session state beyond the current `AsrSessionState` fields (`qwen3_asr_worker.cpp:124-170`) with:

```cpp
struct IncrementalAsrBookkeeping {
  int32_t laneIndex;
  int32_t pcmSamplesConsumed;
  int32_t melFramesConsumed;
  int32_t audioTokensConsumed;
  int32_t prefillTokensConsumed;
  bool runtimeSessionBegun;
  std::vector<int32_t> accumulatedPromptAudioTokenIds;
};
```

For `pcm_b64`, the worker currently overwrites `session.pcmAccum` with the latest cumulative PCM at `qwen3_asr_worker.cpp:886-890` and computes mel over the full buffer at `qwen3_asr_worker.cpp:891-912`. Change the streaming path to slice `pcm[pcmSamplesConsumed:]`, compute mel only for that delta, then advance `pcmSamplesConsumed` after a successful append. For `mel_path`, read the cumulative mel's frame count and slice frames `[melFramesConsumed:T]`; advance `melFramesConsumed` only after successful append. Existing `audio_sec` remains the cap/segmentation signal at `qwen3_asr_worker.cpp:943-952`.

The delta mel is encoded through `Qwen3OmniAudioRunner::encodeMelChunk`, which is explicitly a no-text/no-MRope streaming entry point (`0003-asr-streaming-session.patch:298-329`, header contract at `:394-407`). Build the token slice for only the new audio-pad span plus required suffix tokens, then call `appendPrefillEmbeds` / `appendPrefillEmbedsBatched` with `audioIndexBase = audioTokensConsumed`. The append API is designed for per-chunk embeddings plus cumulative base (`0003-asr-streaming-session.patch:1422-1438`) and extends only the lane's `context.tokenIds[b]` while setting per-chunk length (`0003-asr-streaming-session.patch:1031-1043`).

Single-token guard interaction: never emit a one-token append for a non-final hop. If the delta token slice length is 1, buffer it in session bookkeeping and wait for the next hop; on `last=true`/`end`, append it together with the final suffix or pad to the minimal safe slice according to the v080-0004 guard `[inferred: v080-0004 not readable]`. This avoids the known final-chunk hazard mid-stream while preserving final convergence.

## Per-Hop Lifecycle

`begin`: acquire the session lane from `gSessions` / `SessionLaneManager` `[inferred: v080 worker]`, initialize the shared `SpecDecodeInferenceContext` for active N lanes, call `beginAsrSession(context, stream, activeBatchSize)`, and record the lane. The begin API intentionally runs one-shot setup and initializes session MRope once (`0003-asr-streaming-session.patch:775-800`, header at `:1368-1386`). Existing single-session begin fields are parsed at `qwen3_asr_worker.cpp:726-753`; N-lane work replaces the single `AsrSessionState session` owned in `main` at `qwen3_asr_worker.cpp:1284-1287` and dispatched at `:1380-1392`.

`chunk last=false`: under `gEngineExecMutex`, extract delta, `encodeMelChunk`, append to lane KV, then call `peekAsrSessionTranscript`. Emit `partial` with the peek text. Do not update final text accumulators from the peek except the visible partial cache; the durable state is KV + prefill token/audio cursors.

`chunk last=true`: append the final delta, then call the destructive final decode path (`decodeAsrSessionToCompletion` / `decodeToCompletion`) and `getAsrSessionTranscript` `[inferred: v080-0012]`. Then call `endAsrSession` to reset cache lengths and clear token state (`0003-asr-streaming-session.patch:807-840`) and release the lane.

`end` without final chunk: if no pending delta, final-decode current KV; otherwise flush pending guarded tokens first. Preserve existing bare-end behavior only for empty sessions (`qwen3_asr_worker.cpp:1130-1150`).

N>1 composes by batching appends for all ready lanes with `appendPrefillEmbedsBatched`: one context carries `activeBatchSize=N`, per-lane data lives in `context.tokenIds[b]` / `effectivePrefillLengths[b]`, and inputs are packed `[N,maxChunkLen,H]` with per-lane lengths (`0003-asr-streaming-session.patch:936-948`, `:1059-1189`, header `:1478-1529`). Keep decode peeks serialized by `gEngineExecMutex` until per-lane decode state is proven independent.

## MRope And Convergence

MRope continuity is preserved because begin initializes the full session cache for all lanes (`0003-asr-streaming-session.patch:338-380`, `:784-799`) and `encodeMelChunk` explicitly avoids MRope side effects (`:298-301`). Append uses `audioIndexBase=audioTokensConsumed`, so multimodal indices remain continuous across chunks (`0003-asr-streaming-session.patch:1086-1128`). Peek must restore any decode-side MRope cursor if present `[inferred]`; otherwise restoring tokenIds and KV lengths is sufficient.

Convergence proof sketch: after K hops, the live prefill KV equals the KV that one-shot would have after prefill over the concatenated audio, by induction over chunk appends. Base is `beginAsrSession`; step uses bit-exact chunk encoding (R2 acceptance registered at `0008-build-misc-example-registration.patch:48-65`) and append's additive token/KV contract. Since peeks restore all generated-token state and KV length, they leave the induction invariant unchanged. Final decode starts from the same full-audio KV as one-shot and uses the proven decode loop, so final transcript must match one-shot with CER 0.0000. Gate: partials may revise, but final must equal one-shot on the v0.8 corpus; latency is O(total chunks) because each hop encodes/appends only delta audio, not cumulative audio.

## Phasing, Risk, Effort

1. Runtime peek harness: implement save/restore around the existing test decode loop (`0003-asr-streaming-session.patch:849-907`) and assert KV length/tokenIds unchanged after 100 peeks.
2. Worker N=1 incremental path: add cursors, delta extraction, encode+append, final decode; compare final to one-shot.
3. Enable partial peek: emit partials every hop; add a test that peeks do not change next-hop append KV length.
4. Batch/N-lane integration: wire `gSessions` lanes to `appendPrefillEmbedsBatched`, then serialize peeks under `gEngineExecMutex`.
5. Gates: R2/R3 runtime spikes, streaming final==one-shot CER 0.0000, and hop latency grows linearly with chunks.

Biggest risk is KV/state corruption from an incomplete peek restore, especially leaving a longer KV length after speculative decode. Estimate: 2-3 days for N=1 correctness, 1-2 days for robust N-lane scheduling and gates, plus 0.5 day for Orin profiling.
