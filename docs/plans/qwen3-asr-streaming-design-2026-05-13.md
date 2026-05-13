# Qwen3-ASR Streaming Path Design — chunk-causal encoder + incremental thinker prefill (highperf fork)

Status: revised after codex review (2026-05-13)
Owner: harvest
Date: 2026-05-13
Target fork branch: `suharvest/TensorRT-Edge-LLM @ qwen3-tts-highperf-runtime-w8a16`

> Revision history:
> - **v5 (2026-05-13 final)**: P0 design pivot. Empirical LCS test
>   proved naive chunked encoder prefill loses information (premature
>   EOS truncation, LCS 0.37–0.74 at chunk_blocks 1–4). Root cause
>   verified by codex audit: encoder's `cu_seqlens` bidirectional
>   attention per-call narrows context window. **No retrofit fix:**
>   attention KV cache fixes middle chunks only, first/last fundamentally
>   need future audio. Discovery: **official Qwen3-ASR vLLM streaming
>   uses chunk-and-confirm with prefix prompt rollback**
>   (`qwen_asr/inference/qwen3_asr.py:657-770`). P0 mirrors that.
>   SLI also corrected: target is **end-of-speech → final-emit latency**
>   (≤500 ms), not first-partial (irrelevant — partials already shown
>   during speech). All chunked-encoder work (M3.5 / M3.6 phases) is
>   archived. M1+M2 infrastructure stays as P1/P2 foundation.
> - v4 (2026-05-13 fourth pass): codex re-audit on v3+spike data. Adds
>   §12 implementation order (`appendPrefillEmbeds` first, not last) and
>   §13 gaps-to-resolve (timeout, KV capacity, LoRA cache, W8A16 chunk
>   policy, mel parity).
> - v3 (2026-05-13 third pass): Spike A/B/B2 real-hardware findings.
>   Encoder rebuild dropped, plugin transient state proved clean
>   bit-exact, position-id management auto-derived. P0 estimate 3 weeks.
> - v1 (2026-05-13 draft): initial design.
> - v2 (2026-05-13 revised): codex review applied. Major corrections:
>   - P0 cannot reuse the existing audio engine — encoder profile rebuild is mandatory.
>   - Thinker engine's `max_input_len=128` is a hard cap; must be rebuilt for chunked prefill.
>   - Encoder uses 3× stride-2 conv2d front end + bidirectional block attention, not pure self-attn — P1/P2 are deeper than originally scoped.
>   - Function name was `createChunkwiseAttentionMask`, not `createMergedWindows`.
>   - `audioEmbeddings` lives on `SpecDecodeInferenceContext`, not `SlotStreamState`.
>   - Encoder builder is C++ `AudioBuilder`, not a Python script.
>   - Added: MRope per-chunk offset, KV rollback, audio BOS/PAD/EOS streaming coordination, missing C++ PCM→mel.

## 1. Goal

Convert the highperf Qwen3-ASR product path from one-shot batch inference to
true streaming: client pushes audio in ~100–500 ms chunks, server emits partial
hypotheses per chunk and a final hypothesis at end-of-utterance. Target latency
on Orin Nano: **first partial ≤ 350 ms after first chunk arrives, final ≤ 200 ms
after last chunk** (vs. current ~430–470 ms full-utterance warm roundtrip).

Constraint: no encoder retraining in P0. Quality regression vs. one-shot must
stay within ~1% LCS-similarity on the existing reproduction prompts.

## 2. Non-goals

- Not changing TTS path. TTS is already chunk-streaming.
- Not adding server-side VAD in P0 (client supplies chunk boundaries and
  `last=true` on the terminal chunk; VAD can be added later).
- Not switching to a different ASR model (no Paraformer, no U2++).
- Not addressing barge-in / duplex behavior; that is a `jetson-voice` concern.

## 3. Current (non-streaming) path — file:line evidence

End-to-end, the request path today is:

1. `native/edgellm_voice_worker/qwen3_asr_worker.cpp:127–188` reads one JSON
   line per request, writes a temp file, calls
   `parseRequestFile` then `runtime->handleRequest(...)` exactly once per
   utterance.
2. `runtime->handleRequest` is declared in
   `tensorrt-edge-llm:cpp/runtime/llmInferenceSpecDecodeRuntime.h:189` — one-shot:
   prepare multimodal inputs → encoder infer → thinker prefill → decode loop →
   return outputs.
3. Audio encoder lives in
   `tensorrt-edge-llm:cpp/multimodal/audioRunner.cpp` (Qwen3OmniAudioRunner).
   Key facts:
   - Config knobs `n_window=50`, `n_window_infer=200`
     (audioRunner.cpp:125–126) — they already chunk the mel into windows.
   - Mel is preloaded full-length into `mMelSpectrogram` sized
     `[1, melBins, maxTimeSteps]` (audioRunner.cpp:183).
   - `createPaddedFeatures(... nWindow ...)` (audioRunner.cpp:389) splits the
     full mel into `numChunks` padded chunks.
   - `createChunkwiseAttentionMask(... nWindow, nWindowInfer ...)`
     (`cpp/multimodal/audioUtils.cpp:288,352`, called from `audioRunner.cpp`)
     builds a **bidirectional block-diagonal** attention mask spanning all
     chunks. Crucially: **inside each block the attention is bidirectional,
     not causal**. This is what kills the original P2 "chunk-causal mask"
     plan as a drop-in change.
   - One `enqueueV3` runs the entire encoder over the merged windows.
   - Output `mAudioEmbedding` is `[maxAudioTokens, audioFeatureDim]` —
     allocated once at max size.
4. Encoder output is then handed to the thinker as `audioEmbeddings` on the
   per-request `SpecDecodeInferenceContext` (not on `SlotStreamState`), and
   the thinker is one-shot prefilled via `runBaseModelPrefill →
   executePrefillStep` (`cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:576,
   976, 1117, 1995`). `handleRequest` always wraps prefill with
   `setUpForPrefillExecution → resetForNewSequences`, which is exactly what
   we must bypass for incremental append.
5. `qwen3_asr_worker.cpp:119` calls `captureDecodingCUDAGraph(stream)` — graph
   is captured for the **decode** stage only, encoder shapes are dynamic and
   not graph-captured.

## 4. Key insights that shape the design (post codex review)

(1) The encoder is **already windowed** at the chunk-mask layer
(`audioRunner.cpp:125`), so windowing the *runtime* call is feasible. But
the engine input shape `[num_chunks, mel_bins, n_window*2]` is fixed at
build time (`cpp/multimodal/audioBuilder.cpp:430`) — **streaming requires
an engine rebuild with new shape profiles regardless of which phase we
ship**. The earlier "P0 needs no rebuild" hope is dead.

(2) The encoder has a **3-layer stride-2 conv2d front end** before the
transformer stack (`tensorrt_edgellm/audio_models/qwen3_asr_model.py:111,
117,173,191`). Per-layer KV caching across chunks is **not a drop-in change**:
the conv front end's receptive field crosses chunk boundaries, so naive
caching either drops context or duplicates compute at boundaries. This
deepens the scope of P1.

(3) The intra-block attention is **bidirectional**, not causal
(`audioUtils.cpp:288,352`). Switching to chunk-causal at inference time
without fine-tuning is a likely WER cliff. P2 should be treated as a
**fine-tune-required** phase, not "flip a mask flag".

(4) **Incremental thinker prefill is achievable** as an additive
~150–200 line method that calls `executePrefillStep` while skipping the
`setUpForPrefillExecution` reset (`llmInferenceSpecDecodeRuntime.cpp:576,
976,1117,1995`; engine runner `llmEngineRunner.cpp:1236,1325`). This is
the cheapest piece of the puzzle.

(5) The thinker engine's `max_input_len` default is `128`
(`scripts/build_qwen3_asr_thinker_mlp_int8.py:253`). Any single-chunk
embedding-delta longer than 128 tokens overflows the engine profile.
**Thinker engine must be rebuilt** with a larger `max_input_len` (e.g.
512) before P0 can ship.

Summary of irreducible work for P0:
- Rebuild audio encoder engine with new shape profile (`T_chunk` dynamic,
  optional `K_left` mel-frame prefix).
- Rebuild thinker engine with `max_input_len ≥ 512`.
- Add `appendPrefillEmbeds` to runtime (additive method).
- Add session table + streaming worker protocol.
- Add MRope per-chunk offset bookkeeping (Section 5 expanded).
- Decision: defer encoder KV cache to P1, defer chunk-causal mask to P2.

## 5. Architecture: four-layer change

```
client ──ws── jetson-voice ──ipc── asr_worker ──── EdgeLLM runtime
                                                       ├── MultimodalRunner (encoder + cache)
                                                       └── thinker (incremental prefill)
```

### Layer 1 — Encoder: chunk-causal attention + sliding-window left context

**ONNX surface change (export-time):**
- Inputs add:
  - `enc_kv_cache_in: [num_layers, 2, K_left_max, hidden]` (or per-layer hidden
    state cache; exact layout TBD from inspecting upstream encoder graph).
  - `enc_cache_seq_lens_in: [1]` int64 — how many of K_left_max are valid.
- Outputs add:
  - `enc_kv_cache_out: same shape as in`
  - `enc_cache_seq_lens_out: [1]`
- Mel input shape becomes `[B, melBins, T_chunk]` with `T_chunk` dynamic over
  a profile of `{min: 1 window, opt: 2 windows, max: 4 windows}`.
- Attention mask construction: each query position in the current chunk attends
  to (i) all positions in the cached left context, (ii) all earlier positions
  in the current chunk (block-causal within chunk, full within window). No
  attention to future positions. This is enforced via mask, not via custom
  kernel — same `createAttentionMaskMergedWindows` style but with a strict
  upper-triangular block for positions > current chunk end.

**Risk to flag for codex:** Qwen3-ASR's audio encoder was trained with full
attention across all windows. Applying a chunk-causal mask at inference time
without retraining will degrade WER. We need codex to:
- Read the upstream Qwen3-ASR encoder definition (likely in
  `tensorrt-edge-llm/scripts/export_qwen3_asr*.py` or the upstream `qwen_asr`
  package) to confirm the attention layout.
- Estimate the expected quality hit. If severe, P0 falls back to
  **"shrink left context but keep within-chunk full attention"**: each chunk
  gets its own encoder pass over `[K_left ⊕ T_chunk]` mel frames with the
  original (non-causal) mask, no encoder KV cache, just mel concatenation.
  Cheaper to ship, accepts O(K_left × T_chunk) recompute per chunk but no
  retraining risk.

**TRT engine rebuild:**
- The audio **encoder engine** is built by C++ `AudioBuilder`
  (`cpp/multimodal/audioBuilder.cpp:379,430`), not a standalone Python
  script. Streaming support must add a code path or build-time flag inside
  `AudioBuilder` that emits the chunked profile.
- Encoder profile shapes (P0, no encoder KV cache):
  - `padded_features`: chunk-count min=1, opt=2, max=4 (over `n_window` mel
    spans). Existing `[num_chunks, mel_bins, n_window*2]` shape is preserved;
    only the `num_chunks` dim becomes dynamic.
  - `padded_mask_indices` / `padded_mask_after_cnn`: matching dynamic dims.
- Thinker engine: rebuild via `scripts/build_qwen3_asr_thinker_mlp_int8.py`
  with `--max-input-len 512` (current default 128 at line 253 will be
  raised; confirm no other shape constants downstream).
- Disable CUDA graph for streaming encoder in P0 (dynamic chunk counts);
  thinker decode graph is unchanged (shape `[B, 1]`, see
  `llmInferenceSpecDecodeRuntime.cpp:1798`).

### Layer 2 — Runtime: streaming session API on `LLMInferenceSpecDecodeRuntime`

Add session-scoped methods alongside the existing `handleRequest`:

```cpp
// New, additive — do not touch handleRequest signature.
class LLMInferenceSpecDecodeRuntime {
public:
    struct AsrSessionHandle { uint64_t id; };

    AsrSessionHandle beginAsrSession(
        LLMGenerationRequest const& initRequest, cudaStream_t stream);

    // Returns partial hypothesis tokens generated so far for this session.
    bool appendAudioChunk(
        AsrSessionHandle session,
        std::span<float const> melChunk,        // host-side, [melBins * T_chunk]
        int32_t tChunkFrames,
        bool isLast,
        std::vector<int32_t>& newTokensOut,     // tokens generated this turn
        cudaStream_t stream);

    bool endAsrSession(AsrSessionHandle session, cudaStream_t stream);
};
```

State per session, stored in a `std::unordered_map<uint64_t, AsrSessionState>`:

- Encoder rolling cache `enc_kv_cache` (device tensor sized for K_left_max).
- Encoder `cache_seq_lens` (single int64 on host).
- Thinker KV cache page set (already managed by EdgeLLM's paged KV; we just
  hold the slot id and never release it until `endAsrSession`).
- Accumulated mel offset, accumulated token count.
- A small bounded queue of decoded tokens not yet returned.

Internal flow of `appendAudioChunk`:

1. Copy mel chunk H→D into a session-local scratch tensor.
2. Run encoder TRT context with `enc_kv_cache_in/out` bound to session cache;
   shape = `[1, melBins, tChunkFrames]`.
3. Take the new encoder output slice and append it to the thinker's prompt
   embedding stream, **without re-prefilling the existing KV cache**. This
   means calling the existing prefill kernel with `start_position = current_pos`
   and only the new `delta_audio_embeddings` rows — EdgeLLM already supports
   incremental prefill for spec-decode draft acceptance
   (`llmInferenceSpecDecodeRuntime.cpp` — codex to confirm exact entry point).
4. Run decode for `max_steps_per_chunk` (config, default 16) or until model
   emits a "wait for more audio" signal. P0 uses a fixed cap; final decode
   only flushes on `isLast=true`.
5. Append decoded tokens to session token buffer, return the delta.

On `isLast=true`:
6. Continue decoding until EOS or `max_total_tokens`.
7. Return all remaining tokens, then free the session.

**Confirmed by codex review:** Incremental append is feasible as a new
~150–200 line additive method `appendPrefillEmbeds(session, delta)` that
calls `executePrefillStep` directly, bypassing `setUpForPrefillExecution`
and `resetForNewSequences`. The engine runner binds `kvcache_start_index`
to the current cache length, so two consecutive prefills with growing
start indices should advance KV correctly
(`cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:976,1995`;
`cpp/runtime/llmEngineRunner.cpp:1236,1325`). Must be validated by spike
(Section 11) before P0 coding starts.

### Layer 3 — Multimodal: streaming AudioRunner

Add a sibling class `Qwen3OmniStreamingAudioRunner : public MultimodalRunner`
(codex review recommended subclass over flag — `Qwen3OmniAudioRunner::preprocess`
at `audioRunner.cpp:202,226,483` is a monolithic method that bundles one-shot
preprocessing + text-token insertion + MRope init, and threading streaming
flags through it raises regression risk for the non-streaming path):

- Accepts a per-call `mel chunk + cache in` and produces `embeddings delta + cache out`.
- Owns per-session scratch tensors (or receives them from the runtime — leaning
  toward "runtime owns sessions, runner is stateless and reentrant" because
  paged-KV ownership is already there).
- For P0: reuses `createChunkwiseAttentionMask` unchanged
  (`audioUtils.cpp:288`). Mask stays bidirectional; only the input mel
  shape changes per call. For P2: add a `causal=true` path (still
  bidirectional within block, causal across blocks) — requires fine-tune
  validation per Section 6.

Selection: `MultimodalRunner::create` (`multimodalRunner.h:93`) decides which
runner based on model type. Add `QWEN3_OMNI_AUDIO_ENCODER_STREAMING` to
`cpp/multimodal/modelTypes.h:34` and dispatch on the engine config's
`"streaming": true` flag.

### Layer 3b — Per-chunk bookkeeping (added per codex review)

These items were missing from v1 and must be in the session state:

- **MRope sequential position offset.** `Qwen3OmniAudioRunner` initializes
  sequential MRope cos/sin cache over the full max position range
  per-request at start (`audioRunner.cpp:526,544`). For streaming, the
  session must track `mrope_position_offset` and either (i) re-initialize
  MRope at session start with a generous upper bound and slice by current
  offset per chunk, or (ii) extend MRope cache in place per chunk. P0 picks
  (i) for simplicity: allocate MRope cache at session-max length once, slice
  by `[0, current_pos)` on each prefill.

- **Audio special-token boundaries.** `textPreprocess` replaces the audio
  placeholder with `<audio_bos>` + N × `<audio_pad>` + `<audio_eos>` tokens
  (`audioRunner.cpp:269`). For streaming, the worker must emit `<audio_bos>`
  exactly once on the first chunk, emit pad tokens incrementally matching
  each chunk's encoder output length, and emit `<audio_eos>` only on
  `isLast=true`. The streaming runner is responsible for this token-side
  bookkeeping; the encoder side does not change.

- **KV cache rollback on chunk failure.** `executePrefillStep` advances
  KV lengths on success. There is no existing rollback API. P0 policy:
  any error mid-session is **fatal** — emit `{"event":"error"}`, free
  the session, force the client to restart. A proper rollback API is
  out of P0 scope.

- **PCM → mel in worker.** The current worker reads precomputed mel from
  a JSON file (`audioRunner.cpp:366`); there is no C++ PCM→mel path in
  the runtime. Streaming must add CPU mel feature extraction in
  `qwen3_asr_worker.cpp` (16 kHz PCM → 128-bin log-mel matching the
  encoder's training preprocessing). Reuse upstream `audioUtils` helpers
  if present; otherwise port the Python mel pipeline. Estimated 1–2 days.

### Layer 4 — Worker + protocol

`native/edgellm_voice_worker/qwen3_asr_worker.cpp` becomes session-aware.
Replace the "one JSON line = one utterance" loop with an event-driven
dispatcher:

```
stdin events (each is one JSON line):
  {"event":"begin","id":"sess123","sample_rate":16000,"params":{...}}
  {"event":"chunk","id":"sess123","pcm_b64":"...","last":false}
  {"event":"chunk","id":"sess123","pcm_b64":"...","last":true}

stdout events:
  {"event":"ready","init_ms":...}
  {"event":"partial","id":"sess123","tokens":[...],"text":"..."}
  {"event":"final","id":"sess123","text":"...","total_ms":...}
  {"event":"error","id":"sess123","error":"..."}
```

- Backward compat: keep the old `{"event":"utterance", ...}` path
  (current behavior) as a fallback so non-streaming clients keep working.
- Mel feature extraction (PCM→mel) moves into the worker, run on CPU per
  chunk (already cheap at ~100ms chunk size; existing code reads features
  from `parseRequestFile` so we need to add a CPU mel pipeline — confirm
  whether `audioUtils.h` exposes one).
- Multiple concurrent sessions: P0 = one session at a time per worker
  process; the worker rejects a second `begin` with an error. P1 = multi-
  session via a session table; bounded by `mLlmMaxBatchSize`.

**jetson-voice side (out of scope for this repo but listed for completeness):**
- New WebSocket endpoint `/v1/asr:stream`.
- The existing profile `multilanguage-qwen-highperf-nx` gets a sibling
  `multilanguage-qwen-highperf-nx-streaming` so non-streaming HTTP keeps
  working unchanged.

## 6. Phasing (revised)

**P0 — End-to-end streaming with chunked re-encode (target: 3–4 weeks)**

Scope:
- Rebuild **audio encoder engine** with dynamic `num_chunks` profile (min=1,
  opt=2, max=4). No encoder KV cache; each chunk's encoder call sees mel
  spanning `[K_left ⊕ T_chunk]` and recomputes the prefix.
- Rebuild **thinker engine** with `--max-input-len 512`.
- Add `appendPrefillEmbeds` to `LLMInferenceSpecDecodeRuntime`.
- Add `Qwen3OmniStreamingAudioRunner` subclass.
- Add streaming session API + session table in worker.
- Add CPU PCM→mel in worker.
- Add MRope per-chunk offset bookkeeping + audio BOS/PAD/EOS boundary logic.
- Single-session per worker process.
- Validates entire plumbing end-to-end and produces a shippable artifact.

Expected latency: first partial ≤ 350 ms on 500 ms first chunk; cost is
~1.5–2× full-utterance encoder compute due to re-encode overlap (acceptable
on Orin Nano because encoder is not the dominant cost — confirm via spike).

**P1 — Encoder KV cache (target: +2–4 weeks after P0, depends on spike)**

Scope:
- ONNX export change to expose per-layer encoder hidden state cache I/O.
- Engine rebuild with cache IO.
- Handle conv front-end boundary: the 3× stride-2 conv2d
  (`qwen3_asr_model.py:111,117,173,191`) compresses 4× in time. Options:
  (a) cache pre-conv mel with overlap region equal to conv receptive field
  (simpler, cache is small); (b) cache post-conv per-layer activations
  (smaller, harder boundary math). Choose after spike.
- Eliminates the P0 re-encode overhead.

**Hard dependency**: a separate design pass + measurement of the conv
boundary effect. If post-conv caching is infeasible, P1 reduces to mel-prefix
caching only — modest win.

**P2 — Chunk-causal attention + encoder fine-tune (target: +4–6 weeks)**

Scope:
- Modify `createChunkwiseAttentionMask` to support cross-block causality.
- Short LoRA / full fine-tune of the encoder with the causal mask on a
  representative ASR corpus.
- Eval gate: WER / LCS-similarity within 1% of P0 on the reproduction
  prompt set; CER on a held-out Chinese ASR set within 0.5% of baseline.
- Only worthwhile if P1 + better first-chunk policy still misses the
  latency target, *and* P0 re-encode cost is the dominant bottleneck.

**Decision gates:**
- After spike (Section 11): go/no-go on P0 architecture.
- After P0 ship: if first-partial latency target met, freeze and defer P1/P2.
- After P1 ship: only proceed to P2 if measurement shows encoder compute is
  the bottleneck. Fine-tune budget is non-trivial.

## 7. Open questions for codex review

(Codex: please read the listed files in the fork and answer each.)

1. **Encoder layer attention layout.** In the upstream Qwen3-ASR encoder
   (find the PyTorch module that generates the ONNX), is the attention across
   windows already structured in a way that admits inference-time chunk-causal
   masking, or is it dense full attention? Cite the module + forward.
2. **Incremental prefill entry point.** Inside
   `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp`, what is the actual function
   that writes prefill K/V into paged cache? Can it be called with a
   `(start_pos, embeddings_delta)` pair without resetting state? If not, what
   would the smallest patch look like?
3. **Encoder KV cacheability.** Is the encoder a stack of standard self-attn
   blocks (so per-layer KV cache works) or does it use cross-attention /
   conformer-style convolution / global-local attention that breaks naive
   caching? Cite the relevant module.
4. **`Qwen3OmniAudioRunner` extension vs. subclass.** Given the runner already
   has window/chunk concepts (audioRunner.cpp:125, 389), is a flag inside the
   existing class cleaner than a new subclass? Risks to existing TTS path
   sharing the engine?
5. **Paged KV ownership for long sessions.** Can a session hold a paged-KV
   slot open across many `appendAudioChunk` calls without colliding with
   other workers? What is the slot release path?
6. **CUDA graph rebuild on shape change.** `captureDecodingCUDAGraph` is
   currently called once at init. If we change decode shapes per chunk
   (we shouldn't — decode shape is `[B, 1]` per step regardless of chunking),
   does anything break? Sanity-check the assumption.
7. **W8A16 plugin compatibility.** The W8A16 thinker plugin is shape-specific
   (per the highperf branch). Does incremental prefill with arbitrary
   `embeddings_delta` length stay within the plugin's supported shape range?
8. **First-chunk latency budget.** Where does the current ~430ms warm
   roundtrip actually go? Provide a rough breakdown so we know which layer to
   optimize first.

## 8. Risks and mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Chunk-causal mask destroys WER | high | P0 avoids it; P2 has fine-tune fallback |
| Incremental prefill not supported by spec-decode runtime | medium | P0 falls back to "full re-prefill per chunk" — slower but correct |
| Encoder KV cache infeasible per architecture | medium | Stay on P0 indefinitely; ship without cache |
| Multi-session contention on paged KV | low | P0 = single session; expand later |
| W8A16 plugin shape mismatch | low | Pin chunk sizes to plugin-supported buckets |
| End-of-utterance detection wrong | low | P0 = client-supplied `last=true` |

## 9. Testing / EVIDENCE plan

Per-layer:
- **Encoder ONNX**: numerical diff between streaming encoder output (sum over
  chunks) and one-shot encoder output on the same audio. Tolerance ≤ 1e-3
  L2 per token. PyTorch reference compared to TRT.
- **Runtime session API**: unit test that calls `beginAsrSession` →
  N × `appendAudioChunk` → `endAsrSession` and compares final token sequence
  to `handleRequest` on the same audio. Must match exactly on greedy decode.
- **Worker protocol**: golden-file test feeding scripted JSON events.
- **End-to-end**: extend `scripts/verify_reproduction.sh` with a streaming
  client that splits each reference WAV into 500ms chunks and asserts:
  - First partial arrives ≤ 350 ms after first chunk send.
  - Final text LCS-similarity vs. one-shot ASR ≥ 0.95.
  - Across 3 reproduction prompts × 3 retries (matching existing tolerance).

Required EVIDENCE on any P0 PR:
- Encoder numerical diff numbers (max + mean L2 per token).
- Latency breakdown table: first-chunk send → first-partial-out, last-chunk
  send → final-out, total compute time vs. one-shot.
- Before/after on `verify_reproduction.sh`.
- `docker logs | grep -iE 'error|crash|fail'` clean.

## 10c-real. Spike B2 synthetic prefill — orin-nx (2026-05-13)

**VERDICT: PASS — bit-exact.**

Wrote a minimal harness `/home/harvest/spike_b2/spike_b2.cpp` (~210 LOC,
md5 source `6e2a69c5...`) on orin-nx that instantiates `LLMEngineRunner`
against the shipped thinker engine
(`/home/harvest/qwen3-models/engines/orin-nx/highperf/asr_thinker_full_fp8embed/llm.engine`,
`max_input_len=128`, `hidden_size=1024`, `vocab_size=151936`) and
compares one-shot vs two-step prefill on the same random FP16 embedding
tensor:

| N1 | N2 | total | max abs diff | argmax match | verdict |
|----|----|-------|--------------|--------------|---------|
| 32 | 32 | 64    | **0.0** (bit-exact) | yes | PASS |
| 64 | 64 | 128 (engine max) | **0.0** (bit-exact) | yes | PASS |
| 20 | 44 | 64 (asymmetric)  | **0.0** (bit-exact) | yes | PASS |

Compared full 151,936-element logits vectors — byte identical. Far
stronger than the 1e-2 FP16 tolerance set in the spike plan.

**Two emergent findings that further reduce P0 scope:**

1. **Position IDs are auto-derived inside the engine.** The persistent
   `RopeCosSinCache` indexes positions as `kvCacheStartIndex + local_row_idx`
   internally (`cpp/runtime/llmEngineRunner.cpp:1247-1264`). Callers do
   **not** pass position IDs per chunk — `kKVCacheStartIndex` is bound
   from `mCacheManager.getKVCacheLengths()` and the engine derives
   per-row positions from it. P0's "per-chunk position offset"
   bookkeeping (mentioned earlier in §Layer 3b for MRope) **reduces to a
   single int per session** — the cumulative kv length, which the cache
   manager already tracks.

2. **`outputLogits` already gates on the last token.** Via
   `kLastTokenIds = contextLength-1`, the engine emits logits only for
   the final row. Streaming "sample only on last chunk" is the natural
   default; no `is_final_chunk` flag plumbing needed for the sampling
   side. Intermediate chunks compute hidden states but the sampling head
   is skipped.

**What this means for P0:**

- TRT plugin internal state across `executePrefillStep` calls: PROVED CLEAN.
- Cache plumbing: PROVED CORRECT (per Spike B).
- Position-ID management: AUTO, removed from P0 scope.
- Final-chunk sampling gating: AUTO, removed from P0 scope.

**Revised P0 LOC estimate (third revision):** **300–450 LOC across 2–3
files** (was 400–600 after Spike B). The remaining real work is the
runtime-layer refactor identified in §10b-real:
- Split `setUpForPrefillExecution` into one-shot vs per-chunk halves.
- Add `audioIndexBase` parameter to `generateMultimodalIndices` + the
  multimodal embedding kernel.
- Expose `appendPrefillEmbeds` as a public method on
  `LLMInferenceSpecDecodeRuntime`.
- Worker session API + protocol.
- C++ PCM→mel.

**Revised P0 schedule (third revision):** **3 weeks** (was 3.5–4 after
Spike B; was 2.5–3 after Spike A). All correctness risk is now retired;
remaining work is mechanical engineering.

## 10b-real. Spike B refactor-scope finding (2026-05-13)

Spike B was attempted on orin-nx against fork HEAD `7ab7f1c`. The agent
stopped early per the additive-only guardrail — and that stop **is** the
answer.

**Engine-runner layer: PLUMBING SOUND.** Two back-to-back
`executePrefillStep` calls correctly produce `kv_length = N1 + N2`:
- `cpp/runtime/llmEngineRunner.cpp:1247–1250` — `kvcache_start_index` is
  bound from `mCacheManager.getKVCacheLengths()` at call time, so the
  second call automatically picks up `N1` as its start.
- `cpp/runtime/llmEngineRunner.cpp:1326` →
  `cpp/runtime/hybridCacheManager.cpp:276–289` — `commitSequenceLength`
  uses `kernel::incrementLengthTensor` (additive), not a replace. The
  §10a finding holds at this layer.

**Runtime layer: NOT additive — refactor required.** Three concrete
blockers, all citation-grounded:

1. **Audio index reset.** `generateMultimodalIndices`
   (`cpp/runtime/llmRuntimeUtils.cpp:518`) restarts `audioIndex = 0` on
   every call. The multimodal embedding kernel
   (`cpp/kernels/embeddingKernels/embeddingKernels.cu:413`) reads
   `audioEmbeds[multimodalIdx * hiddenSize]` directly. A second chunk's
   prefill would re-consume the audio embedding tensor from offset 0
   instead of from `audioOffset = N1_audio_tokens`. **Fix: add
   `audioIndexBase` parameter, plumb through to the kernel.**

2. **`setUpForPrefillExecution` is privately paired with
   `runBaseModelPrefill`.** `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp`
   pairs setup (L1895; clears `context.tokenIds` at L1918; calls
   `resetForNewSequences` at L1995 zeroing cache lengths) with prefill
   (L976) inside the private call chain reached only via the public
   `handleRequest`. Both `runBaseModelPrefill` and
   `setUpForPrefillExecution` are private (header L321, L362). **Fix:
   split setup into "one-shot init" (LoRA / KV-restore / token-id seed)
   vs. "per-chunk init" (no reset), and expose a new public
   `appendPrefillEmbeds` entry point.**

3. **Token-stream split couples to audio-stream split.** Splitting
   `context.tokenIds` mid-`<audio_pad>` block decouples token-slice
   from audio-embedding-slice; the kernel cannot express this with the
   current single-call indexing. **Fix: chunk boundary alignment policy
   — each chunk's audio embedding must be a whole number of audio tokens,
   tokenizer-side insertion of `<audio_bos>` / `<audio_pad>` / `<audio_eos>`
   must match the chunk schedule (see §Layer 3b).**

**Revised P0 LOC estimate:** **400–600 LOC across 3 files**
(`llmInferenceSpecDecodeRuntime.{h,cpp}`, `llmRuntimeUtils.cpp`, possibly
`embeddingKernels.cu`), not 150–200 LOC in 1 file. Plus a new test
harness once headers expose seams.

**Unverified risk that remains:** TRT plugin internal state across two
`executePrefillStep` calls. Cache state is provably correct;
attention/MLP plugin transient state is not. This is the *only*
correctness question Spike B was uniquely positioned to answer, and the
additive-only guardrail prevented us from doing so.

**Revised P0 schedule:** **3.5–4 weeks** (was 2.5–3). The encoder-rebuild
saving from Spike A remains intact (~1 week saved); the runtime refactor
discovery costs ~1.5 weeks back.

## 10a-real. Spike A real-hardware results — orin-nx (2026-05-13)

Ran on orin-nx against the shipped highperf engine
`/home/harvest/qwen3-models/engines/orin-nx/highperf/asr_audio_encoder/audio/audio_encoder.engine`
(md5 of spike binary: `b3cabdbdbda96939cd25605e888fc981`).

Sidecar config: `n_window=50`, `n_window_infer=800`, `num_mel_bins=128`,
`max_source_positions=1500`, `max_batch_size=2`, `dtype=fp16`.

**Engine profile (single profile, p=0):**

| input | dtype | MIN | OPT | MAX |
|---|---|---|---|---|
| `padded_feature` | FP16 | `[1,128,100]` | `[30,128,100]` | `[60,128,100]` |
| `padded_mask_after_cnn_indices` | INT64 | `[1,2]` | `[390,2]` | `[780,2]` |
| `attention_mask` | FP16 | `[1,1]` | `[390,390]` | `[780,780]` |
| `last_hidden_state` (out) | FP16 | — | `[-1,1024]` | — |

So the first dim of `padded_feature` is **window count W** (each window
is 50 frames × 2 = 100 mel timesteps), and post-CNN token count
`T = W × 13` drives the two mask inputs. Profile spans `W ∈ [1, 60]`,
i.e. up to 60 s of audio in one call.

**Per-shape latency (median of 10, 3 warmup, random FP16 input):**

| W | T | median (ms) |
|---|---|---|
| 1 | 1 (MIN sanity) | 7.71 |
| 2 | 26 | 8.31 |
| 5 | 65 | 10.11 |
| 10 | 130 | 13.75 |
| 20 | 260 | 23.60 |
| 30 (OPT) | 390 | 32.11 |
| 60 (MAX) | 780 | 65.15 |

Roughly **1 ms/window steady state + 7 ms fixed cost**. The shipped opt
shape (W=30) clocks the same ~32 ms both at OPT and as the sanity
baseline call.

**Verdict for P0:**

- **Encoder engine rebuild dropped from P0 scope.** The shipped engine
  already accepts `W ∈ [1, 60]`, which covers any streaming chunk policy
  we care about (1-window/1-second up to 60-second buffers).
- **First-chunk latency budget is comfortable.** Even on a generous K_left
  ⊕ T_chunk = 5 windows shape, encoder costs ~10 ms — well under the
  350 ms first-partial target. The dominant first-chunk cost is now
  thinker prefill + first-token decode, not encoder. P0 perf tuning
  should focus there.
- **Encoder re-encode of growing left context is cheap.** Going from
  W=5 (10 ms) to W=10 (14 ms) costs ~4 ms — affordable. Skipping P1
  (encoder KV cache) becomes the right default unless measurement
  forces it. P1 may not be needed at all.

**Hard constraint discovered for the streaming worker:**
- `attention_mask` shape[0] and `padded_mask_after_cnn_indices` shape[0]
  are tied via a **named dim** `num_attention_elems` and must equal the
  post-CNN token count `T = W × 13`, **not the window count W**. Setting
  them to W triggers TRT Error 7 (`CHECK_EQUAL` violation, observed in
  spike v1). The streaming worker / Python preprocessing must compute
  `T = W × 13` and shape both mask inputs accordingly.
- W=60 attention_mask is `780 × 780` FP16 = 1.16 MiB — scales O(T²) but
  fine in practice; not a memory concern.

**P0 schedule revision:** Encoder rebuild was estimated at 1–2 days of
the 3-week P0. Removing it lands P0 at **~2.5 weeks** if Spike B passes.

---

## 10a. Spike code-evidence findings (2026-05-13)

A read-only evidence pass found the following before the runtime spikes:

**Encoder profile is more permissive than feared.** `setupQwen3OmniAudioEncoderProfile`
(`cpp/multimodal/audioBuilder.cpp:407-444`) sets:
- `padded_feature` dynamic over `[minChunks, melBins, nWindowDim] →
  [maxChunks, melBins, nWindowDim]`.
- `minChunks = ceil(100 / nWindowDim) = 1`, `maxChunks = ceil(6000 /
  nWindowDim) = 30` for `n_window=100` (so `nWindowDim=200`).
- `padded_mask_after_cnn_indices` / `attention_mask` scale linearly with
  chunk count: max bound `30 × 13 = 390` (audioBuilder.cpp:421-423).

This means **single-chunk (1-window) shapes are already inside the
existing profile**. Spike A on real hardware may show that the existing
shipped encoder engine accepts streaming-chunk calls **without a rebuild**.
This would shorten P0 by 1–2 days and remove one rebuild dependency.
Caveat: TRT plans select tactics at the *opt* shape (`optChunks ≈ 6` in
current build) — running at chunk-count=1 will work but may be slow. P0
should still plan to rebuild with `opt=1` for latency. Not a correctness
blocker; only a perf tuning step.

**Incremental prefill plumbing path is clean** (`cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:472,578,616,976,1117,1995`,
`cpp/runtime/llmEngineRunner.cpp:1236-1251`):
- `executePrefillStep` reads `kvcache_start_index` directly from
  `mCacheManager.getKVCacheLengths()` — so as long as cache lengths
  advance correctly between calls, append "just works".
- `setUpForPrefillExecution` clears `context.tokenIds`,
  `context.effectivePrefillLengths`, and calls
  `baseCacheManager.resetForNewSequences` (line 1995). It does **not**
  touch `mInputsEmbeds` or `mMultimodalIndices`. A streaming
  `appendPrefillEmbeds` skips lines 1995 and the field clears, prepares
  fresh embeddings for the delta, and dispatches `executePrefillStep`.
- Audio embeddings reach the prefill via `embeddingLookupMultimodal`
  (llmInferenceSpecDecodeRuntime.cpp:1018-1069) — they are injected
  **after** the token-embedding lookup, not as a fixed-size prefill input.
  Length is dynamic per call. This decouples encoder output length from
  any `max_input_len` constraint on the embeddings themselves.

**PCM → mel: confirmed brand-new work.** `audioRunner.cpp:366-380`
explicitly requires precomputed mel from a safetensors file and rejects
anything else. `cpp/multimodal/audioUtils.{h,cpp}` only does
post-mel-spectrogram chunking + masking. **No FFT, no mel filterbank in
the runtime.** P0 must add CPU-side mel extraction in
`qwen3_asr_worker.cpp`. Lean options in order of preference:
1. Port the upstream Qwen3-ASR Python mel pipeline (`numpy` STFT +
   `librosa`-style filterbank) into C++ using existing FFT lib (KissFFT
   is permissive-licensed, ~600 LOC, fits in this repo).
2. Sidecar `python -u` subprocess invoked from the worker. Quickest but
   adds an IPC hop per chunk.
3. cuFFT + custom mel filterbank kernel. Fastest, more work.

P0 picks option 1. The cost on CPU is dominated by FFT — at 16 kHz, 25 ms
window, 10 ms hop, ~100 ms chunks contain ~10 windows. KissFFT 400-point
FFT is sub-millisecond per window on Orin Nano CPU, so end-to-end
per-chunk mel extraction stays well under 10 ms. Not a latency concern.

**Implication for P0 scope and schedule:**
- Encoder engine rebuild: optional, perf-only (downgrade from "mandatory").
- Thinker engine rebuild with `--max-input-len 512`: still mandatory.
- `appendPrefillEmbeds`: confirmed ~150–200 LOC additive method.
- C++ mel extraction: new work, ~3–4 days including unit tests.

Revised P0 estimate: **3 weeks** (was 3–4) if real-hardware spike A
confirms the existing encoder engine handles single-chunk shapes.

## 10. Pre-P0 spike status

Per codex recommendation, two read-only experiments must pass before P0
coding begins. Both run in throwaway test binaries, no edits to
production code.

**Spike A — Encoder shape profile validation — ✅ PASS** (2026-05-13)
See §10a-real for raw results. Encoder engine accepts streaming shapes
without rebuild; encoder rebuild is dropped from P0 scope.

**Spike B — Incremental prefill correctness — ✅ PASS via B2 (2026-05-13)**
B stopped early per guardrail and identified a runtime refactor scope
(see §10b-real). B2 (synthetic engine-runner test) then proved the
underlying TRT plugin state is clean across two-step prefill — bit-exact
match against one-shot, see §10c-real. All correctness unknowns retired.

**Original Spike B scope (for record):**
- Construct a single ASR request, run `executePrefillStep` twice on the
  same session: first with embeddings rows `[0, N1)`, second with rows
  `[N1, N1+N2)` and `kvcache_start_index = N1`.
- Compare final decoded tokens (greedy) against a one-shot prefill over
  `[0, N1+N2)`.
- Must match exactly. If not, the gap is the patch P0 must close — either
  position-id offset handling, attention-mask shape, or paged-KV write
  layout. Document the gap, do not paper over it.

Both spikes must report:
- File:line evidence for every claim.
- Raw shape error or token mismatch dumps (no summaries).
- A clear go / no-go on the P0 plan.

## 11. What this design explicitly does not commit to

- A specific WebSocket / SSE wire format on jetson-voice — that gets
  designed alongside the front-end client.
- VAD policy.
- Multi-session worker scheduling (P0 is single-session).
- Any change to the TTS path.
- Encoder retraining (only contingent in P2 if measurement demands it).

## 12. P0 implementation order (per codex v3 re-audit)

The order from earlier sections (worker first, runtime second) is wrong.
**`appendPrefillEmbeds` must be milestone 1**, because no downstream piece
can be trusted until BOS/PAD/EOS slice alignment is verified against
real model output. Revised order:

1. **Milestone 1 — `appendPrefillEmbeds` prototype** (~1 week)
   - On a feature branch off `qwen3-tts-highperf-runtime-w8a16` in the
     EdgeLLM fork, not on the branch itself.
   - Refactor `setUpForPrefillExecution` into `setUpForPrefillExecutionOneShot`
     (existing behavior) + `setUpForPrefillExecutionForChunk` (no
     `resetForNewSequences`, no `context.tokenIds` clear).
   - Add `audioIndexBase` parameter to `generateMultimodalIndices`
     (`cpp/runtime/llmRuntimeUtils.cpp:518`) and the multimodal
     embedding kernel (`cpp/kernels/embeddingKernels/embeddingKernels.cu:413`).
   - Expose `appendPrefillEmbeds(context, audioEmbedsDelta,
     audioIndexBase, tokenSliceDelta)` as a public method on
     `LLMInferenceSpecDecodeRuntime`.
   - **Acceptance test**: drive a real ASR audio sample (precomputed
     mel from reproduction set) through two paths:
     (a) one-shot `handleRequest`.
     (b) two-step: split audio at the boundary between two `<audio_pad>`
         tokens; call `appendPrefillEmbeds` per chunk; call decode at end.
     Compare decoded token sequences — must match exactly under greedy
     decoding.
   - **Side validation**: confirm MRope sequential cache
     (`cpp/multimodal/audioRunner.cpp:526,544`) does NOT need per-chunk
     advancement when initialized once at session start.

2. **Milestone 2 — Session API + KV capacity policy** (~3 days)
   - `beginAsrSession` / `endAsrSession` on the runtime.
   - KV capacity check before each `appendPrefillEmbeds`: refuse new
     chunk if `current_kv_length + chunk_tokens > max_kv_cache_capacity`
     (256 per shipped engine config).
   - Session table in worker. Session-drop / timeout cleanup path.
   - Error/rollback policy (mid-session error → fatal, free session, no
     rollback in P0).

3. **Milestone 3 — Worker streaming protocol** (~3 days)
   - `qwen3_asr_worker.cpp` event-driven dispatcher (begin/chunk/end).
   - Backward-compat one-shot path preserved.
   - jetson-voice WebSocket endpoint (separate repo).

4. **Milestone 4 — C++ PCM → mel with bit-exact parity** (~4–5 days)
   - **Bit-exact reference test first**: dump 1000 samples of
     `WhisperFeatureExtractor` output from the upstream `qwen_asr`
     Python package on a fixed audio set. Save as golden tensors.
   - Port mel extraction to C++ (KissFFT or cuFFT for the FFT). The mel
     filterbank weights must be extracted from the same upstream
     `WhisperFeatureExtractor` (likely shipped as `mel_filters.npz` or
     similar in the model snapshot) — do NOT regenerate them from a
     formula and hope for the best.
   - Acceptance: bit-exact match (or ≤1 ULP) against golden tensors on
     all 1000 samples. ASR end-to-end transcription on the reproduction
     set must match LCS-similarity ≥ 0.95 vs precomputed-mel baseline.

5. **Milestone 5 — End-to-end integration + verify_reproduction.sh
   streaming variant** (~3 days)
   - Streaming client that splits each reproduction WAV into 500ms
     chunks, asserts first-partial ≤ 350ms, final LCS-similarity ≥ 0.95.

Total: ~3 weeks if everything tracks. Add 1 week buffer for milestone 1
unknowns surfaced during the prototype (e.g. if MRope cache turns out
to need per-chunk handling, or if `setUpForPrefillExecution` split is
not as clean as expected).

## 13. Gaps to resolve during P0 (per codex v3 re-audit)

Five items not yet specified in earlier sections — each must have a
concrete decision before its dependent milestone closes:

1. **Timeout / client-drop cleanup** (M2). Policy: per-session
   `last_activity` timestamp; idle > 30s without `chunk` event → force
   `endAsrSession` and free KV slot. Worker emits
   `{"event":"timeout","id":...}` to client (if still connected).

2. **KV capacity check** (M2). Shipped thinker `max_kv_cache_capacity=256`
   (from `asr_thinker_full_fp8embed/config.json`). At ~13 audio tokens
   per 1s window, 256 KV gives **~19.7 s of audio context per session**.
   Worker exposes this limit at `begin` time; client must `last=true`
   before hitting it. P0 enforcement: hard refuse new chunks past
   capacity, emit
   `{"event":"error","error":"kv_capacity_exceeded","kv_length":N,"cap":256}`.

   **Client responsibility (P0):** Qwen3-ASR has no internal Whisper-style
   30 s window — KV accumulates linearly across all `appendPrefillEmbeds`
   calls in a session. The model does NOT auto-segment by commas; commas
   appear only in *decoded text* after the model has consumed *all*
   audio. The 256 KV cap therefore means the client must boundary-mark a
   session before 19.7 s of continuous audio, using one of:
   1. **VAD (recommended)** — ≥ 300 ms silence triggers
      `endAsrSession` → emit current decode → next `beginAsrSession`.
      Natural clause/sentence boundaries (0.3–0.5 s pauses) are caught
      automatically.
   2. **Fixed-duration split** — force a session boundary every N seconds
      (e.g. 10 s). Crude but simple; loses no audio if VAD is unreliable.
   3. **Partial-decode listener** — if streaming partial text is enabled,
      watch for terminal punctuation in the partial and end the session.
      Higher coupling; only useful when partial decode is wired (post-P0).

   If product testing shows 19.7 s is genuinely too short (uncommon —
   typical voice command / dialog is < 8 s), the mitigation is to rebuild
   the thinker engine with `max_kv_cache_capacity = 512` (≈ 39 s,
   ~28 MB extra GPU mem on NX). Sliding-window KV eviction is **P1/P2
   scope only**, not P0.

3. **LoRA / system-prompt cache handling** (M2). The runtime supports
   LoRA weights map and `genAndSaveSystemPromptKVCache`. Streaming
   sessions must not interact with system-prompt KV (no
   `kvCacheLengths` restore between chunks). Audit the
   `setUpForPrefillExecutionForChunk` split to confirm system-prompt
   KV restore is in the one-shot half only.

4. **W8A16 plugin shape policy** (M1). Plugin has fast paths at `m==1`
   and `m>=16`, slow fallback for `m∈[2,15]`. Audio chunks at 1s
   produce ~13 audio tokens per chunk → falls into slow fallback. P0
   policy options: (a) accept the perf hit (chunks dominate by thinker
   prefill, not encoder), (b) batch chunks to ≥16 tokens by buffering
   1.5s before first prefill. Decide after M1 benchmarks.

5. **PCM→mel `WhisperFeatureExtractor` parity** (M4). The real risk
   per codex audit is not the FFT port but matching the exact mel
   filterbank used at training time. Mitigation: golden-tensor test
   from the upstream Python package, bit-exact gate before integration.

## 13b. Audio runner refactor (M3.5 — discovered during M3 dispatch)

The original §13 listed 5 gaps but missed this one. Discovered when the
M3 worker agent tried to encode mel per chunk and found
`Qwen3OmniAudioRunner::preprocess` is a monolithic public entry point
that bundles three steps:

1. `preprocessAudio` — mel → audio embedding (the only piece M3 needs).
2. `textPreprocess` — tokenizes text + splices `<audio_bos>/<audio_pad>/
   <audio_eos>`. The streaming worker already computes its own
   `tokenSliceDelta`; this step is redundant per-chunk.
3. `initializeSequentialMRopeCache(activeBatchSize, ...)` — fills MRope
   cos/sin device cache sized for *this call's* audio token count
   (`cpp/multimodal/audioRunner.cpp:526,544`).

`preprocessAudio` is **private** (`cpp/multimodal/audioRunner.h:111`),
so we cannot call just step (1).

The MRope sub-cache is the load-bearing concern. M1 §10c-real concluded
"MRope sequential cache does NOT need per-chunk advancement *when
initialized once at session start*" — which is correct, but the M1
spike used synthetic random embeddings and bypassed the audio runner
entirely. The base thinker RoPE (`RopeCosSinCache`, engine-internal,
indexed by `kvCacheStartIndex + local_row_idx`) is separate from the
audio MRope cache and was the only one Spike B2 exercised. Real-audio
streaming surfaces the audio MRope cache, and calling `preprocess`
per chunk would overwrite it with a per-chunk-sized cache, breaking
the runtime's cumulative-session view.

**M3.5 deliverables** (target: 1 day on EdgeLLM fork branch
`streaming-asr/m1-append-prefill-embeds`):

1. **Expose mel-only encoder entry point**:
   ```cpp
   bool Qwen3OmniAudioRunner::encodeMelChunk(
       rt::audioUtils::AudioData const& mel,
       rt::Tensor& outEmbedding,
       cudaStream_t stream);
   ```
   Internally calls the existing `preprocessAudio` body. No text
   preprocessing, no MRope init.

2. **Session-scoped MRope init**:
   ```cpp
   bool Qwen3OmniAudioRunner::initializeMRopeForSession(
       int32_t maxAudioTokens,    // bounded by max_kv_cache_capacity = 256
       rt::Tensor& ropeRotaryCosSinDevice,
       cudaStream_t stream);
   ```
   Fills MRope cache sized for the worst-case session length, once,
   at session start.

3. **Wire (2) into runtime**: `LLMInferenceSpecDecodeRuntime::beginAsrSession`
   calls `initializeMRopeForSession(getMaxKvCacheCapacity(), ...)` so the
   worker never has to coordinate MRope. Acceptance preserved bit-exact
   on existing one-shot path.

4. **Acceptance test**: encode same mel one-shot vs three chunks via
   `encodeMelChunk`; concatenate per-chunk outputs; compare against
   one-shot embedding tensor. Tolerance: bit-exact (zero diff). If not
   bit-exact, dump the first-divergence row.

Estimated 80–120 LOC on the fork. After M3.5 lands, M3 resumes as
planned.

## 14. Stale architecture text — superseded by v5 §15

§Layer 1 (encoder), §Layer 2 (runtime), §Layer 3 (multimodal),
§Layer 3b, §10b-real (refactor scope), §13b (audio runner refactor)
all describe paths that assume **chunked encoder inference**. v5
abandons that approach entirely after the empirical LCS test (§Layer 4
findings + cross-reference §16 archived dead paths). Read §15 for
current P0 spec; §1–§14 are design history.

---

## 15. P0 spec (v5) — chunk-and-confirm streaming with prefix prompt

This section supersedes everything above. P0 ships streaming ASR by
**mirroring the official Qwen3-ASR vLLM streaming mechanism** at
`/Users/harvest/project/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:657-770`,
running on the existing EdgeLLM fork without encoder / engine / runtime
changes.

### 15.1 Mechanism

Per streaming session:

```
state = {
    audio_accum: empty PCM buffer,
    raw_decoded: "",               // last full decode output
    chunk_id: 0,
    chunk_size_sec: 0.5,           // tunable; see §15.3
    unfixed_chunk_num: 2,          // first 2 chunks decode without prefix prompt
    unfixed_token_num: 5,          // rollback K tokens for prefix prompt
}

on each `chunk` event:
    state.audio_accum.append(pcm)
    while len(state.audio_accum_since_last_hop) >= chunk_size_samples:
        run_one_hop(state)

on `last` event (or last=true on chunk):
    run_one_hop(state, force=true)
    emit final(parse(state.raw_decoded))
    end_asr_session()


def run_one_hop(state):
    if state.chunk_id < state.unfixed_chunk_num:
        prefix_text = ""
    else:
        tokens = tokenizer.encode(state.raw_decoded)
        prefix_text = tokenizer.decode(tokens[:-state.unfixed_token_num])
        # guard against UTF-8 mid-character truncation: bump K and retry

    prompt = base_asr_prompt + prefix_text
    response = handleRequest(prompt, audio=state.audio_accum)

    state.raw_decoded = prefix_text + response.text
    state.chunk_id += 1
    emit partial(parse(state.raw_decoded))
```

### 15.2 Why this works

- **No information loss**: every hop runs **full one-shot** encoder
  inference on the growing buffer. Encoder sees full bidirectional
  context every time. The empirical LCS test failure mode (premature
  EOS) does not occur here because each hop is a complete, non-chunked
  ASR call.
- **Thinker compute amortized via prefix prompt**: the previous hop's
  decoded text minus tail K tokens is fed as prompt continuation.
  Thinker prefills those tokens (fast — known KV) and only **decodes
  the tail K tokens + any new audio's worth of text**. This is the key
  efficiency optimization in the official implementation.
- **First N hops have no prefix** (`unfixed_chunk_num`): early hypotheses
  are unreliable; better to let the model decode freely than commit to
  bad early text.
- **`unfixed_token_num` rollback**: keeps the last K tokens "unfixed"
  so they can re-decode if more audio gives them better context.
  Trades stability vs commitment.

### 15.3 SLI: end-of-speech latency, not first-partial

**Target SLI**: ≤ 500 ms median, ≤ 1000 ms p95 from `last=true` event
to `final` emission.

Why this is the right SLI:
- The user's perception of "fast" ASR is "how quickly do I see the
  final transcription after I stop talking".
- Partials shown during speech are **already** continuously updating
  (every `chunk_size_sec`); first-partial latency is incidental.
- A 500 ms end-of-speech latency means the system feels responsive at
  conversation boundaries (handoff to LLM / TTS / next turn).

Latency breakdown for a 5 s utterance at hop=500 ms, K=5, on Orin NX:

| Component | Time |
|---|---|
| Unprocessed audio tail (between last hop and `last=true`) | 0–500 ms (avg 250 ms; 0 if client aligns) |
| Encoder forward on full 5 s audio (~5 mel-blocks) | ~10 ms (Spike A) |
| Thinker prefill on prefix text + audio embeddings | ~50–100 ms |
| Thinker decode of last K=5 tokens | ~150 ms |
| **End-of-speech latency** | **~210–760 ms** |

Aligned-client best case (`last=true` immediately after hop fires):
~260 ms. Worst-case (`last=true` arrives right before next hop would
fire): ~760 ms. p95 should fall around 600–700 ms.

### 15.4 First-partial latency (secondary)

Reported for completeness; not a P0 gate.

- Hop 0 (t=hop_ms): from-scratch full decode → first partial at
  ~hop_ms + 300 ms.
- Hop 1 (t=2·hop_ms): also from-scratch (unfixed_chunk_num=2) → second
  partial.
- Hop 2+ (t≥3·hop_ms): prefix prompt active → faster per-hop
  processing.

For hop=500 ms: first partial visible at ~800 ms after start-of-speech.
For hop=1000 ms: ~1.3 s. Both well below human attention threshold for
"partials are alive".

### 15.5 Tunable parameters

| Param | Default | Range | Effect |
|---|---|---|---|
| `chunk_size_sec` (hop) | 0.5 | 0.25–2.0 | Smaller = more compute, faster partials, smaller worst-case end-of-speech tail. |
| `unfixed_chunk_num` | 2 | 1–4 | Higher = more decode freedom early, more compute on early hops, slightly slower partial stabilization. |
| `unfixed_token_num` | 5 | 3–10 | Higher = more late-stage revision, more compute per hop. |
| `max_buffer_sec` | **see formula** | derived | See §15.5.1 below — actual cap is the thinker engine's `max_input_len`, NOT the KV cap. |
| `max_decode_tokens_per_hop` | 64 | 32–128 | Per-hop generation budget to prevent a long-running hop from blocking next partial. If EOS not seen by this cap, emit current partial anyway. |

Defaults chosen for Orin NX 0.6B model based on official defaults
(2.0 / 2 / 5) scaled down for our model size + hardware. **All numeric
timing/length claims in §15.3–§15.4 are estimates pending M5
measurement.**

### 15.5.1 The real buffer cap is `max_input_len`, not KV capacity

Codex audit (v5 pre-implementation review) flagged that earlier drafts
confused two different limits:
- `max_kv_cache_capacity=256` (M2 introduced) — only applies to the
  unused `appendPrefillEmbeds` session path. **Not relevant to P0.**
- **`max_input_len=128`** — actual per-hop prefill input cap, set at
  thinker engine build time (`scripts/build_qwen3_asr_thinker_mlp_int8.py:253`).
  Setup rejects oversize input
  (`cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:1985-1993`).

Per-hop input token budget breakdown:
- Prompt tokens (chat template + system + user role markers): ~30
- Audio special tokens (`<audio_bos>` + `<audio_eos>`): 2
- Prefix-prompt text tokens (rollback of previous decode): up to
  ~3 × `unfixed_token_num` ≈ 15 (variable)
- **Audio embedding tokens (the main consumer): 13 per second of audio**

With current engine `max_input_len=128`:
```
audio_token_budget = 128 - 30 - 2 - 15 ≈ 81
max_buffer_sec ≈ 81 / 13 ≈ 6.2 seconds
```

**Per-session cap: ≈ 5.5 s** (6 s budget with 0.5 s safety margin).
For longer utterances P0 does **transparent auto-segmentation** — the
worker internally rotates sessions without client intervention. Client
sees one final transcript at end-of-speech. Spec in §15.5.2.

**Mitigation path (deferred to post-P0)**: rebuild thinker engine with
`max_input_len=256` or `512`. Each doubling adds ~few MB engine size,
~no runtime memory cost (per-step decode shape is `[B,1]`, unaffected).
This is a **1-day artifact rebuild** (re-run
`scripts/build_qwen3_asr_thinker_mlp_int8.py` with `--max-input-len 256`,
re-package, re-upload HF artifact, update jetson-voice profile pin).
Defer until product telemetry shows the 5.5 s cap is biting badly
(e.g. segment boundary visible in transcript quality on real
utterances).

### 15.5.2 Transparent auto-segmentation

When buffered audio approaches the 5.5 s cap, the worker rotates
sessions **internally**. Client sees one continuous transcription.

Mechanics:
```
on chunk that would push buffer past 5.5 s, OR on internal hop where
predicted next input-token-count would exceed engine cap:

    1. Run a final hop on current buffer → segment_text
    2. Append segment_text to accumulator: full_text += segment_text
    3. Trim audio_accum to last `carryover_sec` (default 0.8 s,
       aligned to mel-block boundary = 80 ms multiples)
    4. Increment segment_id (debug only; not exposed to client)
    5. Reset prefix-prompt state: chunk_id = 0, raw_decoded = ""
       (the new session's hop 0 will run with no prefix, same as
       fresh session)
    6. Continue normally

on last=true:
    final segment hop → segment_text
    full_text += segment_text
    emit {"event":"final","text":full_text}
```

Why this works:
- Each segment runs as a self-contained ASR. Quality within a segment
  matches non-streaming one-shot.
- 0.8 s carry-over gives the new segment enough context to:
  - Detect language correctly (Qwen3-ASR uses early audio for language
    detection)
  - Avoid cutting mid-word audio that would otherwise become unrecognizable
- Concatenation of segment texts works **without overlap dedup** in P0
  because the carry-over is short (0.8 s ≈ 2–4 syllables) and Qwen3-ASR
  is trained to transcribe leading audio without inventing duplicated
  prefix tokens (verified informally; if duplicate prefixes appear in
  measurement, add LCS-based dedup at segment join).

Boundary policy (P0):
- Cut at fixed time when buffer reaches 5.5 s. No silence detection
  (would require VAD; not in P0).
- Cuts may land mid-word. Carry-over recovers most cases but is not
  perfect. Acceptable for product (short utterance commands rarely hit
  this; long dictation users will tolerate occasional boundary
  artifacts).

Boundary metadata (P0):
- Internal segments are NOT exposed to the client. Only the final
  concatenated text emits.
- If product wants to surface segment boundaries (e.g., for retry of a
  specific segment), add `{"event":"segment_final", ...}` events in a
  post-P0 enhancement. Not in scope for P0.

Limit guard:
- Per-hop predicted input length still enforced as in §15.6 step 4.
  Auto-segmentation triggers BEFORE the hard refuse path. Refuse path
  remains as a backstop for pathological clients (e.g. chunks sized
  >5 s each).

### 15.6 Implementation scope

**EdgeLLM fork** (`/Users/harvest/project/tensorrt-edge-llm`, feature
branch `streaming-asr/m1-append-prefill-embeds`):
- **No changes for P0.** Existing `handleRequest` is the only runtime
  entry point needed. M1/M2/M3.5 additions stay on the branch as P1/P2
  foundation but are not active code paths.

**This repo (`qwen3-edgellm-jetson`)**:
- `native/edgellm_voice_worker/qwen3_asr_worker.cpp` —
  event-driven session protocol + per-hop chunk-and-confirm loop +
  prefix-prompt construction. Existing one-shot JSON path preserved
  for backward compat.

**Worker steps** (each one commit; step 2 is a measurement spike, not
a shippable feature):

1. **Streaming protocol scaffold**: stdin/stdout events `begin / chunk
   / end`, single-session state struct, backward-compat for existing
   one-shot path. Emit `session_started`, `session_ended`, plus
   one-shot stubs for `chunk` / `end`. Event dispatch rules (per Codex
   Q7):
   - Line has no `event` field → existing one-shot path (backward
     compat).
   - Line has `event` ∈ `{begin, chunk, end}` → streaming handler.
   - Line has unknown `event` → emit
     `{"event":"error","error":"unknown_event"}`, do NOT silently fall
     through to one-shot.
   - Line fails JSON parse → existing error path.

2. **Measurement spike (NOT a shippable commit)**: no-prefix
   chunk-and-confirm loop. Drive a 5 s reproduction WAV in 500 ms
   chunks. Record per-hop: encoder time, thinker prefill time, thinker
   decode time, total wall-clock. Report median + p95 for hop 0
   through hop 9 (the audio is 10 hops worth). Validates or falsifies
   §15.3 / §15.4 latency estimates BEFORE prefix logic is built.
   Acceptance: hop processing time stays within hop interval (e.g.
   ≤ 500 ms for hop=500 ms) for hops 2+ (where prefix would normally
   kick in). If hop processing > hop interval at any hop, P0 needs a
   smaller `max_decode_tokens_per_hop` or a different default
   `chunk_size_sec`. Output goes into design doc §17 (M5 telemetry,
   to be added).

3. **Prefix prompt rollback**: implement `unfixed_chunk_num` /
   `unfixed_token_num` logic + `_build_text_prompt` exact replication
   (chat template, audio placeholders `<audio_bos>` `<audio_pad>*N`
   `<audio_eos>`, forced-language suffix per `qwen3_asr.py:448-465`).
   Use `state._raw_decoded` (not parsed `state.text`) for prefix
   construction. UTF-8 mid-character guard: implement the per-hop
   retry loop at `qwen3_asr.py:734-743` exactly — bump K and retry
   while decoded prefix contains U+FFFD; on finish, use the
   `max(1, ...)` variant per `qwen3_asr.py:809-816`.

4. **`max_input_len` enforcement + auto-segmentation + session
   cleanup**:
   - Before each `handleRequest`, compute predicted input length
     (audio tokens + prefix tokens + prompt overhead).
   - If projected > 5.5 s of audio (configurable), trigger
     auto-segmentation per §15.5.2: run final hop, save segment text,
     trim audio_accum to last 0.8 s, reset prefix state, continue
     transparently.
   - If projected > engine `max_input_len` (128) AND auto-segment
     can't help (e.g. client sent a single 8 s chunk), refuse with
     `{"event":"error","error":"chunk_too_long","limit_sec":5}`,
     end the session, free state.
   - All error paths must delete temp files AND clear session table
     entry (current worker only deletes temp files,
     `qwen3_asr_worker.cpp:282-290`).

5. **Acceptance test**: drive a real reproduction WAV (target: ≤ 5 s)
   in 500 ms chunks. Measure end-of-speech latency vs `last=true`.
   Compare final text against one-shot `handleRequest` baseline. Gate:
   LCS ≥ 0.95 AND end-of-speech ≤ 500 ms median, ≤ 1000 ms p95. Also
   verify forced-language path produces same shape as one-shot.

**M4 (PCM→mel) still applies** as documented in §12. Without it the
worker can only accept pre-computed mel; for true microphone streaming
PCM→mel must be in-process.

**M5 (E2E verify_reproduction streaming variant) still applies** as
the final gate.

### 15.7 What's removed from P0 vs v4

- No `appendPrefillEmbeds` worker integration (M1 not on critical path).
- No `beginAsrSession` / `endAsrSession` runtime use (M2 not on critical
  path). KV capacity is implicitly handled by `max_buffer_sec` gate.
- No `encodeMelChunk` use (M3.5 not on critical path).
- No conv KV cache work (M3.6 phase A/B/C all abandoned — see §16).
- No M3 worker event protocol with chunked encoder (replaced by §15.6).

The retained-but-unused infrastructure stays on the feature branch as
the foundation for P1 (attention KV cache for middle chunks, latency
optimization) and P2 (chunk-causal encoder fine-tune for sub-500 ms
SLI).

---

## 16. Archived dead paths (do not re-attempt without revisiting v5 decision)

These were explored, found infeasible or unnecessary, and are
documented here so future work doesn't repeat the mistakes.

**16.1 Naive chunked encoder + chunked thinker prefill**
- Tested via M3.5 spike. CNN edge effect blamed (wrong — real cause is
  attention).
- Empirically: LCS 0.37 at 1-block chunks, 0.74 at 4-block chunks.
- Failure mode: premature EOS due to narrowed `cu_seqlens` attention.
- Spike test `spike_m35_audio_runner_split.cpp` retained as a sentinel.

**16.2 Conv KV cache (M3.6 Phase A/B)**
- Implemented in PyTorch (`patch_qwen_asr_conv_cache.py` on wsl2-local).
- POC bit-exact for contiguous-conv reference.
- Discovered: the shipped encoder ONNX **does not use contiguous conv**.
  It pre-chunks input into per-100-mel blocks and convolves each
  independently with zero padding. Conv cache solves a problem that
  does not exist in production.
- Status: patch + ONNX export artifacts left on wsl2-local; do not
  re-export against new model versions without confirming the encoder
  has changed.

**16.3 Attention KV cache (would have been M3.6 Phase B' or P1)**
- Codex confirmed: would only help middle chunks. First chunk has no
  past, last chunk has no future. Architectural limit.
- Not implemented. Re-visit only if P0 ships and middle-chunk recompute
  becomes a measured bottleneck.

**16.4 Overlap-and-discard at attention level**
- Also limited by "no future for first chunk" problem.
- Not implemented.

**16.5 Block-aligned 8-block chunks**
- Theoretically bit-exact if chunks align to attention windows.
- Requires ≥8 s of audio per chunk → defeats streaming goal.
- Not viable for product.

**16.6 Chunk-causal encoder retrain (P2)**
- Codex estimate: LoRA experiment 1–2 weeks; full retrain 4–6 weeks.
- Only path to **true sub-500 ms first-partial latency**. Not P0.
- Re-evaluate if product needs sub-500 ms first-partial (not just
  sub-500 ms end-of-speech).

