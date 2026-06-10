# v0.8.0 `qwen3_asr_worker` rework — re-port N>1 concurrency (#9) + true streaming partials (#10)

**Date:** 2026-06-10
**Status:** DESIGN (read-only authored; no engines run, no device touched)
**Scope:** Re-port two capabilities the v0.7.1 worker had wired but the v0.8.0
migration left out of the serving worker. Both runtime building blocks already
exist + are proven in v0.8.0 spikes — the only gap is the serving worker
(`native/edgellm_voice_worker/qwen3_asr_worker.cpp`).

| Task | Capability | v0.8.0 building block (proven) | Worker today |
|---|---|---|---|
| **#9** | N>1 concurrent ASR sessions (slot-pool / lanes) | `SessionLaneManager` + `HybridCacheManager::resetLanes` (v080-0005, lane-isolation gate diff=0.0) | `max_slots:1`, single serial runtime |
| **#10** | true streaming partials (begin/chunk/end → incremental partial) | `AsrStreamingSessionRuntime` begin/append/end + `decodeToTranscript` + `encodeMelChunk` (v080-0012/0001) | `begin/chunk/end` → 501 stub |

### Source-of-truth files (all read-only refs)

- **Re-port source (v0.7.1 worker, both wired):**
  `/Users/harvest/project/seeed-local-voice/third_party/qwen3-edgellm-jetson/native/edgellm_voice_worker/qwen3_asr_worker.cpp`
  (1416 lines; single-session streaming via `LLMInferenceSpecDecodeRuntime`; the N=2
  slot-pool variant lives in this overlay's git history — `git log` commits
  `17f20a8` "single-threaded slot-pool for concurrent ASR sessions" /
  `fd39323` "use shared SlotPool<AsrSlot>").
- **Worker to rework (v0.8.0 one-shot):**
  `native/edgellm_voice_worker/qwen3_asr_worker.cpp` (this overlay; 407 lines).
- **voxedge backend protocol (the contract the worker MUST satisfy):**
  `/Users/harvest/project/voxedge/voxedge/backends/jetson/trt_edge_llm_asr.py` (1168 lines)
  + `worker_io.py` (307 lines).
- **v0.8.0 runtime blocks (overlay patches under `engine-overlay/patches/`):**
  - `v080-0012-asr-streaming-decode-hook.patch` → `AsrStreamingSessionRuntime`
    begin/append/end + `decodeToTranscript`/`getTranscript`; m6 audio-streaming spike
    `examples/llm/spike_v080_m6_audio_streaming.cpp`.
  - `v080-0005-per-lane-kv-reset-and-lane-manager.patch` → `SessionLaneManager`,
    `HybridCacheManager::resetLanes/setLaneLength/maxBatchSize`, lane-managed
    `AsrStreamingSessionRuntime(runtime, laneManager, ownerId)` ctor; m3 lane-isolation
    spike `examples/llm/spike_v080_m3_lane_isolation.cpp`.
  - `v080-0006-asr-continuous-batcher.patch` → `AsrContinuousBatcher` (token-id
    ragged-batch scheduler); m4 spike `examples/llm/spike_v080_m4_concurrent_lanes.cpp`.
  - `v080-0001-asr-audio-chunk-api.patch` → `Qwen3OmniAudioRunner::encodeMelChunk`.
- **N=2 isolation evidence:** `docs/plans/v080-phase4-acceptance-results.md:70,87,93,97`
  (R1 lane-isolation GATE PASS; R2 + cross-session max-abs diff = `0.000000e+00`).
- **Encoder profile facts:** `docs/plans/v080-0019-serve-gate-acceptance.md:13-55`
  (engine rebuilt `[1,128,100]..[30,128,100]`, `minChunks = divUp(minTimeSteps,
  nWindowDim) = 1`).

---

## 1. The worker protocol contract (begin/chunk/end + partial/final + max_slots)

The voxedge backend speaks a **JSON-line, strictly request→single-reply** protocol
over the worker's stdin/stdout. `WorkerIO` demuxes replies by `id`. **Critical
invariant:** `_worker_request` reads exactly ONE reply per sent line and breaks
(`trt_edge_llm_asr.py:480-486` — `for ev in gen: output_data=ev; break`). So every
`begin`/`chunk`/`end` line MUST produce exactly one stdout event tagged with the
caller's `id`. Any unsolicited / extra event for a live `id` is queued but the
caller never reads it (it sits in the per-request `queue.Queue` until cancel/close).

### 1.1 Worker boot

- **Launch (`trt_edge_llm_asr.py:425-449`):**
  `qwen3_asr_worker --engineDir <llm> --multimodalEngineDir <audio> [--max_slots N]
  [--melSettings <json> --melFilters <bin>]`.
  `--max_slots` is emitted **only when N>1** (`:432-436`, main fix `b1cb1a5`); at N=1
  it is omitted for back-compat. `--melSettings/--melFilters` (or env
  `EDGE_LLM_ASR_MEL_{SETTINGS,FILTERS}`) gate PCM input (`pcm_b64` chunks).
- **`ready` line (worker→backend, `:458-465`):** worker must emit
  `{"event":"ready", "init_ms":..., "max_slots":N}` as the **first stdout line**.
  Backend reads it directly (before handing stdout to the `WorkerIO` reader thread).
  Then `WorkerIO(self._worker, concurrency=self._max_slots)` is created
  (`:469`) — its semaphore back-pressures to N in-flight.

### 1.2 Streaming session (`_TRTEdgeLLMStreamingASRStream`, only when `stream_mode=worker`)

`_use_streaming_worker()` is true when `stream_mode ∈ {worker,stream,streaming,
chunk_confirm,prefix}` (`:378-381`). Production default is `accumulate` → the
streaming protocol is currently **dormant** (one-shot path is live). #10 re-enables
the `worker` mode.

- **begin (`:1009-1023`)** — backend sends:
  ```json
  {"event":"begin","id":"<sid hex>","sample_rate":16000,"chunk_size_sec":0.5,
   "unfixed_chunk_num":2,"unfixed_token_num":5,"context":"",
   "force_language":"<lang>"?}    // force_language present only if language != "auto"
  ```
  Worker MUST reply `{"event":"begin_ack","id":"<sid>"}` (anything else → backend
  raises, `:1022-1023`).
- **chunk (`:1025-1054`)** — backend sends cumulative PCM each hop:
  ```json
  {"event":"chunk","id":"<sid>","pcm_b64":"<float32-LE base64, CUMULATIVE>",
   "audio_sec":<float>,"last":<bool>}
  ```
  Worker MUST reply **exactly one** of:
  - `{"event":"partial","id":"<sid>","text":"<transcript-so-far>", ...}` — `last:false`.
    Backend strips the language prefix and stores `_partial_text` (`:1041-1046`).
  - `{"event":"final","id":"<sid>","text":"<full transcript>", ...}` — `last:true`.
    Backend sets `_final_text`, marks `_closed` (`:1047-1053`).
  - `{"event":"segment_rotation","id":"<sid>","carryover_sec":<float>, ...}` —
    worker auto-segmented. Backend trims `_audio_accum` to the last `carryover_sec`
    and continues (`:1036-1040`). NOTE: production v0.8.0 segmentation is driven
    *backend-side* (`_rotate_segment`, `:1065-1091`, clean cut, fresh session); the
    worker `segment_rotation` path is the v0.7.1 in-worker variant and is OPTIONAL
    for the re-port (see §3.5).
- **end (`:end`/`finalize`)** — bare `end` is sent on close when no `last:true` chunk
  was issued; worker replies `{"event":"final",...}` if it has accumulated text else
  `{"event":"end_ack","id":"<sid>"}` (v0.7.1 `handleEnd`, ref worker
  `:1130-1151`). The backend's normal finalize flow uses `last:true` on the final
  chunk, so `final` typically arrives via chunk, not a bare `end`.

### 1.3 Typed errors / saturation (`_classify_worker_response`, `:212-242`)

The worker's error events are mapped to typed exceptions:
- **Pool saturation (`:229-235`, `PoolSaturatedError`, status 4429):** worker emits
  `{"event":"error","ok":false,"error":"pool_saturated","status":4429,"max_slots":N}`
  when an (N+1)th **distinct** session id calls `begin` while all lanes busy. This is
  a fast-fail busy reject — **NOT** a worker fault (does not trigger a restart,
  `:200-202`).
- `no_active_session` / `no active session` → `NoActiveSessionError` (`:236`).
- `session_already_active` / `already active` → `SessionAlreadyActiveError` (`:238`).
- worker EOF mid-request → `WorkerExitError` (`worker_io.py:170-171`).

### 1.4 One-shot path (unchanged, must stay byte-identical)

Any stdin line **without** an `event` field flows to `handleOneShot` (worker
`:362-368`). This is the `{"requests":[...]}` offline path the production
`accumulate` mode uses via `transcribe()`. It is golden (CER 0.0000) and MUST NOT
regress. #9 and #10 add the lane pool + streaming on top; the one-shot path keeps
driving a single serial runtime call.

---

## 2. #9 — N>1 concurrent ASR sessions design (file:line)

### 2.1 What exists vs what the worker does today

The v0.8.0 worker (`native/.../qwen3_asr_worker.cpp:272-281`) constructs ONE
`rt::LLMInferenceRuntime`, advertises `{"max_slots":1}` (`:281`), and `handleBegin`
is a 501 stub (`:176`). The runtime blocks needed for N>1 are present and proven:

- **`SessionLaneManager(maxBatchSize, asrMax, ttsMax)`** — partitions physical KV
  lanes ASR `[0,asrMax)` / TTS `[asrMax, asrMax+ttsMax)`; `acquire(kAsr, ownerId)`
  hands out a free ASR lane (or `-1` when the partition is full); `release(laneId)`
  frees it (v080-0005 `asrStreamingSessionRuntime.h`, ctor `:107`, acquire `:111`/
  impl `:169`, release `:114`/`:205`).
- **`HybridCacheManager::resetLanes(span<int32_t const>, stream)`** — zeroes ONLY
  the listed lane's KV length, no reshape (v080-0005, `hybridCacheManager.cpp`
  impl; gate `docs/plans/v080-phase4-acceptance-results.md:70`).
- **Lane-managed runtime ctor** —
  `AsrStreamingSessionRuntime(LLMInferenceRuntime& runtime,
  SessionLaneManager& laneManager, int64_t ownerId)` (v080-0005 `:146`). It
  `acquire`s its lane inside `beginAsrSession` and `release`s in `endAsrSession`
  (`:250-268`).
- **Runtime lane capacity:** `runtime.maxSessionBatchSize()` returns the engine's
  native batch dim = the physical lane count (v080-0003 spike usage `:188`;
  used as `int32_t const maxBatch = mRuntime.maxSessionBatchSize()` in 0005 `:336`).

### 2.2 Concurrency model — serialized engine steps, parallel sessions

Per migration doc `asr-streaming-v080-migration.md:93`: ASR sessions + TTS slots
sharing ONE `PipelineIO`/runtime require a single `engineExecMutex` held across each
prefill/decode step, because the runtime-owned shared mutable buffers
(`inputsEmbeds`, `mIdsInput`, `hostContextLengths`, `outputLogits`, `mropeCosSin`,
`mDeviceKVCacheLengths`) are corrupted by concurrent engine calls with different
batch shapes. **Therefore N>1 = logical-lane reservation + serialized engine
execution**, NOT truly-parallel engine forwards. Lane reservation + MRope init
happen outside the mutex; only the engine step is serialized. This matches the
proven m4 model and is sufficient for the gate (2 correct, isolated sessions).

Two valid implementations of "N lanes through one worker":

- **(A) `SessionLaneManager` + per-session `AsrStreamingSessionRuntime`** (RECOMMENDED).
  Each active session owns an `AsrStreamingSessionRuntime` bound to its acquired
  lane; all sessions share the single `LLMInferenceRuntime`. A worker-level
  `std::mutex gEngineExecMutex` serializes each `appendChunk`/`decodeToTranscript`
  call. This is the smallest delta and is exactly the m6 single-lane path
  generalized to N lanes via the lane-managed ctor.
- **(B) `AsrContinuousBatcher`** (ragged batched prefill — token-id sessions only).
  `admit(sessionId, promptTokenIds, maxPositions, stream)` → lane;
  `enqueueChunk(lane, chunkTokenIds, isFinalChunk)`; `tick(stream)` runs a ragged
  batched prefill over all lanes with a pending chunk (v080-0006, m4 spike). This
  gives *batched* (not just serialized) prefill but operates on **token-id chunks**
  ("text-only sessions in this scheduler", batcher impl `:168,:191-226`) — it is the
  proof-of-isolation primitive, not the audio path. Use (B) only if true batched
  throughput is later required; (A) is correct + far less worker rewrite.

**Decision: implement (A).** (B) stays available for a future throughput pass.

### 2.3 Worker changes (`native/edgellm_voice_worker/qwen3_asr_worker.cpp`)

Note: the v0.8.0 worker *already* parses `--max_slots` (`:55,:101-106,:121-130`) and
clamps `>=1`; it just ignores it. Concrete edits:

1. **Boot (`:272-281`):** keep the single `rt::LLMInferenceRuntime`. After
   construction, build `auto laneMgr = std::make_unique<SessionLaneManager>(
   runtime->maxSessionBatchSize(), /*asrMax=*/args.maxSlots, /*ttsMax=*/0);`
   Clamp `args.maxSlots = std::min(args.maxSlots, runtime->maxSessionBatchSize())`
   and log if clamped (engine batch dim is the hard ceiling). Emit
   `{"event":"ready","init_ms":...,"max_slots":args.maxSlots}` (replace the
   hardcoded `1` at `:281`).
2. **Session registry:** replace the migration-comment placeholder (`:288-301`) with
   a real `std::unordered_map<std::string, std::unique_ptr<AsrStreamingSessionRuntime>>
   gSessions;` keyed by session id, plus a `std::mutex gEngineExecMutex;`. (Single
   stdin reader thread means the map itself needs no lock; the mutex guards engine
   steps only — required once the lane-managed runtime touches shared `PipelineIO`.)
3. **`handleBegin` (`:176`/`:371-375`):** create an `AsrStreamingSessionRuntime`
   via the lane-managed ctor (`runtime`, `*laneMgr`, `ownerId = hash(sid)`), call
   `beginAsrSession(promptTokenIds, maxPositions, maxAudioTokensPerChunk, stream,
   maxGenLen)` (m6 spike usage, 0012 `:263-266`). If `beginAsrSession` returns false
   because `acquire` yielded `-1` (partition full), emit the
   `pool_saturated`/`status:4429`/`max_slots` event (§1.3) and do NOT register the
   session. Otherwise register in `gSessions[sid]` and reply `begin_ack`.
4. **`handleChunk` (`:376-379`):** `lookupSlot` = `gSessions.find(sid)`; if absent →
   `no_active_session`. Under `gEngineExecMutex`, run the per-chunk encode + append
   + (incremental decode → §3). Lane routing is implicit: the bound
   `AsrStreamingSessionRuntime` already owns its lane.
5. **`handleEnd` (`:380-383`):** under `gEngineExecMutex`, `decodeToTranscript` +
   `getTranscript` (if not already finalized), then `endAsrSession(stream)` (releases
   the lane via `laneMgr->release`), erase `gSessions[sid]`, emit `final`.
6. **Idle sweep (`:301`/`checkIdleTimeout`):** the v0.8.0 stub is a no-op; restore the
   v0.7.1 per-session timeout sweep (ref worker `:1293-1308`) generalized to iterate
   `gSessions`, force-`endAsrSession` + release lanes idle > `kIdleTimeoutMs`, emit a
   `timeout` event per reaped session. This prevents a half-open session from
   permanently holding a lane (a real production hazard — cf. the v2v session-slot
   leak memory).
7. **`--max_slots` env fallback** already wired to `EDGE_LLM_ASR_MAX_CONCURRENT`
   (`:121-130`); leave as-is.

### 2.4 Cross-talk safety

`resetLanes({lane})` provably leaves other lanes' KV length AND bytes untouched
(`docs/plans/v080-phase4-acceptance-results.md:70`). `endAsrSession` releases the lane
and the next `beginAsrSession` reuses it with a clean KV length. The shared-engine
serialized-step model means no two sessions mutate `PipelineIO` simultaneously. Net:
session B's transcript cannot be contaminated by session A.

---

## 3. #10 — Streaming-partials design (file:line) + encoder-window verdict

### 3.1 The streaming runtime maps directly to begin/chunk/end

`AsrStreamingSessionRuntime` (v080-0012, m6 spike `spike_v080_m6_audio_streaming.cpp`)
is the exact mechanism, proven end-to-end to the golden transcript:

```
beginAsrSession(promptTokenIds, maxPositions, maxAudioTokensPerChunk, stream, maxGenLen)
  → appendChunk(textSlice, AudioChunk, isFinal=false, stream)   // per chunk
  → ... (KV accumulates across chunks)
  → appendChunk(textSlice, AudioChunk, isFinal=true, stream)    // final: samples first token
  → decodeToTranscript(stream)   // autoregressive decode to EOS; EVICTS the lane
  → getTranscript()              // the decoded text
  → endAsrSession(stream)
```

- `appendChunk` runs encoder + chunked prefill only; it does NOT decode (m6 spike
  comment 0012 `:13-15`: "Kept separate from appendChunk so the final-chunk prefill
  state stays byte-identical to the prior (decode-less) behavior the parity spikes
  assert"). The first sampled token is produced by the final `appendChunk`
  (`sampleAfterPrefill`).
- `decodeToTranscript` requires `status()==kFinished` (0012 `:49-52`), drives
  `mRuntime.decodeAsrSessionToCompletion(mContext)` then
  `mRuntime.getAsrSessionTranscript(mContext)` (0012 `:60-65`), and **evicts the
  lane** — so any per-lane introspection (`peekKvCacheLength`) must run BEFORE it.

### 3.2 ENCODER-WINDOW VERDICT — [1,30] handles small chunks ONLY with frame-padding

**Verdict: the [1,30] profile does NOT let you feed a sub-1-second chunk directly;
the per-chunk mel MUST be padded to a whole number of encoder windows (multiple of
`kEncoderMelFramesPerChunk = 100` frames). With that padding, single small chunks are
inside the profile. No accumulating-window re-encode is needed.** Rationale, with
file:line:

1. The encoder profile dim that `[1,30]` constrains is **audio chunks**, where one
   chunk = `nWindowDim = n_window*2 = 100` mel frames (`v080-0019:17-19`). The engine
   is built `padded_feature` `[1,128,100]..[30,128,100]` with
   `minChunks = divUp(minTimeSteps=100, 100) = 1` (`v080-0019:27-29`). So min=1 means
   "1 chunk = 100 mel frames = ~1.0s", NOT "1 mel frame".
2. A 0.5s streaming hop is ~50 mel frames — **below** one window. Feeding it raw
   yields shape `[?,128,50]` which violates the `*,*,100` window dim. The v0.7.1
   worker already solved this: `gMelExtractor->compute(pcm, &n_frames,
   kEncoderMelFramesPerChunk)` pads the mel time dim up to a multiple of 100 frames
   (ref worker `:892-896`; `kEncoderMelFramesPerChunk=100` in `mel_extractor.h:25`;
   commit `bcc8846` "pad time dim to multiple of 100 frames for encoder"). The same
   `MelExtractor` + padding is present in this overlay's worker tree
   (`native/.../mel_extractor.h:25`).
3. The migration design independently confirms this is fine:
   `qwen3-asr-streaming-design-2026-05-13.md:675-684` — "Encoder profile is more
   permissive than feared … `minChunks = ceil(100/nWindowDim) = 1` … single-chunk
   (1-window) shapes are already inside the [profile]."

**Consequence for the re-port:** the streaming chunk path feeds **cumulative** PCM
(the backend sends cumulative `pcm_b64` every hop, `trt_edge_llm_asr.py:1026-1033`),
runs `MelExtractor::compute(..., kEncoderMelFramesPerChunk)` to get a window-aligned
mel, and produces a single `AudioChunk`. Because the backend sends cumulative audio,
the worker re-encodes the whole utterance-so-far each hop — i.e. **the accumulating
window IS the protocol** (driven by the backend, not an in-worker re-encode buffer).
The per-chunk encode just needs window-padding, which `MelExtractor` already does.

> Note on the two append models. v0.8.0 `AsrStreamingSessionRuntime.appendChunk`
> accumulates KV **incrementally** (chunk k appends only chunk k's tokens; KV grows).
> The voxedge backend, however, sends *cumulative* PCM per hop. The simplest correct
> re-port is **per-hop one-shot re-decode** (§3.3 option 1): each cumulative hop is a
> fresh begin→appendChunk(final)→decode→partial, mirroring the production
> `accumulate` semantics that are already CER-0.0000. The incremental-KV model
> (§3.3 option 2) is the throughput optimization and needs the cumulative→delta
> conversion; defer it.

### 3.3 Two ways to produce partials

- **Option 1 — cumulative re-decode per hop (RECOMMENDED for first cut).** On each
  non-final `chunk`: build the mel from the cumulative PCM, run a full
  `begin→appendChunk(isFinal=true)→decodeToTranscript→getTranscript` on a transient
  session (or reuse the one-shot runtime path with the cumulative mel), emit
  `{"event":"partial","text":<transcript-so-far>}`. On the `last:true` chunk, do the
  same and emit `final`. This is the v0.7.1 behavior in spirit and is guaranteed to
  converge to == one-shot because the final hop IS a one-shot decode of the full
  audio. Cost: re-encodes/re-decodes cumulative audio each hop (acceptable for the
  gate; the engine caps utterance length via backend `segment_cap_sec`).
- **Option 2 — incremental KV partials (throughput, defer).** Keep one
  `AsrStreamingSessionRuntime` per session across the whole utterance; convert each
  cumulative hop into the *delta* PCM (subtract prior hop's samples), append only the
  delta as a chunk, and call `decodeAsrSessionToCompletion` on the accumulated KV to
  read a partial. This needs (a) backend to send delta (or worker to diff cumulative),
  and (b) a non-evicting "decode-so-far" peek (the current `decodeToTranscript`
  evicts the lane). Out of scope for the first re-port; note as follow-up.

### 3.4 Partials are decode-so-far, NOT the VAD-finalized one-shot

The production `accumulate` path emits exactly ONE final per VAD-finalized utterance
(no intermediate partials). #10's value is emitting partials **before**
end-of-utterance. Under Option 1 each partial is a genuine decode of the
audio-accumulated-so-far (decode-so-far), and the final partial == the one-shot
finalize of the full audio. The gate (§5) asserts both convergence and
"partial-before-end".

### 3.5 In-worker `segment_rotation` is OPTIONAL

Production v0.8.0 does long-audio segmentation **backend-side** (`_rotate_segment`,
`trt_edge_llm_asr.py:1065-1091`: clean cut, commit text, fresh worker session — no
carryover, no boundary dedup). The v0.7.1 worker's in-worker VAD-aligned
`segment_rotation` + `dedupAtBoundary` machinery (ref worker `:802-829,:978-1098`) is
NOT required for the re-port and adds significant surface. **Recommendation: do NOT
re-port in-worker segmentation in #10.** Let the backend cap at `segment_cap_sec`
and drive clean-cut rotation. The worker only needs to correctly handle a fresh
`begin` after the backend rotates (which #9's per-session lifecycle already does).

---

## 4. How #9 and #10 intertwine + implementation order

### 4.1 Combined architecture

```
                       ┌─────────────────────── qwen3_asr_worker (one process) ───────────────────────┐
 voxedge backend       │  stdin reader loop (single thread, line-demux)                                │
 (WorkerIO, Sem(N)) ──▶│    begin(sid) → handleBegin → laneMgr.acquire(kAsr,sid)                       │
                       │                              → AsrStreamingSessionRuntime{runtime, laneMgr}    │
                       │                              → gSessions[sid]; reply begin_ack                 │
                       │    chunk(sid,pcm,last) → handleChunk → [gEngineExecMutex]                      │
                       │         MelExtractor.compute(pcm, pad=100) → AudioChunk                        │
                       │         (Option 1) begin→appendChunk(final)→decodeToTranscript→getTranscript   │
                       │         reply partial / final                                                  │
                       │    end(sid) → handleEnd → decodeToTranscript (if needed) → endAsrSession       │
                       │              → laneMgr.release(lane); erase gSessions[sid]; reply final        │
                       │    idle sweep → reap stale sessions → endAsrSession + release lane             │
                       │    (line w/o "event") → handleOneShot → single serial runtime call (golden)    │
                       └──────────────────────────────────────────────────────────────────────────────┘
   ONE LLMInferenceRuntime · ONE SessionLaneManager(maxBatchSize, asrMax=N, ttsMax=0) · gEngineExecMutex
```

- **Shared by #9 and #10:** the single `LLMInferenceRuntime`, the
  `SessionLaneManager` (lane = KV row), and `gEngineExecMutex` (serializes every
  engine step across all sessions). N concurrent streaming sessions each emit
  partials; their lanes are isolated by `resetLanes`/per-lane begin/release; their
  engine steps are serialized by the mutex.
- The `WorkerIO` semaphore (N) on the backend side + `--max_slots N` on the worker
  side must agree; the `ready.max_slots` echo lets the backend confirm.

### 4.2 Recommended implementation ORDER (strict)

1. **#9 first, on the ACCUMULATE/one-shot path (lowest risk).** Wire
   `SessionLaneManager` + per-session lane acquire/release into `handleBegin/End`,
   but keep `handleChunk` doing a *cumulative one-shot decode per hop* that emits a
   single `final` at `last:true` (no intermediate partials yet). This proves N>1
   isolation + 4429 saturation through the worker without touching the partial
   mechanism. Gate: §5 #9.
2. **#10 next, on top.** Add intermediate `partial` emission (Option 1 cumulative
   re-decode) inside `handleChunk` for `last:false` hops. Gate: §5 #10 (convergence
   to one-shot + partial-before-end).
3. **(Later, separate workstream)** Option 2 incremental-KV partials and/or
   `AsrContinuousBatcher` batched prefill for throughput. Not required for the gates.

This order matches the prompt's hypothesis and the migration doc's risk chain
(`asr-streaming-v080-migration.md:74` "Risk 4 → Risk 1 → Risk 3 → Risk 2"): lane
allocator (R1/R4) before streaming-decode partials.

---

## 5. Gates

### #9 — N>1 concurrency (lane isolation through the worker)

- **G9.1 Correct + isolated:** drive 2 concurrent sessions through ONE worker
  (`--max_slots 2`), each a distinct full utterance; both `final` transcripts MUST
  equal their respective one-shot/golden transcripts (CER 0.0000), with NO cross-talk
  (session A's words never appear in B's output). Mirrors the proven m4/m3 isolation
  (`docs/plans/v080-phase4-acceptance-results.md:70,97` diff `0.0`) but **through the
  worker stdin/stdout protocol**, not the spike harness.
- **G9.2 Saturation, no false 4429 at N=2:** with `--max_slots 2`, two simultaneous
  `begin`s both succeed (`begin_ack`); a 3rd distinct `begin` returns
  `{"error":"pool_saturated","status":4429,"max_slots":2}` and the backend raises
  `PoolSaturatedError` WITHOUT restarting the worker. After one session ends, a new
  `begin` succeeds (lane reuse).
- **G9.3 One-shot untouched:** an `event`-less line still returns the golden
  `{"event":"done","responses":[...]}` byte-identical (modulo `total_ms`).

### #10 — streaming partials

- **G10.1 Convergence:** the sequence of `partial` texts for an utterance converges,
  and the `final` text == the one-shot finalize of the same full audio (CER 0.0000).
- **G10.2 Partial-before-end:** at least one `partial` event is emitted strictly
  before the `last:true` chunk (i.e. before end-of-utterance). Verify the partial
  stream is non-trivial (text grows / stabilizes across hops).
- **G10.3 Combined N×streaming:** run G9.1 with streaming partials enabled — 2
  concurrent streaming sessions each emit converging partials + correct final, lanes
  isolated, no 4429 at N=2.

---

## 6. Biggest risk + rough effort

### Biggest risk

**Shared `PipelineIO` corruption under concurrent engine steps if the
`gEngineExecMutex` boundary is wrong.** The runtime-owned tensors
(`inputsEmbeds/mIdsInput/hostContextLengths/outputLogits/mropeCosSin/
mDeviceKVCacheLengths`) are shared mutable buffers; a concurrent engine call with a
different active batch shape silently corrupts another lane's transcript
(`asr-streaming-v080-migration.md:93`). The single stdin reader thread *naturally*
serializes today, but as soon as any step is offloaded (or the batcher path is later
added) the mutex must wrap the **entire** prefill+decode step, with lane reservation
and MRope init OUTSIDE it. Mitigation: keep the worker single-threaded for the first
cut (the reader loop already is — worker `:306-399`), make `gEngineExecMutex` a
correctness assertion rather than a perf lever, and only revisit when adding true
parallelism (separate runtime/cacheManager/PipelineIO per lane multiplies engine
memory — explicitly out of scope).

Secondary risk: **lane leak on half-open sessions** (begin with no end) permanently
holding a lane → N-session deadlock. Mitigation: the idle-timeout sweep (§2.3 step 6)
is mandatory, not optional (cf. the v2v session-slot-leak memory where a half-open
connection wedged the whole pipeline).

### Rough effort

| Task | Effort | Notes |
|---|---|---|
| **#9** N>1 lanes | **2–3 dev-days** | `SessionLaneManager` + `AsrStreamingSessionRuntime(laneMgr)` ctor already exist; worker delta is the session map, begin/chunk/end routing, 4429 emission, idle sweep, `ready.max_slots`. All against proven blocks. |
| **#10** streaming partials | **2–4 dev-days** | Option 1 (cumulative re-decode) reuses the m6 begin/append/decode path + existing `MelExtractor` 100-frame padding. Most effort is the partial-emission plumbing + convergence gate harness through the worker protocol. Option 2 (incremental KV) deferred. |
| Worker e2e gate harness | **1–2 dev-days** | Scripted 2-session concurrent + streaming-partial driver over the worker stdin/stdout (no device needed to author; runs on-device at gate time). |

**Total first-cut (#9 + #10, Option 1): ~5–9 dev-days**, all on top of proven v0.8.0
runtime blocks. No engine rebuild required (the `[1,30]`/min-1 encoder is already
shipped per `v080-0019`).
