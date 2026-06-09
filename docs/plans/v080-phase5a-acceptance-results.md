# v0.8.0 ASR-streaming migration — Phase 5a acceptance (batch-lane N=2 concurrency + isolation gate)

Status: **LANDED 2026-06-10**. Patch `engine-overlay/patches/v080-0006-asr-continuous-batcher.patch`
(md5 `c8df497c4f73e08ae8a99660029328ad`).
Spec: `docs/plans/asr-streaming-v080-migration.md` §6 v2 (Batch-Lane Concurrency — the
`AsrContinuousBatcher` design). Builds on the prior phases v080-0001..0005 (audio chunk API,
runtime hooks, spikes, 1-token guard, per-lane reset + `SessionLaneManager`).

Host: orin-nx (`Linux aarch64`, hostname `orinnx`), repo worktree `~/project/edgellm-v080`
(worktree of `~/project/TensorRT-Edge-LLM`; migration is tracked as PATCHES, not device commits).

## 0. Host-identity proof
```
$ fleet exec orin-nx -- 'uname -srm; hostname; ls -d ~/project/edgellm-v080'
Linux 5.15.148-tegra aarch64
orinnx
/home/harvest/project/edgellm-v080
```

## 1. maxBatchSize=2 v0.8.0 qwen3-asr engine build

The maxBatch=2 LLM (thinker) engine was built from the **same on-box ONNX** used for the Phase-3b
maxBatch=1 engine (`onnx-v080/llm`, exported from `Qwen/Qwen3-ASR-0.6B` with the v0.8.0 schema:
`edgellm_version 0.8.0`, `kv_cache_dtype fp16`). `--maxBatchSize` is a `llm_build` *build*-time
flag, not an export flag, so no re-export was needed (saves disk on the 97%-full nvme + several
minutes). Official build entrypoint only (`build/examples/llm/llm_build`, the repo's C++ builder):

```
$ ./build/examples/llm/llm_build \
    --onnxDir   $WS/Qwen3-ASR-0.6B/onnx-v080/llm \
    --engineDir $WS/Qwen3-ASR-0.6B/engines-v080-b2/llm \
    --maxBatchSize 2 --maxInputLen 4096 --maxKVCacheCapacity 4096
... [llm_build.cpp:246:main] LLM engine built successfully.
```

Engine + config (orin-nx):
```
4122dfcc666fe82b8b0cae4b93c97b70  engines-v080-b2/llm/llm.engine   (1.21 GB)
engines-v080-b2/llm/config.json:  "max_batch_size": 2
parseEngineConfig: ... maxBatch=2 maxInputLen=4096 maxKVCapacity=4096
```
(The maxBatch=1 engine `engines-v080/llm/llm.engine` md5 `b133dff24c8aa96ac1679b95e2f97153` is
**untouched**. The audio encoder engine is unchanged and unused by the text-token spike path.)

## 2. AsrContinuousBatcher — the one net-new scheduling layer

New `cpp/runtime/asrContinuousBatcher.{h,cpp}` (auto-picked by the `runtime/*.cpp` GLOB into
`edgellmCore`). It is the only piece v0.8.0 lacks: the native request runtime fixes the admitted
set at `handleRequest` entry and its decode loop only EVICTS finished lanes (never refills
mid-loop), so async streaming sessions cannot ride it. Everything below the batcher is native
v0.8.0 batching reused as-is.

### Signatures
- `AsrContinuousBatcher(LLMInferenceRuntime& runtime, SessionLaneManager& laneManager)`
- `int32_t admit(int64_t sessionId, vector<int32_t> const& promptTokenIds, int32_t maxPositions, stream)`
- `bool enqueueChunk(int32_t lane, vector<int32_t> const& chunkTokenIds, bool isFinalChunk)`
- `int32_t tick(stream)` — ragged batched prefill over all lanes with a pending chunk this tick
- `void evict(int32_t lane, stream)` — per-lane KV reset + lane release
- introspection: `activeBatchSize()`, `statusOf`, `appendedTokensOf`, `chunkCountOf`,
  `peekKvCacheLength(lane)`, `getLogitsForTesting(lane)`

### Object graph
ONE shared `DecodingInferenceContext mContext` (row i == physical KV lane i == MRope row i); one
`LaneRecord` per admitted lane (`lane`, `sessionId`, `status`, `maxPositions`, `appendedTokens`,
`chunkCount`, `pendingTextTail`, per-lane chunk queue); the shared `SessionLaneManager` (ASR
partition `[0,asrMax)`); one internal mutex serializing admit/tick/evict (shared PipelineIO is
mutated by the engine step). NO replicated context / scratch / PipelineIO — true-parallel batch
execution comes from the engine's native batch dimension.

### tick() flow
1. Collect the lanes (ascending == context-row order) that have a pending chunk; **idle lanes are
   NOT packed** (an empty span yields `selectTokenIndices = ctxLen-1 = -1` in StepPreparer and the
   attention path still launches over the padded seq-len → would corrupt the idle lane).
2. Per ready lane, build effective step tokens with the v080-0004 carry-over guard (defer the
   trailing token of a non-final chunk so the next step packs >= 2 tokens).
3. Assert the ready set is the contiguous prefix `{0..k-1}` (row i == span index i; the primitive
   implies row 0). Set `mContext.activeBatchSize = k`.
4. **Batch-wide single-token guard:** refuse the tick iff `max(stepLen)==1 && anyPriorKv` — the
   AttentionPlugin keys chunked-prefill vs decode on the BATCH-WIDE `qInputTensor.shape[1]`
   (= max chunk len), so a mixed `[1,5]` tick is safe (shape[1]=5 keeps the whole batch on the
   chunked-prefill path), but a max-1 tick on non-empty KV would misroute the whole batch to decode.
5. Build `vector<SpanI32>` (one span per ready lane, row order) and call ONE
   `appendPrefillChunk(mContext, spans, audioEmbeds=nullopt, sampleAfterPrefill=anyFinal)` →
   reuses `runBaseModelPrefillChunk` across the active batch (per-lane pack → StepPreparer(kPrefill,
   activeBatchSize) → prefillDims(activeBatchSize,...) → per-lane commitSequenceLength → per-lane
   sample). Commit per-lane state only on success (refusal-safe).

`admit` (re)initializes the shared context to cover all admitted lanes (`initialize(batchSize=N)`
sets activeBatchSize + identity batchIndexMapping), runs the one-shot `setUpAsrSessionPrefill`
once, and inits each lane's MRope rows. `evict` calls `resetSessionLane(lane)` →
`HybridCacheManager::resetLanes({lane})` (single-element memset, no reshape) → other lanes untouched.

### Codex design verdict (file:line-anchored, captured in PORT_NOTES)
Row i <-> KV row i <-> MRope row i alignment confirmed (no indirection in the prefill helper; the
RoPE kernel indexes `cosSinCacheBatchIdx=batchIdx` when batch != 1). Idle lanes must never be
packed. Mixed-length batches dodge the v080-0004 decode-misroute because the plugin's mode decision
is batch-wide (shape[1]).

## 3. THE N=2 ISOLATION GATE — raw output (the moat property)

Driver `examples/llm/spike_v080_m4_concurrent_lanes.cpp`: TWO distinct token streams (seeds 1234 /
9876 → genuinely different KV + logits), text tokens only (isolates the batched prefill seam, no
mel). SOLO baseline per session (single-lane admit), then CONCURRENT (admit A->lane0, B->lane1;
lockstep two-lane ticks at activeBatchSize=2). Per-lane concurrent logits diffed against solo.

```
$ ./build/examples/llm/spike_v080_m4_concurrent_lanes $ENG 64 32
maxSessionBatchSize=2
--- SOLO baselines (activeBatchSize=1) ---
solo A: argmax=11  |logits|=151936
solo B: argmax=315 |logits|=151936
--- CONCURRENT N=2 (activeBatchSize=2, lockstep) ---
admit: laneA=0 laneB=1 activeBatch=2
tick1 advanced=2  kvA=31 kvB=31
tick2 advanced=2  kvA=64 kvB=64
CUDA last-error after concurrent run: no error => CLEAN
=== N=2 ASR ISOLATION GATE ===
lane A(0): argmax cc=11  solo=11  (MATCH)  max-abs diff = 0.000000e+00  => PASS
lane B(1): argmax cc=315 solo=315 (MATCH)  max-abs diff = 0.000000e+00  => PASS
0 CUDA errors across concurrent run: YES
=== M4 (N=2 ISOLATION) ACCEPTANCE: PASS ===
```

Hardened across splits (incl. the 1-token-final carry-over case 64/63 and N=100):
```
N=64 split=63  lane A diff 0.0 (argmax 11/11)   lane B diff 0.0 (315/315)   CUDA CLEAN  PASS
N=64 split=17  lane A diff 0.0 (argmax 11/11)   lane B diff 0.0 (315/315)   CUDA CLEAN  PASS
N=100 split=50 lane A diff 0.0 (argmax 13/13)   lane B diff 0.0 (151645/151645)  CLEAN  PASS
```

=> **Concurrency introduces ZERO cross-talk.** Each of the two concurrent sessions' sampled
logits / argmax are BYTE-IDENTICAL (max-abs diff exactly 0.0) to that session run solo, with 0
CUDA errors. This is the moat property Phase 5a set out to prove.

## 4. No-regression on the maxBatch=2 engine (single-lane subset) — raw output

```
$ ./build/examples/llm/spike_v080_m1_append_prefill $ENG/engines-v080-b2/llm 64 63
parseEngineConfig: ... maxBatch=2 ...
R3 KV continuity: A_kv=64  B_kv1=62(exp 62 or 63, carry-over)  B_kv2=64  => PASS
R2 argmax: A=11  B=11  => MATCH
R2 max-abs logit diff = 0.000000e+00  => PASS
=== M1 (R2/R3) ACCEPTANCE: PASS ===

$ ./build/examples/llm/spike_v080_m1_append_prefill $ENG/engines-v080-b2/llm 64 32
R3 KV continuity: A_kv=64  B_kv1=31  B_kv2=64  => PASS
R2 argmax: A=11  B=11  => MATCH   R2 max-abs logit diff = 0.000000e+00  => PASS
=== M1 (R2/R3) ACCEPTANCE: PASS ===

$ ./build/examples/llm/spike_v080_m2_session_lifecycle $ENG/engines-v080-b2/llm
  Scenario 1: max-abs diff between sessions = 0.000000e+00 => PASS
  Scenario 2 (KV-overflow refusal):            PASS
  Scenario 3 (append-after-end refusal):       PASS
=== M2 ACCEPTANCE: PASS ===
```
The maxBatch=2 engine is single-lane byte-identical to the maxBatch=1 engine — no accuracy
regression from raising the builder batch dimension.

## 5. Build / patch evidence
- `cmake -S . -B build` (re-configure to pick up the new GLOB'd source) → "Configuring done /
  Generating done".
- `cmake --build build --target edgellmCore spike_v080_m4_concurrent_lanes -j4` → `BUILD_EXIT=0`,
  0 errors, `Built target edgellmCore` + `Built target spike_v080_m4_concurrent_lanes`
  (edgellmCore relinked with the new `asrContinuousBatcher.cpp.o`).
- Patch `v080-0006-asr-continuous-batcher.patch` (4 files: 3 new + `examples/llm/CMakeLists.txt`)
  md5 `c8df497c4f73e08ae8a99660029328ad`; `git apply --reverse --check` OK (matches the working
  tree exactly) and `git apply --check` OK after reverse (applies clean to pristine).

## 6. Container restore
For the engine build + validation, `seeed-voice` and `edge-llm-chat-service` (both `Up`) were
stopped to free RAM (available 4.6GB → 12.3GB), then restarted at the end:
```
$ docker ps --format '{{.Names}}\t{{.Status}}'
seeed-voice               Up 15 seconds
translator                Up 8 hours (healthy)        # untouched
industrial-security-demo  Restarting (1) ...           # pre-existing crash-loop, untouched
edge-llm-chat-service     Up 14 seconds (healthy)
```

## 7. Verdict
- **N=2 ASR concurrency isolation: GREEN.** Two concurrent sessions over the maxBatchSize=2 engine
  produce per-lane logits BYTE-IDENTICAL (diff 0.0) to solo runs, 0 CUDA errors, across multiple
  splits. The `AsrContinuousBatcher` is the single net-new layer; it composes the already-validated
  R1/R2/R3 primitives (per-lane reset, chunk-prefill helper, MRope-per-lane) into a true-parallel
  batched-lane scheduler with zero cross-talk.
- **What remains for the ASR track:** (a) a non-prefix / lone-survivor lane needs explicit
  compaction (mirror `performBatchEvict` KV+MRope compaction) before a session that finishes first
  can free row 0 while a higher row continues — out of scope for the lockstep isolation gate but
  required for asynchronous arrival/eviction in production; (b) audio (mel) chunks through the
  batcher (the gate is text-token-only to isolate the prefill seam); (c) wiring the batcher into the
  ASR worker / server loop; (d) co-residency with TTS lanes (the ASR/TTS lane partition exists in
  `SessionLaneManager` but a live ASR+TTS batch has not been run).
- **TTS-migration track is SEPARATE** (spec §6 v2 "TTS" + Note on scope): Talker-batched +
  CodePredictor/Code2Wav sequential, replacing patch-0004's runtime replication with a single
  `maxBatchSize=N` Talker runtime. Independent of this ASR gate.
