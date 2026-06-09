# v0.8.0 ASR-streaming migration — Phase 4 acceptance (per-lane KV reset + SessionLaneManager)

Status: **LANDED 2026-06-10**. Patch `engine-overlay/patches/v080-0005-per-lane-kv-reset-and-lane-manager.patch`.
Spec: `docs/plans/asr-streaming-v080-migration.md` §5 R1 + §6 v2; work-list #2 in
`engine-overlay/addon/examples/llm/spike_v080_PORT_NOTES.md`.

Host: orin-nx (`Linux aarch64`, hostname `orinnx`), repo worktree `~/project/edgellm-v080`
(worktree of `~/project/TensorRT-Edge-LLM`; migration is tracked as PATCHES, not device commits).
Build: official `cmake --build`, Release. ASR engine `llm.engine` md5 `b133dff24c8aa96ac1679b95e2f97153`
(same engine as Phase 3b).

## 0. Host-identity proof
```
$ fleet exec orin-nx -- 'uname -srm; hostname; ls -d ~/project/edgellm-v080'
Linux 5.15.148-tegra aarch64
orinnx
/home/harvest/project/edgellm-v080
```

## 1. What landed

### HybridCacheManager — 3 new APIs (next to `resetForNewSequences`)
- `void resetLanes(std::vector<int32_t> const& batchIdxs, cudaStream_t)` — per-lane teardown.
  Implementation: for each lane, `cudaMemsetAsync(mDeviceKVCacheLengths.dataPointer<int32_t>() + idx,
  0, sizeof(int32_t), stream)` — a **single-element device memset**, validated against
  `mConfig.maxBatchSize`. **NO reshape.** Sets `mKVCacheAllEmpty = false` (conservative).
- `void setLaneLength(int32_t batchIdx, int32_t length, cudaStream_t)` — single-element H2D
  memcpy (or memset for length 0) into `mDeviceKVCacheLengths[batchIdx]`. No reshape.
- `int32_t maxBatchSize() const noexcept` — returns `mConfig.maxBatchSize`.

**Memset-not-reshape rationale (the R1 correctness core):** `mDeviceKVCacheLengths` is allocated
once at `{mConfig.maxBatchSize}` capacity in the ctor (`hybridCacheManager.cpp:73-76`) and never
freed; `resetForNewSequences` only *reshapes the logical extent* down to `activeBatchSize`. So
element `[batchIdx]` for any `batchIdx < maxBatchSize` is always physically addressable, and a
per-element memset touches exactly that lane's INT32 — never the whole tensor.

**Full-batch `resetForNewSequences` is byte-unchanged** (verified by `diff` against the pristine
file — the patch hunk for `.cpp` only INSERTS the three new methods before the GPU
`commitSequenceLength`; the reset body is untouched). It remains the one-shot reshape path.

### SessionLaneManager (new, in `asrStreamingSessionRuntime.{h,cpp}`)
`vector<LaneRecord>{ownerKind, ownerId, slotId, inUse, kvLength}` + `std::mutex`. Constructed with
`(maxBatchSize, asrMax, ttsMax)` and static partitions: ASR `[0, asrMax)`, TTS `[asrMax,
asrMax+ttsMax)` (`asrMax+ttsMax <= maxBatchSize` enforced). `acquire(LaneOwnerKind, ownerId) →
laneId` (first free lane in the owner's partition, `-1` if full); `release(laneId)`; plus
`setKvLength`/`recordOf`/`ttsBase`/`maxBatchSize` helpers. Thread-safe (internal mutex).

### Wiring
- `LLMInferenceRuntime::resetSessionLane(lane, stream)` → `cacheManagers[0]->resetLanes({lane})`
  (Phase-4 per-lane teardown); `LLMInferenceRuntime::cacheManagerMaxBatchSize()` for partition sizing.
- `AsrStreamingSessionRuntime` gained ctor `(runtime, SessionLaneManager&, ownerId)`:
  `beginAsrSession` acquires an ASR lane (the laneId IS the HybridCacheManager batch row used for
  MRope/prefill); `endAsrSession` calls `resetSessionLane(lane)` (**NOT** full-batch) + `release(lane)`.
  The legacy `(runtime, lane)` ctor keeps the Phase-3 full-batch `resetForSessionEnd` fallback.

## 2. Validation path: OPTION (a) — mechanism-level
Chosen per spec: the on-box v0.8.0 ASR engine was built `maxBatchSize=1` (engine log:
`maxBatch=1`), so it cannot host 2 live lanes for a full 2-session engine run — that belongs to
Phase 5. Option (a) drives the `resetLanes` MECHANISM directly against a hand-built 2-lane
`HybridCacheManager` (1 attention layer, 0 mamba, kHALF, maxBatchSize=2), which proves the
memset-not-reshape isolation cleanly with no engine forward. New driver:
`examples/llm/spike_v080_m3_lane_isolation.cpp`.

## 3. THE R1 LANE-ISOLATION GATE — raw output
```
$ ./build/examples/llm/spike_v080_m3_lane_isolation
[m3] maxBatchSize()=2 (want 2)
[m3] BEFORE resetLanes({0}): len0=5 len1=7  lane1KV[0..3]=bb bb bb bb
[m3] AFTER  resetLanes({0}): len0=0 len1=7  lane1KV[0..3]=bb bb bb bb
[m3] LANE-ISOLATION GATE PASS: lane1 length+bytes UNCHANGED after resetLanes({0}); lane0 length=0; lane0 KV bytes intact; allEmpty=false.
[m3] SessionLaneManager: asr=0,1 (3rd=-1) tts=2 (ttsBase=2)
[m3] SessionLaneManager: release/re-acquire ASR lane -> 0 OK
[m3] ALL PASS
M3_EXIT=0
```
Asserts proven: lane1 KV-length 7→7 (UNCHANGED); lane1 KV bytes byte-equal + all 0xBB (UNCHANGED);
lane0 length →0; lane0 KV bytes intact (length-only reset, not the slab); `getKVCacheAllEmpty()`
false after partial reset. SessionLaneManager: ASR partition `[0,2)` (3rd acquire = -1 full), TTS
lane = 2 (ttsBase 2), freed ASR lane reused.

## 4. No-regression — raw output
```
$ ./build/examples/llm/spike_v080_m1_append_prefill $ENG/llm 64 63
... maxBatch=1 ...
R3 KV continuity: A_kv=64  B_kv1=62(exp 62 or 63, carry-over)  B_kv2=64(exp 64)  => PASS
R2 argmax: A=11  B=11  => MATCH
R2 max-abs logit diff = 0.000000e+00  (threshold 1e-2) => PASS
=== M1 (R2/R3) ACCEPTANCE: PASS ===

$ ./build/examples/llm/spike_v080_m1_append_prefill $ENG/llm 64 32
R3 KV continuity: A_kv=64  B_kv1=31(exp 31 or 32, carry-over)  B_kv2=64(exp 64)  => PASS
R2 argmax: A=11  B=11  => MATCH
R2 max-abs logit diff = 0.000000e+00  (threshold 1e-2) => PASS
=== M1 (R2/R3) ACCEPTANCE: PASS ===

$ ./build/examples/llm/spike_v080_m2_session_lifecycle $ENG/llm
  Scenario 1: max-abs diff between sessions = 0.000000e+00 => PASS
  Scenario 2: PASS   (KV-overflow refusal; successes=2 before refusal, state preserved)
  Scenario 3: PASS   (append-after-end refusal)
  Scenario 1 (session pair / clean teardown):  PASS
  Scenario 2 (KV-overflow refusal):            PASS
  Scenario 3 (append-after-end refusal):       PASS
=== M2 ACCEPTANCE: PASS ===
```
m2 exercises the Phase-3 full-batch fallback (legacy ctor) → confirms it stays byte-identical.

## 5. Build / patch evidence
- `cmake --build build --target edgellmCore` → `DONE_0`, `Built target edgellmCore` (my changed TUs
  compiled + linked, 0 errors).
- `cmake --build build --target spike_v080_m3_lane_isolation spike_v080_m1_append_prefill
  spike_v080_m2_session_lifecycle` → `DONE_0`, all three `Built target ...`, 0 errors.
- Patch `v080-0005-per-lane-kv-reset-and-lane-manager.patch` (8 files: 7 modified + 1 new test)
  md5 `5e3501dcc06669a3ea2cf0a4fcca852c`; `patch -p1 --dry-run` against the pristine originals →
  exit 0 (applies clean).

## 6. Container restore
For the build, `seeed-voice` and `edge-llm-chat-service` (both `Up`) were stopped to free RAM
(`free` available 3.5GB → 11GB), then restarted at the end:
```
$ docker ps --format '{{.Names}}\t{{.Status}}'
seeed-voice               Up About a minute
translator                Up 8 hours (healthy)        # untouched
industrial-security-demo  Restarting (1) ...           # pre-existing crash-loop, untouched
edge-llm-chat-service     Up About a minute (healthy)
```

## 7. Verdict
- **R1 lane isolation: GREEN.** `resetLanes({0})` provably leaves lane 1's KV length AND bytes
  byte-unchanged while zeroing lane 0 — memset-not-reshape confirmed; full-batch reset untouched.
- **Phase 5 (batch-lane concurrency over a maxBatchSize=2 engine): UNBLOCKED.** The isolation
  mechanism Phase 5 depends on is proven; the remaining Phase-5 work is re-exporting/building a
  maxBatchSize=2 ASR engine + a continuous-batch scheduler (spec §6 v2), independent of this gate.
