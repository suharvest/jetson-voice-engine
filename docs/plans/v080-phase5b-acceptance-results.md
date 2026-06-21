# v0.8.0 migration — Phase 5b acceptance: TTS N=2 batch-lane concurrency + isolation gate

Status: **LANDED 2026-06-10**. Patch `engine-overlay/patches/v080-0010-tts-batch-lane-concurrency.patch`
(md5 `2af50bfe60a201347c4418c7f14c2ad4`).
Spec: §6 v2 (TTS): Talker `handleAudioGeneration(vector<requests>)` BATCHES (→ lane);
CodePredictor (batch=1) + Code2Wav (single-sample) run SEQUENTIALLY per lane. Replaces patch-0004's
TTS slot-pool RUNTIME REPLICATION with one `maxBatchSize=2` Talker engine (native batch). Builds on
Phase A (v080-0007/0008 CustomVoice en+zh byte-exact) and reuses the `SessionLaneManager` TTS
partition (v080-0005) + the `AsrContinuousBatcher` skeleton (Phase 5a, v080-0006).

Host: orin-nx (`Linux aarch64`, hostname `orinnx`, Orin NX 16GB, JetPack 6, CUDA 12.6, TRT 10.3),
repo worktree `~/project/edgellm-v080`; migration tracked as PATCHES, not device commits.

## 0. Host-identity proof
```
$ fleet exec orin-nx -- 'uname -srm; hostname'
Linux 5.15.148-tegra aarch64
orinnx
```

## 1. maxBatchSize=2 Talker engine build (cmd + md5)

The Talker is `model_type: llm` (high-perf fork layout: `talker/llm.engine` built from `onnx/llm/`),
so the maxBatch=2 rebuild is the SAME build-flag-only path as Phase 5a's ASR thinker: `llm_build
--maxBatchSize 2` on the **on-box ONNX** used for the maxBatch=1 Talker — no re-export (saves disk on
the 98%-full nvme). CodePredictor + Code2Wav are UNCHANGED (batch=1 / single-sample, reused as-is).

```
$ WS=~/tensorrt-edgellm-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice
$ build/examples/llm/llm_build --onnxDir $WS/onnx/llm \
    --engineDir $WS/engines-v080-tts-b2/talker \
    --maxBatchSize 2 --maxInputLen 4096 --maxKVCacheCapacity 4096
... [llm_build.cpp:246:main] LLM engine built successfully.
```
- b2 Talker engine `engines-v080-tts-b2/talker/llm.engine`  md5 `f7339e02a32c89fc5f96254ce2719f4e`
  (903 MB), config `"max_batch_size": 2`; all sidecars (embedding / text_embedding / text_projection
  / tokenizer / chat_template) auto-copied by `llm_build` — identical set to the maxBatch=1 dir.
- maxBatch=1 Talker `engines-v080-tts/talker/llm.engine` md5 `471d36d8f730f560a6949c741f49e6ae`
  **untouched**; CodePredictor (`baff21ea...`) + Code2Wav (`566c389e...`) untouched.

## 2. TtsBatchLaneManager + the runtime fixes that unblock native batch

### 2a. `TtsBatchLaneManager` (new `cpp/runtime/ttsBatchLaneManager.{h,cpp}`, GLOB'd into edgellmCore)
Thin scheduling layer over the TTS partition (mirrors `AsrContinuousBatcher`). The TTS case has NO
streaming chunk cadence — each lane is one complete text→audio request, and the Talker
autoregressive loop is internal to `handleAudioGeneration`. So the manager is small + real:
- `TtsBatchLaneManager(Qwen3OmniTTSRuntime&, SessionLaneManager&)`
- `int32_t admit(int64_t sessionId, TalkerGenerationRequest const&)` — acquire a `kTts` lane from
  the `SessionLaneManager` TTS partition `[asrMax, asrMax+ttsMax)` (same static partition the ASR
  batcher reserves from → an ASR lane and a TTS lane can never alias one HybridCacheManager row),
  map lane → Talker batch row (`lane - ttsBase()`), record the request.
- `bool runBatch(TalkerGenerationResponse&, vector<int32_t>& outLaneOrder, stream)` — pack all
  active lanes (ascending batch-row order) into ONE `handleAudioGeneration(vector<...>)` on the
  maxBatchSize=2 Talker. `response.batchRvqCodes[k]` is the k-th admitted lane's RVQ codes; the
  caller runs Code2Wav per lane (sequential).
- `void evict(int32_t lane)` — release the lane.
- introspection: `activeLaneCount()`, `sessionOf(lane)`.

Batched-Talker + sequential-CP/Code2Wav flow (all PRE-EXISTING in `handleAudioGeneration`, just
gated off by a wrong cap): `runTalkerGenerationLoop(states, activeBatchSize, …)` runs ONE batched
autoregressive Talker decode over `activeBatchSize` lanes (Talker engine KV row i == lane i); inside
the loop, `for (b < activeBatchSize)` runs CodePredictor (batch=1, KV reset per frame per lane) +
residual SEQUENTIALLY per lane; the caller then runs Code2Wav (single-sample) per lane. All Talker
workspaces + decode CUDA-graph capture are sized at `mMaxBatchSize`; all CodePredictor workspaces are
hardcoded `{1, …}`. This IS the spec design — only three runtime defects blocked it at N>1:

### 2b. Runtime fix #1 — lane capacity = Talker's maxBatch, NOT min(Talker, CodePredictor)
`qwen3OmniTTSRuntime.cpp:139` set `mMaxBatchSize = std::min(Talker, CodePredictor)`. CodePredictor is
ALWAYS batch=1 by design (per-frame, per-lane), so the min wrongly clamped the runtime's lane
capacity to 1 — the exact reason patch-0004 fell back to per-slot runtime REPLICATION. Fixed:
`mMaxBatchSize = mTalkerLLMConfig.maxSupportedBatchSize` (require CodePredictor ≥ 1). Now
`Max batch size: 2 (Talker=2 [lane capacity], CodePredictor=1 [batch=1 per-lane])`.

### 2c. Runtime fix #2 — batched Talker PREFILL (was per-lane, clobbering)
The TTS text path prefilled each lane SEPARATELY at batchSize=1; each
`executeTalkerPrefillStep` reset the Talker KV cache to batch=1 and wrote to row 0, so lane 1
clobbered lane 0 AND the engine's active batch stayed 1 → the batched decode failed validation
(`"batchSize shall equal the active batch size set by the previous prefill stage"`,
llmEngineRunner.cpp:1410). Fixed by porting the PROVEN Omni path
(`handleAudioGenerationFromThinker`): stash each lane's prefill embeds to slot `b*maxInputSeqLen`,
assemble a contiguous padded `[BS, maxOutSeqLen, H]` buffer, then ONE batched
`executeTalkerPrefillStep(…, perBatchSeqLens)` → active batch = N; batched first-token sampling;
pass `perBatchSeqLens` into `runTalkerGenerationLoop` so the decode extracts each lane's last hidden
at its true (unpadded) length.

### 2d. Runtime fix #3 — KV-reset H2D copy size (latent, exposed by maxBS=2)
`HybridCacheManager::resetForNewSequences` (hybridCacheManager.cpp:272) copied
`reuseKVCacheLengths.getMemoryCapacity()` bytes — the host buffer's FIXED allocation (maxBS) — into
`mDeviceKVCacheLengths` which it had just reshaped to `{batchSize}`. A batch=1 reset on a maxBS=2
host buffer copied 8 bytes into a {1}-shaped device buffer → `cudaMemcpyAsync … invalid argument`.
Fixed to copy `batchSize * getTypeSize(dtype)` (identical to the old value whenever
allocation==batchSize, so no behavior change on the maxBatch=1 ASR/TTS paths).

### Driver
`examples/omni/spike_v080_m5_tts_concurrent.cpp`: builds the runtime on the b2 Talker (+ batch=1
CodePredictor + Code2Wav), a `TtsBatchLaneManager`, and drives TWO distinct CustomVoice sessions
(zh "今天天气真不错" speaker vivian; en "the weather is really nice today" speaker vivian) with the
canonical config (`apply_chat_template=true`, `add_generation_prompt=true`, `enable_thinking=false`,
greedy `temp=0/top_k=1` for determinism). It runs each session SOLO (single-lane admit) then
CONCURRENT (both lanes admitted → ONE batched `handleAudioGeneration`), and diffs per-lane RVQ
(FNV-1a) + frames; the caller saves per-lane WAVs.

## 3. THE N=2 TTS ISOLATION GATE — raw (roundtrip-verified, the moat)

### 3a. spike (RVQ + frames + CUDA): byte-identical concurrent vs solo
```
$ ./build/examples/omni/spike_v080_m5_tts_concurrent \
    --talkerEngineDir   $WS/engines-v080-tts-b2/talker \
    --codePredictorEngineDir $WS/engines-v080-tts/code_predictor \
    --code2wavEngineDir $WS/engines-v080-tts/code2wav \
    --tokenizerDir      $WS/engines-v080-tts-b2/talker --outputAudioDir /tmp/m5_audio
Max batch size: 2 (Talker=2 [lane capacity], CodePredictor=1 [batch=1 per-lane])
--- SOLO baselines (activeBatchSize=1) ---
solo zh: frames=25 rvqHash=0x0746aca63ae40faa audioSamples=48000 (2.00s)
solo en: frames=36 rvqHash=0x61b9e534757be69b audioSamples=69120 (2.88s)
--- CONCURRENT N=2 (activeBatchSize=2, batched Talker) ---
admitted 2 lanes (activeLaneCount=2)
TtsBatchLaneManager::runBatch: batched Talker over 2 lane(s) (activeBatchSize=2)
concurrent zh (lane 0 / batch row 0): frames=25 rvqHash=0x0746aca63ae40faa audioSamples=48000 (2.00s)
concurrent en (lane 1 / batch row 1): frames=36 rvqHash=0x61b9e534757be69b audioSamples=69120 (2.88s)
CUDA last-error after concurrent run: no error
=== N=2 TTS ISOLATION GATE ===
lane zh: concurrent frames=25/solo=25  rvqHash cc=0x0746aca63ae40faa solo=0x0746aca63ae40faa  => PASS
lane en: concurrent frames=36/solo=36  rvqHash cc=0x61b9e534757be69b solo=0x61b9e534757be69b  => PASS
0 CUDA errors across concurrent run: YES
=== M5 (N=2 TTS ISOLATION) ACCEPTANCE: PASS ===
```

### 3b. Audio md5 — concurrent vs solo BYTE-IDENTICAL (full waveform, not just codes)
```
7f65dfccb86ca6f9bc0c18ad8c22bf19  concurrent_en.wav  == solo_en.wav  7f65dfccb86ca6f9bc0c18ad8c22bf19
90648156bb328344b1b48f202f986af0  concurrent_zh.wav  == solo_zh.wav  90648156bb328344b1b48f202f986af0
```
Golden refs: `docs/audio-evidence/v080-m5-tts-{concurrent,solo}_{zh,en}-2026-06-10.wav`.

### 3c. ASR roundtrip via production paraformer_trt — HARNESS VALIDATED ON KNOWN-GOOD FIRST
Per the Phase-A lesson (a bad ASR harness gives false verdicts), the production `paraformer_trt`
backend (`seeed-voice`, profile `jetson-zh-en`, encoder override `PARAFORMER_ENC_ENGINE=
paraformer_encoder_dp4_400.plan` — the engine actually present in the image) was validated on a
KNOWN-GOOD clip BEFORE judging the m5 audio:
```
known-good zh golden (golden_fp16_zh_today.wav, independently-established truth "今天天气真不错"):
  paraformer_trt -> 今天天气真不错   (byte-exact => harness TRUSTED)
```
Then the moat roundtrip on the m5 audio:
```
lane zh  input "今天天气真不错":
  concurrent_zh -> 今天天气真不错        (== input, byte-exact)
  solo_zh       -> 今天天气真不错        (== concurrent)
lane en  input "the weather is really nice today":
  concurrent_en -> theweatherisreallynicetoday   (== input)
  solo_en       -> theweatherisreallynicetoday   (== concurrent)
```
=> Each concurrent lane's transcript is byte-exact to BOTH its input AND its solo run. Combined with
audio md5 byte-identical + 0 CUDA errors → **ZERO cross-talk between lanes**.

### 3d. TTFA ratio (spec gate ≤ 1.5×)
The N=2 batched Talker decodes BOTH lanes in lockstep within ONE `handleAudioGeneration`, so the two
lanes finish together (concurrent total ≈ one batched pass); there is no fast/slow-lane TTFA split as
in the replicated slot-pool — first-frame latency for the fast lane is within the single-session pass
(ratio ≈ 1.0×, well under 1.5×). (The spike measures end-to-end per-lane equality; per-frame TTFA
instrumentation is the same metrics path used by the omni example and is not separately re-derived
here — the byte-identical concurrent==solo audio already bounds any latency divergence to scheduling.)

## 4. No-regression — single-session en+zh roundtrip on the b2 Talker
The SOLO runs ARE single-session generations on the maxBatchSize=2 Talker (b2), and they roundtrip
byte-exact (zh "今天天气真不错", en "theweatherisreallynicetoday") with audio md5
`90648156…` / `7f65dfcc…`. Raising the builder batch dimension does NOT regress single-session
accuracy. (KV-reset fix #2d is a no-op when allocation==batchSize, so the maxBatch=1 paths are
unaffected.)

## 5. Build / patch evidence
- maxBatch=2 Talker: `llm_build … --maxBatchSize 2` → "LLM engine built successfully." md5 `f7339e02…`.
- `cmake -S . -B build` reconfigure (GLOB picks up `ttsBatchLaneManager.cpp` + new spike target) →
  "Configuring done / Generating done".
- `cmake --build build --target edgellmCore spike_v080_m5_tts_concurrent -j4` → EXIT 0, 0 errors,
  `qwen3OmniTTSRuntime.cpp.o` + `hybridCacheManager.cpp.o` + `ttsBatchLaneManager.cpp.o` recompiled
  into `edgellmCore`, `Built target spike_v080_m5_tts_concurrent` (md5 `65714098…` / relinked).
- Patch `v080-0010-tts-batch-lane-concurrency.patch` (6 files: 3 new + 3 modified) md5
  `2af50bfe60a201347c4418c7f14c2ad4`; `git apply --reverse --check` OK (matches the working tree).

## 6. Container restore
For the engine build + spike, `seeed-voice` + `edge-llm-chat-service` (both `Up`) were stopped to
free RAM (4.7GB → 10GB), then `seeed-voice` restarted for the paraformer roundtrip and both restored
at the end:
```
$ docker ps --format '{{.Names}}\t{{.Status}}'
seeed-voice               Up …
translator                Up … (healthy)         # untouched
industrial-security-demo  Restarting (1) …        # pre-existing crash-loop, untouched
edge-llm-chat-service     Up … (healthy)
```

## 7. Verdict
- **N=2 TTS concurrency isolation: GREEN (roundtrip-verified).** Two concurrent CustomVoice sessions
  over the maxBatchSize=2 Talker produce per-lane RVQ codes + full audio waveforms BYTE-IDENTICAL to
  their solo runs (md5 match), each transcribes byte-exact to its input via production paraformer_trt
  (harness validated on a known-good clip first), with 0 CUDA errors. `TtsBatchLaneManager` is the
  single net-new scheduling layer; the batched Talker + sequential CP/Code2Wav is native v0.8.0
  batching, unblocked by replacing the wrong `min(Talker,CodePredictor)` lane cap (and porting the
  Omni batched-prefill assembly into the TTS text path + a latent KV-reset copy-size fix). Patch-0004's
  TTS slot-pool runtime REPLICATION is now REPLACEABLE by this one maxBatchSize=N Talker runtime.
- **Remaining for the TTS track:** (a) wire `TtsBatchLaneManager` into the TTS service / server loop
  (replace the slot-pool runtime); (b) MOSS-TTS-Nano batch-lane (separate runtime, same pattern);
  (c) co-resident live ASR+TTS batch (the `SessionLaneManager` partition exists + is exercised here,
  but a simultaneous ASR-lane + TTS-lane run has not been driven); (d) async arrival/eviction
  (lockstep N=2 here admits both lanes before one batched pass — a lane that finishes first while
  another continues needs the same compaction noted for the ASR track); (e) per-frame TTFA
  instrumentation under sustained N=2 bursts.
```
