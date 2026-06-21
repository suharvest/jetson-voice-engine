# v080-0022 — ASR N>1 + streaming PARTIALs THROUGH the deployed v0.8.0 service (acceptance)

**Date:** 2026-06-10
**Host:** orin-nx (`Linux 5.15.148-tegra aarch64`, hostname `orinnx`, disk 96% / 9.8–11 G free)
**Scope:** Integrate the NEW `qwen3_asr_worker` (#9 N>1 + #10 streaming PARTIALs,
md5 `f9bb821dec9f32e53697c7930254b4f3`) + the maxBatch=2 ASR llm engine into the
v0.8.0 image and verify N>1 + streaming partials **through `/asr/stream`** (not
just the standalone worker — the #9/#10 standalone gates already passed in
v080-0021).

---

## 0. Host + disk (raw)
```
Linux 5.15.148-tegra aarch64
orinnx
/dev/nvme0n1p1  233G  213G  11G  96% /   (9.8 G after staging/cleanup)
```

## 1. HF re-stage (md5-verified, via Mac → huggingface.co; wsl2-local relay proxy was down)
New worker + b2 llm engine uploaded to
`harvestsu/seeed-local-voice-artifacts/sm87-trt10.3-jp6.2/v0.8.0/`:

| HF path | md5 | size |
|---|---|---|
| `workers/qwen3_asr_worker` | `f9bb821dec9f32e53697c7930254b4f3` | 14161784 |
| `engines/asr-b2/llm/llm.engine` (maxBatch=2) | `4122dfcc666fe82b8b0cae4b93c97b70` | 1208622100 |
| `engines/asr-b2/llm/config.json` (`max_batch_size:2`) | `0e6bb1f661484765327b9ed1e7efa385` | 943 |

- The OLD on-HF worker was `be7bee91…` (one-shot); now replaced by `f9bb821d…`.
- The OLD on-HF llm.engine was `b133dff2…` (b1, max_batch_size=1); b2 added at a
  NEW path `engines/asr-b2/llm/` (b1 left in place).
- **Audio encoder already correct on HF:** `engines/asr/audio/config.json` md5
  `3b9ff631075bab6a8e44631fe6fa1c5f` == the box `engines-v080-minchunk1` config,
  `min_time_steps=100` (the [1,30] small-chunk encoder, re-staged in v080-0019).
  No re-stage needed.
- b2 engine dir sidecars (tokenizer/embedding/chat-template) are b1/b2-identical
  and live under `engines/asr/llm/`; the updated pull script stages the b1
  sidecars then overlays the b2 engine+config.

## 2. Image rebuilt
`sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.8.0-edgellm-20260610f`
(thin overlay FROM `:prod-unified-v8`, build EXIT 0). Changes vs `:e`:
- bakes NEW worker (`f9bb821d`),
- adds profile `jetson-edgellm-v080-n2` (`EDGE_LLM_ASR_STREAM_MODE=worker`,
  `asr_max_slots=2`),
- `OVS_PROFILE=jetson-edgellm-v080-n2`, `EDGE_LLM_ASR_ENGINE_DIR=…/asr-b2/llm`,
  operator-owned `EDGE_LLM_ASR_STREAM_MODE=worker` + `EDGE_LLM_ASR_MAX_CONCURRENT=2`
  + mel settings/filters.

Image-baked verification (raw):
```
md5  /opt/jv-workers/qwen3_asr_worker = f9bb821dec9f32e53697c7930254b4f3
PROFILE=jetson-edgellm-v080-n2  STREAM=worker  MAXCONC=2
ENGDIR=/opt/edgellm-v080/engines/asr-b2/llm
```

## 3. THROUGH-SERVICE GATES (throwaway `v080-verify-n2`, port 8646, engines
pre-staged via hardlinks, `EDGELLM_V080_AUTO_PULL=0`)

### Boot — coherent
```
EDGELLM_V080_AUTO_PULL=0 → skipping HF pull
Applied profile jetson-edgellm-v080-n2 (37 env keys; 0 stale cleared)
coordinator: asr.supports_parallel=True/max=2
ASR backend preload OK (engine_dir=…/asr-b2/llm, max_slots=2, stream_mode='worker',
  mel_settings_path/mel_filters_path resolved)
/health → asr:true asr_capabilities:["streaming","offline","multi_language"]
```

### Service-config finding (session admission gate) — FYI
`SessionLimiter effective_limit = min(asr.max_concurrent=2, tts.max_concurrent=1) = 1`.
The global voice-session gate clamps to the TTS concurrency (1). So even with ASR
N=2 wired, only one `/asr/stream` is admitted unless TTS concurrency is also ≥2 OR
the gate is made ASR-endpoint-specific. (The v0.8.0 generic-runner TTS worker is
concurrency=1 by design and rejects `--max_slots` → cannot lift via config without
crashing TTS.) **This is moot below — the worker gap blocks streaming N>1 first.**

### N>1 / streaming through /asr/stream — **FAILS (worker gap)**
With `stream_mode=worker`, the service correctly drives the begin/chunk(pcm_b64)/end
protocol. The worker **aborts on the first real-PCM `chunk`**:
```
ASR stream opened (backend=trt_edgellm)
ASR worker exited before response: worker subprocess died mid-request:
  MmapReader: Cannot open file: …/engines/asr/audio/action/action.engine
qwen3_asr_worker: …/nlohmann/json.hpp:2175: …operator[]… Assertion `it != end()' failed.
```
Result frame: `{type:error}` then empty final (CER 1.0, 0 partials).

**Reproduced STANDALONE** (worker binary direct, same `pcm_b64` protocol, staged
b2 engine + minchunk1 encoder + real mel assets) — isolates it to the worker, not
the service:
```
READY: {"event":"ready","init_ms":6831,"max_slots":2}      ← constructs fine
BEGIN_ACK: {"event":"begin_ack","id":"s1","lane":0}        ← #9 lane reservation OK
(begin A+B, 3rd → {"error":"pool_saturated","status":4429,"max_slots":2})  ← #9 4429 OK
chunk(pcm_b64) → WORKER_DIED rc=-6 (SIGABRT)
  …MmapReader Cannot open …/action/action.engine
  …json.hpp operator[] Assertion `it != end()' failed
```

**Root cause (worker, file:line):**
`native/edgellm_voice_worker/qwen3_asr_worker.cpp:handleChunk` (≈L353) builds a
one-shot request with `content:[{type:audio, pcm_b64:<b64>}]` (see
`buildFinalizeRequest` ≈L386 — its own comment: *"the runtime's mel-extractor
consumes pcm_b64 … when mel assets are wired; mel_path is the primary fixture
path."*) and calls `runOneShotCore` → `rt::LLMInferenceRuntime`. The runtime's
**PCM→mel request path is unwired**: it reaches `LLMInferenceRuntime` action-load
(`cpp/runtime/llmInferenceRuntime.cpp:315` `multimodalEngineDir + "/action"` →
`Alpamayo1ActionRunner` ctor, `cpp/action/alpamayo1ActionRunner.cpp:143`
`MmapReader(engineDir+"/action.engine")`) which **abort()s** via an uncatchable
`MmapReader`/nlohmann `operator[]` assert when `action.engine` is absent (there is
**no `action.engine` anywhere on the box** — Alpamayo is an unrelated driving
expert, never built for ASR). The try/catch at L316–327 cannot catch an `assert`.

**Why #9/#10 passed standalone but fails here:** v080-0021's #10 gate fed
PRE-COMPUTED **mel `.safetensors` via `mel_path`** (the fixture path), which the
runtime loads directly and never enters the PCM mel-extractor / action-load path.
The deployed service supplies **real PCM via `pcm_b64`** (the only thing
`_TRTEdgeLLMStreamingASRStream` can send), which the worker has never actually
exercised.

### Production-safety check — accumulate mode (the deployable path) — **PASS**
Same NEW worker + b2 engine, `EDGE_LLM_ASR_STREAM_MODE=accumulate` (Python
accumulates PCM, one-shot `requests` via mel on finalize — the worker `begin/chunk`
is never sent → never hits the crash):
```
/asr/stream zh_long_01 → "这并不是告别 这是一个篇章的结束，也是新篇章的开始 嗯。"
  ref "这并不是告别，这是一个篇章的结束，也是新篇章的开始。"  CER 0.0435  (clean close, no crash)
```
Logs clean (no abort/assert/501/satisfyProfile). Long-audio (zh_long_02, 13.8 s)
truncated but **did NOT crash** (clean connection close) — pre-existing
accumulate long-audio behavior, not a regression.

## 4. Container restore
- throwaway `v080-verify-n2` `docker rm -f` → removed; verify staging dir removed.
- production `seeed-voice` healthy (`Status=running`, `OOMKilled=false`,
  ExitCode=0, /health 200, ASR=paraformer_trt TTS=matcha_trt). RestartCount=3 =
  CLEAN `restart: unless-stopped` restarts under GPU memory co-residency during the
  two verify boots; recovered fully each time. `translator` + `edge-llm-chat-service`
  Up (healthy). `industrial-security-demo` Restarting = **pre-existing** crash loop
  (already Restarting at task start, untouched). Named volumes `edgellm-v080` +
  `speech-models` never removed.

## 5. Verdict
**Do N>1 + streaming PARTIALs work THROUGH the deployed v0.8.0 service? NO — blocked
by a WORKER gap (not service-config).**

- The service-side wiring is correct and verified: `stream_mode=worker` routes
  `/asr/stream` to the streaming begin/chunk/end protocol; `asr_max_slots=2` →
  `--max_slots 2`; the b2 (maxBatch=2) engine + minchunk1 (min=100) encoder load;
  `pool_saturated→4429` plumbed end-to-end.
- The worker's **#9 N>1 lane reservation + 4429 saturation are sound** (verified
  standalone via the real pcm_b64 protocol).
- The worker's **real-PCM streaming chunk decode is broken**: `handleChunk`'s
  `pcm_b64 → runOneShotCore` path hits an unwired PCM mel-extractor + an
  unconditional Alpamayo `action.engine` load that `abort()`s. Streaming PARTIALs
  through the service are therefore impossible until the worker is fixed.
- A secondary service-config item (session gate `min(asr,tts)=1`) would also need
  lifting for ASR N=2 to be admitted, but it is downstream of the worker fix.

**Production-safe NOW:** the new worker + b2 engine in **accumulate** mode
transcribes correctly through the service (CER 0.04 on the clean case) and does
not crash — so the deployable image can ship accumulate (one-shot per finalize)
without regression; N>1 there is bounded by the same session gate.

### Required worker fix (next task)
In `qwen3_asr_worker.cpp` streaming `handleChunk` (and the `LLMInferenceRuntime`
request path): wire the PCM→mel extractor for `pcm_b64` content (mirror the
`mel_path` fixture path used by #10), OR have `handleChunk` convert `pcm_b64`→mel
via the already-configured `--melSettings/--melFilters` before `runOneShotCore`
(the worker already loads those). Independently, guard the Alpamayo action-load so
a missing `action/action.engine` does NOT `abort()` (it is irrelevant to ASR) —
e.g. skip when the dir/file is absent instead of constructing `Alpamayo1ActionRunner`.
```
