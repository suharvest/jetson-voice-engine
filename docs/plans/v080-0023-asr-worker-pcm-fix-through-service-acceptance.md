# v080-0023 — qwen3_asr_worker pcm_b64 fix: N>1 + streaming PARTIALs THROUGH the deployed /asr/stream service (acceptance)

**Date:** 2026-06-10
**Host:** orin-nx (`Linux 5.15.148-tegra aarch64`, hostname `orinnx`)
**Scope:** Fix the v0.8.0 `qwen3_asr_worker` so it handles the REAL service input
(`pcm_b64`) that #11 found aborts, then re-verify N>1 + streaming partials
**through `/asr/stream`** (the v080-0022 gate FAILED here — worker gap).

---

## 1. What v080-0016 → #9 lost (the diff) + the fix

### Root cause (confirmed, file:line)
The production `/asr/stream` path runs `stream_mode=worker` →
`StreamingWorkerSession._send_chunk` (voxedge `trt_edge_llm_asr.py:1025`) sends the
cumulative audio as **raw float32-LE PCM** in `pcm_b64`:
```python
pcm = np.asarray(self._audio_accum, dtype="<f4")
pcm_b64 = base64.b64encode(pcm.tobytes()).decode("ascii")
{"event":"chunk", "id":..., "pcm_b64":pcm_b64, ...}
```

The #9 refactor (`qwen3_asr_worker.cpp`, commit `ee4c4ed` on the device build
worktree / `dae9002` on overlay) DROPPED the PCM→mel front-end that the v0.7.x
streaming worker (`~/project/asr-worker-build-verify/qwen3_asr_worker.cpp`, 1746
lines) had. The #9 `buildFinalizeRequest` instead stuffed the raw `pcm_b64` into the
engine request as `content:[{type:audio, pcm_b64:<b64>}]` (its own comment admitted
the conversion was unwired). `requestFileParser.cpp:223` then does
`contentItemJson["audio"].get<std::string>()` — there is **no `audio` key** (only
`pcm_b64`), so nlohmann `operator[]` on the missing key trips an
`assert(it != end())` (json.hpp:2175) → **SIGABRT** (uncatchable, not a throw). The
`MmapReader: Cannot open .../action/action.engine` line in the v080-0022 logs is the
BENIGN caught case (`llmInferenceRuntime.cpp:316-328` wraps the Alpamayo action load
in try/catch; `MmapReader`/`ELLM_CHECK` *throws* `std::runtime_error`, which is
caught — it never aborts). The real abort is solely the missing-`audio`-key assert.

**Why #9/#10 passed standalone but failed through the service:** the v080-0021 #10
gate fed PRE-COMPUTED mel `.safetensors` via `mel_path`, which the parser loads
directly (extension `.safetensors`) — never the PCM path. The service supplies real
PCM via `pcm_b64`, which the #9 worker never actually exercised. The v080-0016
`.bak` (407-line one-shot) ALSO never handled `pcm_b64` — it only ever saw `mel_path`
because the live profile ran `stream_mode=accumulate` (Python computes the mel).

### The fix (`native/edgellm_voice_worker/qwen3_asr_worker.cpp`)
Restored the v0.7.x PCM→mel front-end and wired it into the #9 N>1 path:
1. **`gMelExtractor`** global, loaded in `main()` from `--melSettings`/`--melFilters`
   (the same mel front-end the `mel_path` fixtures were pre-computed with).
2. **`base64Decode` + `writeMelSafetensors` + `pcmB64ToMelSafetensors`** helpers
   (verbatim from the v0.7.x reference): decode float32-LE PCM → `gMelExtractor->
   compute(pcm, &n_frames, kEncoderMelFramesPerChunk=100)` → write a `[1,128,T]` fp16
   mel `.safetensors` temp file.
3. **`handleChunk` / `handleEnd`**: when a chunk/end carries `pcm_b64`, convert the
   cumulative PCM to a mel safetensors (`st.melPathOwned`) BEFORE `runOneShotCore`,
   then `buildFinalizeRequest` emits ONLY `{type:audio, audio:<.safetensors>}` — the
   parser never sees a bare `pcm_b64` again. Owned tempfiles are rm'd on lane
   release + idle sweep. Refuse cleanly (`pcm_input_unsupported`) if mel assets are
   absent.
4. **action-engine guard**: confirmed already-safe — the engine ctor's action load is
   try/caught and `MmapReader` throws (not asserts) on absence; ASR never builds
   trajectory-history requests, so the action runner is simply null. No engine change
   needed.
5. `mel_path` path + #9 N>1 lanes + #10 streaming partials all preserved.

Worker source md5 `ab09b99258c1f894ad385dc13b042af2`; built binary md5
**`5ebd436bb2f322304f97540afea4e893`** (was `f9bb821d…`), built EXIT 0 against
`/home/harvest/project/edgellm-v080/build` (`libedgellmCore.a`).

## 2. Image rebuilt
`sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.8.0-edgellm-20260610g`
(thin overlay FROM `:prod-unified-v8`, build EXIT 0). Bakes the FIXED worker; same
n2 wiring as `:f` (profile `jetson-edgellm-v080-n2`, `STREAM=worker`, `MAXCONC=2`,
`ENGDIR=…/asr-b2/llm`). Image-baked verification:
```
md5 /opt/jv-workers/qwen3_asr_worker = 5ebd436bb2f322304f97540afea4e893
PROFILE=jetson-edgellm-v080-n2 STREAM=worker MAXCONC=2 ENGDIR=…/asr-b2/llm
MEL_S/MEL_F resolved
```

## 3. THROUGH-SERVICE GATES (throwaway `v080-verify-n2b`, port 8647, engine-cache
hardlink-mounted, minchunk1 encoder (`min_time_steps=100`, md5 `3b9ff631…`) bind-
mounted over `asr/audio/audio`, b2 engine md5 `4122dfcc…`, `EDGELLM_V080_AUTO_PULL=0`)

### Boot — coherent
```
Applied profile jetson-edgellm-v080-n2 (36 env keys; 0 stale cleared)
asr.supports_parallel=True/max=2 ; stream_mode='worker' ; engine_dir=…/asr-b2/llm
ASR backend preload OK (max_slots=2, mel_settings/mel_filters resolved)
Application startup complete ; /health → asr:true asr_capabilities:[streaming,offline,multi_language]
worker proc: --engineDir …/asr-b2/llm --max_slots 2 --melSettings … --melFilters …
```

### Session-gate change (required, service-config knob)
`SessionLimiter effective_limit = min(asr=2, tts=1) = 1` clamps ASR admission to 1.
Lifting `OVS_TTS_WORKER_CONCURRENCY=2` alone CRASHES the generic-runner TTS worker
(`unrecognized option '--max_slots'`). Resolved with a verify-only env triplet that
does NOT touch the TTS worker:
`OVS_TTS_WORKER_CONCURRENCY=2` (lifts the aggregate ceiling to 2) +
`LAZY_TTS=1` (skips TTS preload so the worker is never spawned with `--max_slots`) +
`OVS_MAX_CONCURRENT_SESSIONS=2`. → `effective_limit=2`, TTS never invoked (ASR-only
gate), no crash. (The production-correct fix is an ASR-endpoint-specific gate; out of
scope for the worker fix.)

### Single-session streaming via /asr/stream — PASS (the path that SIGABRT'd before)
zh (`zh_seeed_solution_source.wav`, real PCM, 300 ms hops):
```
15 partials (is_final:false), progressive:
  欢迎。 → 喜地科技。 → 喜地科技，Seed Studio对话。 → 喜地科技，Seed Studio对话式AI。
FINAL (is_final:true): 喜地科技、Seed Studio对话式AI。   (correct)
worker ALIVE after, --max_slots 2, no SIGABRT
```

### N=2 concurrent + isolation via /asr/stream — PASS
A=zh + B=en streamed concurrently:
```
A: 15 partials, final 喜地科技、Seed Studio对话式AI。  (Chinese)
B:  8 partials, final Studio conversational AI.        (English)
NO cross-talk (A stays zh, B stays en) ; worker ALIVE after ; logs clean
```

### 3rd session → 4429 — PASS
With A+B holding both lanes, a 3rd /asr/stream:
```
C_third err: ConnectionClosedError received 4429 {"error":"too_many_sessions","current":2,"limit":2}
service log: session_limiter: WS 4429 endpoint=/asr/stream active=2 limit=2
```

### Logs clean
No SIGABRT / abort / action.engine-fatal / satisfyProfile / 501 / Traceback across
all gates. (The first concurrent attempt failed only due to a stale OOM-killed TTS
worker from a prior throwaway eating RAM down to 1.2 G; after stopping production for
RAM, RAM=7 G and both sessions passed.)

## 4. Container restore
Production stopped for RAM (snapshot taken), then **started ALL** (never rm/down/-v):
```
seeed-voice            Up (8621)  prod-unified-v8       /health 200, "Speech service ready" (paraformer_trt + matcha_trt)
edge-llm-chat-service  Up (8000)  openai-compat-v6      healthy
translator             Up (9001)  translator-cuda-jetson-v2  healthy (never touched)
industrial-security-demo  Restarting (pre-existing loop, not touched)
v080-verify-n2b        REMOVED (docker rm -f)
```

## 5. Commit + HF re-stage
- HF: `sm87-trt10.3-jp6.2/v0.8.0/workers/qwen3_asr_worker` replaced
  `f9bb821d…` → **`5ebd436bb2f322304f97540afea4e893`** (download-verified byte-match).
- Commit `v080-0023-*` to overlay worktree `feat/edgellm-v080-migration` (NOT pushed).

## 6. Verdict
**YES** — N>1 + streaming partials now work THROUGH the deployed `/asr/stream`:
- single-session streaming partials (15 `is_final:false` before a correct final),
- N=2 concurrent + isolated (zh+en, no cross-talk),
- 3rd → 4429,
- worker survives (no SIGABRT), logs clean.

The cumulative re-decode per hop (the #10 first-cut) is correct but O(audio-so-far).
**Ready for #12 (true incremental-KV per-hop append)** — the per-hop optimization
that replaces the cumulative re-decode; the pcm→mel front-end this task restored is a
prerequisite the incremental path also needs.
