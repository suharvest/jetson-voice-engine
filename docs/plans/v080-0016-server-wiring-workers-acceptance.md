# v080-0016 — Server-wiring serving workers: build + accuracy-through-worker acceptance

**Date:** 2026-06-10
**Host:** orin-nx (`Linux 5.15.148-tegra aarch64`, hostname `orinnx`)
**Scope:** Task #2 of the v0.8.0 voice-stack migration — build the 3 serving
workers (ASR / TTS / MOSS) against the v0.8.0 TensorRT-Edge-LLM engine baseline
and prove each reproduces its proven accuracy gate **through the worker JSONL
protocol**, then document the `deploy_paths.py` env contract for repointing.

**Engine baseline:** `~/project/edgellm-v080` (v0.8.0 `+` landed migration
patches), build dir `~/project/edgellm-v080/build` (`libedgellmCore.a`,
`libNvInfer_edgellm_plugin.so`).
**Worker sources (this repo):** `native/edgellm_voice_worker/{qwen3_asr_worker.cpp,
qwen3_tts_worker.cpp}` + `engine-overlay/addon/cpp/workers/moss_tts_nano_worker.cpp`.

---

## 1. Per-worker build + adaptation

| Worker | Adaptation | Build | Binary md5 / bytes |
|---|---|---|---|
| **MOSS** `moss_tts_nano_worker` | NONE (standalone `MossTtsNanoRuntime`, v0.7.1↔v0.8.0 engine-compatible) | `cpp/workers/build_moss_worker.sh` (EDGELLM_SRC=edgellm-v080, ORT_ROOT=/opt/onnxruntime-linux-aarch64-1.23.2, SP_ROOT=/usr) | `6a03bdf5c7a26b09f60597b95008ebfe` / 562272 |
| **TTS** `qwen3_tts_worker` | **STRUCT-API PORT** (v0.8.0 `TalkerGenerationRequest`/`Response` fields changed; `CodecFrameCallback` removed) | CMake `--target qwen3_tts_worker` (EDGE_LLM_SOURCE_DIR/BUILD_DIR=edgellm-v080) | `22216e8dc724bd8619d4fca26b0c2d5b` / 14090720 |
| **ASR** `qwen3_asr_worker` | **RUNTIME PORT** (v0.7.1 spec-decode runtime + slot pool DELETED → driven on v0.8.0 `LLMInferenceRuntime` one-shot) | CMake `--target qwen3_asr_worker` | `be7bee91728c63253e5926a5933896c0` / 13921904 |

### 1.1 ASR adaptation — DELETED-runtime confirmation + LLMInferenceRuntime swap

Confirmed DELETED in the v0.8.0 baseline (no occurrence anywhere in
`~/project/edgellm-v080`):
- `runtime/llmInferenceSpecDecodeRuntime.h` → `rt::LLMInferenceSpecDecodeRuntime`
  (its `AppendPrefillStatus` enum, public `appendPrefillChunk`, `getBaseEngine`)
- `runtime/slotPool.h` → `tensorrt_edge_llm::runtime::SlotPool<T>`

Present replacement: `cpp/runtime/llmInferenceRuntime.h` → `rt::LLMInferenceRuntime`
with the SAME one-shot surface the v080-0012 golden (`llm_inference`) used:
`LLMInferenceRuntime(engineDir, multimodalEngineDir, loraWeightsMap, stream)` +
`handleRequest(LLMGenerationRequest, LLMGenerationResponse, stream)` →
`response.outputTexts`. `LLMGenerationRequest/Response` are UNCHANGED.

**Adaptation (Option B — production-equivalent one-shot):** production runs
`stream_mode=accumulate`, so only the one-shot `requests` path is live. The worker
was reduced (1746 → 407 lines) to:
- a single local `std::unique_ptr<rt::LLMInferenceRuntime>` + one `cudaStream_t`
  in `main()` (vanilla ctor + `captureDecodingCUDAGraph`); no slot pool, no
  shared-engine `getBaseEngine` multi-slot path (impossible — API removed).
- `handleOneShot(...)` re-typed to `rt::LLMInferenceRuntime&`; the
  `mapAppendStatusToErrorEvent` (AppendPrefillStatus) error branch replaced with a
  generic structured error. Everything else (`exampleUtils::parseRequestFile`,
  `runtime.handleRequest`, `outputTexts`) is byte-identical to the proven path.
- streaming `begin`/`chunk`/`end` → `{"event":"error",
  "error":"streaming_not_supported_v080_oneshot_worker","status":501}` stub.
  Re-enabling real sub-WAV streaming = Option A (`AsrStreamingSessionRuntime`),
  deferred — dormant in production.
- JSONL protocol surface preserved: startup `{"event":"ready","init_ms",
  "max_slots":1}`; one-shot `requests` reply `{"event":"done","ok":true,
  "responses":[{"output_text":…}]}`.

### 1.2 TTS adaptation — v0.8.0 struct-API port

v0.8.0 `Qwen3OmniTTSRuntime` is the SAME runtime as the v080-0009 CORRECTED golden
(`qwen3_tts_inference`), but the request/response structs changed:
- Request: removed `codecEosLogitOffset`, `predictorTemperature/TopK/TopP`,
  `minAudioLength`; the worker now sets `applyChatTemplate`/`addGenerationPrompt`/
  `enableThinking` and coerces a single `user` message → `assistant` (matches the
  golden role-coercion).
- `handleAudioGeneration` is now **fully batched, NO per-frame `CodecFrameCallback`**
  (that hook was removed). The worker's streaming `TtsStreamAdapter`/callback path
  was deleted; it now calls the 3-arg `handleAudioGeneration(req, resp, stream)`,
  reads `resp.batchRvqCodes[0]` (`[frame][codeLayer]`, replacing `rvqCodes`) +
  `resp.numFramesPerSample` (replacing `numFrames`), transposes
  `[frames][layers]→[layers][frames]`, and runs Code2Wav (golden recipe). Streaming
  output is now "generate-then-chunk": the completed frame buffer is flushed through
  the existing `submitChunk()` so the wire protocol (`chunk` events + final `done`)
  is unchanged.

---

## 2. PER-WORKER ACCURACY GATE — raw roundtrip through each worker

**ASR validated on known-good audio FIRST.** Paraformer harness driven
in-container exactly as v080-0011: `seeed-voice` kept UP, profile `jetson-zh-en`,
`PARAFORMER_ENC_ENGINE=…/paraformer_encoder_dp4_400.plan`,
`create_asr_backend().transcribe(wav, language="zh")`.

```
known-good zh_3.wav → 说起咱北京的烤鸭啊那可真是外焦里嫩色泽金黄一口咬下去满嘴流油   (clean → harness trustworthy)
```

### 2.1 MOSS — worker JSONL roundtrip (default voice, 48kHz stereo)

| prompt (input) | RMS | ASR roundtrip through worker | verdict |
|---|---|---|---|
| 今天天气很好，我们一起测试语音合成。 | 0.04194 | `天天气很好我们一起测测试语音合成` | PASS (matches v080-0011) |
| 语音合成的稳定性。 | 0.05975 | `语音合合成的稳定性` | PASS |
| 说起咱北京的烤鸭啊，那可真是外焦里嫩。 | 0.06328 | `起咱北京的烤烤鸭啊那可真是外焦里嫩` | PASS |

Minor leading-clip/doubled-char are paraformer streaming artifacts (48k-stereo→16k),
not MOSS errors; matches v080-0011 exactly.

### 2.2 TTS — worker JSONL roundtrip (speaker `vivian`, greedy, apply_chat_template=true)

```
EN input: "The weather is really nice today, let us go for a walk together."
  worker → 57 frames (== v080-0009 golden), 109440 samples
  paraformer roundtrip: theweatherisreallynicetodayletusgoforawalktogether   → BYTE-EXACT vs v080-0009 golden. CORRECT.

ZH input: "今天天气真不错"
  worker → 24 frames (== v080-0009 golden), 46080 samples
  paraformer roundtrip: 今天天气真不错   → BYTE-EXACT vs v080-0009 golden. CORRECT.
```
Caller requirement preserved: `apply_chat_template=true` + `add_generation_prompt=true`
hard-set in the worker request (drop to non-canonical → runaway/garbled, per v080-0009).

### 2.3 ASR — worker one-shot JSONL (long zh WAV)

```
fixture: asr_oneshot_zhlong01.json (mel zh_long_01.safetensors)
worker one-shot output_text: language Chinese这并不是告别，这是一个篇章的结束，也是新篇章的开始。
  strip "language Chinese" prefix → 这并不是告别，这是一个篇章的结束，也是新篇章的开始。
  v0.7.1 / v080-0012 golden       → 这并不是告别，这是一个篇章的结束，也是新篇章的开始。
  → BYTE-EXACT. CER 0.0000.
```

---

## 3. deploy_paths.py env contract (repoint — config-only, no code change)

The voxedge backends + `voxedge_backend_config.py` are UNCHANGED. Repoint is via
the `server/core/deploy_paths.py` env-override keys (set in deploy
`.tmpl`/compose env at #7 Docker time — not committed here per the no-compose-edit
rule). The v0.8.0 targets:

| Key | v0.8.0 value (orin-nx) |
|---|---|
| `EDGE_LLM_ASR_WORKER_BIN` | `…/v080-worker-build/build_v080/workers/qwen3_asr_worker` |
| `EDGE_LLM_ASR_ENGINE_DIR` | `…/Qwen3-ASR-0.6B/engines-v080/llm` |
| `EDGE_LLM_ASR_AUDIO_ENC_DIR` | `…/Qwen3-ASR-0.6B/engines-v080/audio` (runtime appends `/audio` → loads `audio/audio/audio_encoder.engine`; `llmInferenceRuntime.cpp:305`) |
| `EDGE_LLM_TTS_WORKER_BIN` | `…/v080-worker-build/build_v080/workers/qwen3_tts_worker` |
| `EDGE_LLM_TTS_TALKER_DIR` | `…/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-v080-tts/talker` (also the `--tokenizerDir`) |
| `EDGE_LLM_TTS_CP_DIR` | `…/engines-v080-tts/code_predictor` |
| `EDGE_LLM_TTS_CODE2WAV_DIR` | `…/engines-v080-tts/code2wav` |
| `EDGE_LLM_TTS_TOKENIZER_DIR` | `…/engines-v080-tts/talker` |
| `MOSS_WORKER_BIN` | v0.8.0 `moss_tts_nano_worker` (staged via `moss_artifacts.py` → `/opt/jv-workers`) |
| `MOSS_ENGINE_DIR` / `MOSS_CODEC_ONNX_DIR` | `/opt/models/moss-tts-nano/engines` / `/opt/models/moss-tts-nano/codec_onnx` (unchanged; engines v0.7.1↔v0.8.0 compatible) |
| `EDGELLM_PLUGIN_PATH` | `…/edgellm-v080/build/libNvInfer_edgellm_plugin.so` (shared ASR+TTS) |
| `EDGE_LLM_CHAT_URL` | v0.8.0 `edge-llm-chat-service` endpoint (URL swap only; no worker) |

ASR profile/env unchanged: `stream_mode=accumulate` keeps the one-shot path live;
`EDGE_LLM_ASR_MAX_CONCURRENT` → `--max_slots` (the v0.8.0 one-shot worker reports
`max_slots:1` and ignores >1, since slot-pool concurrency requires the deferred
Option A streaming-session port).

---

## 4. Container restore

Snapshot pre-build: seeed-voice Up, translator Up, edge-llm-chat-service Up,
industrial-security-demo Restarting (pre-existing loop). For RAM during the C++
builds, `translator` + `edge-llm-chat-service` were stopped (seeed-voice kept UP
for the paraformer roundtrip), then restarted. `industrial-security-demo`
(production-data) untouched throughout. **Final:** seeed-voice Up, translator Up,
edge-llm-chat-service Up.

## 5. Verdict

**3 serving workers BUILD against the v0.8.0 engine baseline AND each reproduces
its accuracy gate THROUGH THE WORKER JSONL protocol** (raw transcripts above;
ASR validated on known-good audio first):
- MOSS — zh roundtrip == v080-0011.
- TTS — en+zh roundtrip BYTE-EXACT == v080-0009 CORRECTED.
- ASR — one-shot output BYTE-EXACT == v0.7.1 / v080-0012 golden, CER 0.0000.

deploy_paths repoint = the documented env contract (§3). **Ready for #7 Docker.**

Deferred (non-blocking, dormant in production): ASR streaming via
`AsrStreamingSessionRuntime` (Option A) — currently a 501 stub; ASR `--max_slots>1`
concurrency (needs the slot-pool re-port onto the v0.8.0 session runtime); TTS
multi-chunk streaming windowing (currently single post-generation flush — wire
protocol valid).
