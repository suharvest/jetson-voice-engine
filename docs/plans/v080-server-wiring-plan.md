# v0.8.0 Server Wiring Plan — ASR / TTS / MOSS / LLM into seeed-voice (voxedge)

**Date:** 2026-06-10
**Scope:** Task #2 of the v0.8.0 voice-stack migration — wire the v0.8.0
ASR/TTS/MOSS engines + serving workers into the seeed-voice Python service so
the standard endpoints serve the migrated stack. Read-mostly design; sets up
the #2 implementation + the #7 Docker image.
**Status of inputs:** v0.8.0 streaming-ASR runtime DONE (v080-0012), TTS
CustomVoice accuracy GREEN (v080-0009 CORRECTED), MOSS port roundtrip-verified
(v080-0011), TTS N=2 batch-lane (v080-0010). Pipeline co-residency (v080-0013).

---

## 0. The two-layer architecture (what plugs into what)

The serving path has **three** layers; v0.8.0 only changes the bottom one.

```
seeed-voice service (server/)
  ├─ endpoints (WS /asr, /tts, /v2v) ── unchanged
  ├─ server/core/voxedge_backend_config.py  ── env/profile → voxedge config dataclass  (TRANSLATION LAYER)
  ├─ server/core/deploy_paths.py            ── binary + engine-dir RESOLVERS          (★ THE WIRING SURFACE)
  └─ voxedge.backends.jetson.*              ── env-free backend adapters (spawn worker, talk JSONL)
        spawns ↓
  native C++ serving workers (this repo: native/edgellm_voice_worker/, engine-overlay/addon/cpp/workers/)
        link ↓
  TensorRT-Edge-LLM v0.8.0 engine baseline (the box build_sm87 + v080-* patches)
```

**Key consequence:** the voxedge backends and the service translation layer are
**unchanged by v0.8.0**. The migration is almost entirely (a) rebuilding the
serving worker binaries against the v0.8.0 engine, and (b) repointing
`deploy_paths.py` defaults / deploy env at the v0.8.0 binaries + engine dirs.
The voxedge config dataclasses already carry every field the workers need
(`--max_slots`, explicit-KV flags, mel assets, stream params).

---

## 1. The voxedge worker protocol (per backend)

All three backends are resident-subprocess + JSON-line-over-stdio, demuxed by
`request_id`/`id`. ASR and TTS share `voxedge.backends.jetson.worker_io.WorkerIO`
(`voxedge/backends/jetson/worker_io.py`); MOSS has its own inline demuxer.

### 1.1 ASR — `qwen3_asr_worker`

Adapter: `voxedge/backends/jetson/trt_edge_llm_asr.py`.
Worker source (this repo): `native/edgellm_voice_worker/qwen3_asr_worker.cpp`.

**Spawn** (`_ensure_worker`, `trt_edge_llm_asr.py:422-469`):
```
<worker_binary> --engineDir <engine_dir> --multimodalEngineDir <audio_encoder_dir>
                [--max_slots N]            # only when N>1 (b1cb1a5)
                [--melSettings <json> --melFilters <bin>]   # enables pcm_b64 streaming
env: EDGELLM_PLUGIN_PATH=<plugin_path>, EDGE_LLM_ASR_CUDA_GRAPH=<worker_cuda_graph>
```
Worker emits a single startup line `{"event":"ready","init_ms":...,"max_slots":N}`
read by `_ensure_worker` BEFORE handing stdout to the WorkerIO reader thread
(`trt_edge_llm_asr.py:457-469`).

**Two request shapes** (worker dispatches on presence of `event`):

1. **One-shot / offline** (no `event` field) — `_transcribe_worker`
   (`trt_edge_llm_asr.py:570-591`):
   ```json
   {"id":"<rid>","requests":[{"messages":[{"role":"user","content":[{"type":"audio","audio":"<mel.safetensors path>"}]}]}],
    "batch_size":1,"temperature":1.0,"top_p":1.0,"top_k":1,"max_generate_length":200,
    "apply_chat_template":true,"add_generation_prompt":true}
   ```
   Reply: `{"ok":true,"responses":[{"output_text":"language Chinese ……"}]}`.
   Handled by `handleOneShot` (worker `qwen3_asr_worker.cpp:1463`), slot 0.

2. **Streaming** (only when `stream_mode=worker`) — `_TRTEdgeLLMStreamingASRStream`
   (`trt_edge_llm_asr.py:979-1168`):
   - `begin`  → `{"event":"begin_ack","id":sid}`  (worker `:1008`)
   - `chunk`  with `pcm_b64` (float32 cumulative PCM) + `audio_sec` + `last`
     → `{"event":"partial"|"final","text":...}` or `{"event":"segment_rotation","carryover_sec":...}` (worker `:1374,:1396,:1442`)
   - `end`    → terminal
   - pool full → `{"event":"error","error":"pool_saturated","status":4429,"max_slots":N}` (worker `:983-985`)
   - cancel   → `{"type":"cancel","id":rid}` (WorkerIO.cancel, `worker_io.py:246`)

**Streaming partials/finals flow:** synchronous per-line — the backend uses
`WorkerIO.request()` (sync generator) for each `begin/chunk/end`, reading exactly
one reply line per request (`_worker_request`, `trt_edge_llm_asr.py:471-508`).
The worker's `chunk` event accepts `pcm_b64` ONLY if started with
`--melSettings/--melFilters` (worker `:1142-1168`); else it expects `mel_path`.
The **production default is `stream_mode=accumulate`** (`_TRTEdgeLLMAccumulatingASRStream`):
no streaming protocol — buffer all audio, split at silence, run the one-shot
path per segment on `finalize()`. So the streaming worker protocol is **dormant
in production**; the offline/one-shot path is the live one.

**Language-tag prefix:** v0.8.0 ASR output is prefixed `language Chinese …`. The
service strips it at the service layer via `_strip_language_prefix`
(`trt_edge_llm_asr.py:545-568`), already wired in both one-shot and streaming.
**Caller requirement preserved:** `apply_chat_template=true` + `add_generation_prompt=true`
are already hard-set in the request builders (`:589-590`, `:693-694`).

### 1.2 TTS — `qwen3_tts_worker`

Adapter: `voxedge/backends/jetson/trt_edge_llm_tts.py`.
Worker source (this repo): `native/edgellm_voice_worker/qwen3_tts_worker.cpp`.

**Spawn** (`_ensure_worker`, `trt_edge_llm_tts.py:795-831`):
```
<worker_binary> --talkerEngineDir <talker_dir> --codePredictorEngineDir <cp_dir>
                --tokenizerDir <tokenizer_dir> --code2wavEngineDir <code2wav_dir>
                [explicit-KV flags]  [--max_slots N]
explicit-KV flags (each emitted iff config field set, _explicit_kv_flags :782-793):
  --qwen3TtsTalkerBackend --qwen3TtsTalkerEngine --codePredictorBackend
  --qwen3TtsTextProjection --qwen3TtsPromptKvCache
```
Startup line `{"event":"ready","init_ms":...}` (worker `qwen3_tts_worker.cpp:323`).

**Request** (`_synthesize_worker` / stream variants, `trt_edge_llm_tts.py:851+`):
```json
{"id":"<rid>","text":"…","output_file":"<wav path>"(offline) | "stream":true(streaming),
 "language":"chinese","talker_temperature":0.9,"talker_top_k":50,"talker_top_p":1.0,
 "predictor_*":…,"first_chunk_frames":5,"chunk_frames":25,"max_chunk_frames":…,
 "apply_chat_template":true,"add_generation_prompt":true,"ref_audio_b64":…(voice clone)}
```
**Streaming:** worker emits `{"event":"chunk","id":rid,...,"audio_b64"|samples}`
repeatedly then `{"event":"done",...}` (worker `:430-431,:610,:661`); errors as
`{"event":"error","ok":false,"error":...}` (`:679`). Saturation → `status:4429`
(`_tts_pool_saturated_error`, `trt_edge_llm_tts.py:232-252`).

### 1.3 MOSS — `moss_tts_nano_worker`

Adapter: `voxedge/backends/jetson/moss_tts_nano.py`.
Worker source (this repo): `engine-overlay/addon/cpp/workers/moss_tts_nano_worker.cpp`.

**Spawn** (`preload`, `moss_tts_nano.py:164-172`):
```
<worker_bin> --engine-dir <engine_dir> --tokenizer-model <tok.model>
             --codec-onnx-dir <codec_onnx_dir> --max-slots N --max-seq-len 2048
```
(A `.py`-suffixed `worker_bin` switches to the ORT persistent worker variant,
`moss_tts_nano.py:156-163` — not the Jetson default.)
Startup `{"event":"worker_ready",...}` (`moss_tts_nano.py:213`).

**Request** (`_build_request`, `moss_tts_nano.py:370-390`):
```json
{"id":"<rid>","request_id":"<rid>","text":"…","stream":true,
 "chunk_transport":"base64","chunk_format":"pcm_s16le","chunk_frames":8,
 "ref_audio_b64":…,"ref_audio_sample_rate":48000}
```
**Streaming:** `{"event":"ready"}` → `{"event":"chunk","audio_b64":…}`× →
`{"event":"done","ttfa_ms","total_frames","wall_ms"}`; errors `{"event":"error"}`;
crash `{"event":"worker_exit"}` (`moss_tts_nano.py:259-289`). Stereo s16le PCM @
48kHz, wrapped to WAV in Python.

### 1.4 LLM (GDN) — NOT a worker

Served separately by `edge-llm-chat-service` over OpenAI-compat HTTP. The voice
service is a plain HTTP client: `server/core/edge_llm_backend.py:51-103`
(`EDGE_LLM_BASE_URL` → `EDGE_LLM_CHAT_URL` → localhost default, normalized to
`/v1/chat/completions`). v0.8.0 wiring = **repoint the URL** at the v0.8.0 GDN
server; no worker build.

---

## 2. Serving-worker status table (exists / build / adapt)

| Worker | Source (this repo) | Speaks voxedge protocol? | v0.8.0 BUILD status | Verdict |
|---|---|---|---|---|
| **ASR** `qwen3_asr_worker` | `native/edgellm_voice_worker/qwen3_asr_worker.cpp` | ✅ YES — `ready`/`begin_ack`/`partial`/`final`/`segment_rotation`/one-shot `requests`/`pcm_b64`/`mel_path`/`--max_slots`/`--melSettings` all present | ⚠️ **MUST-BUILD + LIKELY-ADAPT** — source uses the **v0.7.1-era runtime API** (`#include "runtime/llmInferenceSpecDecodeRuntime.h"` + `runtime/slotPool.h`, `qwen3_asr_worker.cpp:15-17,227,338-348`), NOT the v0.8.0 streaming-session API (`AsrStreamingSessionRuntime`/`decodeAsrSessionToCompletion`/`getAsrSessionTranscript`) that v080-0012 validated. Those v0.8.0 APIs were proven only via the **spike driver** `spike_v080_m6_audio_streaming` (`docs/plans/v080-0012-asr-streaming-decode-hook-acceptance.md:7,44-51`), never wired into the serving worker. | **BIGGEST RISK** — see §5 |
| **TTS** `qwen3_tts_worker` | `native/edgellm_voice_worker/qwen3_tts_worker.cpp` | ✅ YES — `ready`/`chunk`/`done`/`error`, `--talkerEngineDir`/`--code2wavEngineDir`/`--tokenizerDir`/`--codePredictorEngineDir` (`qwen3_tts_worker.cpp:87-117,323,430,610,679`) | ⚠️ **MUST-BUILD + VERIFY** — uses the **standard v0.8.0 runtime** `Qwen3OmniTTSRuntime` + `Code2WavRunner` (`qwen3_tts_worker.cpp:307-314`), the SAME runtime the v0.8.0 `qwen3_tts_inference` accuracy gate (v080-0009 CORRECTED) ran on. The worker is a thin streaming wrapper around it. **GAP:** v080-0009 verified `qwen3_tts_inference` (one-shot), NOT this serving worker; AND the worker does NOT parse the explicit-KV CLI flags (`--qwen3Tts*`/`--codePredictorBackend`) that the voxedge config can pass (only 4 getopt entries at `:97-100`). | **MUST-BUILD; verify roundtrip + flag-coverage** — see §5 |
| **MOSS** `moss_tts_nano_worker` | `engine-overlay/addon/cpp/workers/moss_tts_nano_worker.cpp` | ✅ YES — `worker_ready`/`ready`/`chunk`/`done`/`error`/`worker_exit`, `--engine-dir`/`--tokenizer-model`/`--codec-onnx-dir`/`--max-slots`/`--max-seq-len` | ✅ **DONE** — port roundtrip-verified on v0.8.0 (`docs/plans/v080-0011-moss-tts-nano-roundtrip-acceptance.md`): ZERO source adaptations (standalone `MossTtsNanoRuntime`, no v0.8.0 runtime inheritance), the v080-0011 roundtrip **used this very worker** (JSONL roundtrip = correct zh speech). | **EXISTS — just build + stage binary** |
| **LLM (GDN)** | n/a (HTTP) | n/a | ✅ separate `edge-llm-chat-service` v0.8.0 (memory fix already proven, migration-spec §"Memory driver RESOLVED") | **Repoint URL only** |

---

## 3. Per-backend wiring deltas (what to change)

All path/binary changes land in **`server/core/deploy_paths.py`** (env-overridable)
and/or the deploy env (compose `.tmpl` / `agent.yaml.tmpl`). The voxedge backends
and `voxedge_backend_config.py` need **no code change** for the path repoint.

### 3.1 ASR
- `EDGE_LLM_ASR_WORKER_BIN` → v0.8.0 `qwen3_asr_worker` (default resolved in
  `deploy_paths.py:91-97` / `resolve_asr_worker_binary` `:270-281` — repoint the
  build dir `_VOICE_WORKER_BUILD`/`_EDGE_LLM_BUILD`).
- `EDGE_LLM_ASR_ENGINE_DIR` → v0.8.0 thinker engine (`engines-v080/`).
  Default cascade `deploy_paths.py:310-338` keys off `~/qwen3-asr-edgellm-runtime/engines/…`;
  add/point the v0.8.0 engine dir (e.g. `thinker_full_fp8embed` per v080-0012,
  **max_input_len=256** — matches worker constant `kEngineMaxInputLen=256`,
  `qwen3_asr_worker.cpp:102`).
- `EDGE_LLM_ASR_AUDIO_ENC_DIR` → v0.8.0 audio encoder (`engines-v080/audio_encoder`,
  `deploy_paths.py:339-344`).
- `EDGELLM_PLUGIN_PATH`/`EDGE_LLM_ASR_PLUGIN_PATH` → v0.8.0 plugin .so
  (`deploy_paths.py:98-106`).
- Profile/env unchanged: `stream_mode=accumulate` (production default — keeps the
  one-shot path live), `EDGE_LLM_ASR_MAX_CONCURRENT` (→ `--max_slots`).

### 3.2 TTS (qwen3-tts)
- `EDGE_LLM_TTS_WORKER_BIN` → v0.8.0 `qwen3_tts_worker` (`deploy_paths.py:80-86` /
  `resolve_tts_worker_binary` `:253-267`).
- `EDGE_LLM_TTS_TALKER_DIR` / `_CP_DIR` / `_CODE2WAV_DIR` / `_TOKENIZER_DIR` →
  v0.8.0 TTS engines (`engines-v080-tts/…`); resolvers `deploy_paths.py:184-250`.
- `EDGELLM_PLUGIN_PATH` shared with ASR.
- **CALLER REQUIREMENT (preserved):** `apply_chat_template=true` +
  `add_generation_prompt=true` already hard-set in the worker request
  (`trt_edge_llm_tts.py:1276-1277`). Confirm the **serving worker** honors them
  the way `qwen3_tts_inference` does (v080-0009 role-coercion `:392-398`).
- **Explicit-KV flag gap:** if the v0.8.0 highperf-nx profile ships a single-
  optimization-profile w8a16 talker engine, either (a) add the
  `--qwen3Tts*`/`--codePredictorBackend` getopt entries to the v0.8.0
  `qwen3_tts_worker.cpp`, or (b) deploy the generic 2-profile engine and leave
  `EDGE_LLM_TTS_TALKER_BACKEND` etc. empty (legacy generic-runner path; v080-0010
  used `maxBatchSize=2` batch-lane, not slot replication — verify which engine).

### 3.3 MOSS
- `MOSS_WORKER_BIN` → v0.8.0 `moss_tts_nano_worker` (default `/opt/jv-workers/...`,
  config builder `voxedge_backend_config.py:~972`; staged by
  `server/core/moss_artifacts.py` → `/opt/jv-workers`).
- `MOSS_ENGINE_DIR` / `MOSS_CODEC_ONNX_DIR` → v0.8.0 MOSS engines/codec (unchanged
  paths if re-staged from HF; v080-0011 confirms engines are v0.7.1↔v0.8.0
  binary-compatible — ZERO source change).
- No protocol delta.

### 3.4 LLM
- `EDGE_LLM_CHAT_URL` / `EDGE_LLM_BASE_URL` → v0.8.0 `edge-llm-chat-service`
  endpoint (`edge_llm_backend.py:51-103`). Container/URL swap only.

---

## 4. Ordered implementation steps (Task #2) + gates

> Build only via the documented worker build (`native/edgellm_voice_worker`
> README:23-34: `cmake -S native/edgellm_voice_worker … -DEDGE_LLM_BUILD_DIR=<v0.8.0 build>`).
> Never bare-cmake the engine. MOSS worker builds via
> `engine-overlay/addon/cpp/workers/build_moss_worker.sh`.

**Step 0 — Engines staged (precondition).** v0.8.0 ASR (`engines-v080/` thinker
+ audio encoder), TTS (`engines-v080-tts/` talker/cp/code2wav/tokenizer), MOSS
(engines + codec), plugin `.so`, all present on-device.
*Gate:* `preload()` file-existence checks pass (ASR `trt_edge_llm_asr.py:309-323`;
TTS `:669`; MOSS `:146-218`).

**Step 1 — Build the 3 serving workers against v0.8.0.**
  - MOSS: `build_moss_worker.sh` → `moss_tts_nano_worker`. *Gate:* re-run the
    v080-0011 JSONL roundtrip (worker stdin → zh speech → ASR back, CER≈0).
  - TTS: `cmake --target qwen3_tts_worker`. *Gate:* drive the **worker** (not
    `qwen3_tts_inference`) with `apply_chat_template=true` → ASR-roundtrip zh+en
    CORRECT (reuse the v080-0009 CORRECTED harness, but feed the worker).
  - ASR: see §5 — likely an **adapt** (port serving worker onto
    `AsrStreamingSessionRuntime` OR build the v0.7.1 runtime into the v0.8.0
    engine). *Gate:* worker one-shot `requests` output == v080-0012 one-shot
    transcript (CER 0.0000 vs the v0.8.0 `llm_inference` golden), zh_long_01 == golden.

**Step 2 — Repoint `deploy_paths.py` + deploy env at v0.8.0 binaries + engines**
(§3). No voxedge / config-builder code change for the repoint.
*Gate:* `build_trt_edge_llm_asr_config()` / `_tts_config()` / `build_moss_tts_nano_config()`
(`server/core/voxedge_backend_config.py`) resolve to the v0.8.0 paths
(unit-test pattern: `server/tests/test_voxedge_backend_config.py`,
`test_trt_edge_llm_ipc_paths.py`, `test_jetson_backends_env_fresh.py`).

**Step 3 — Repoint the LLM URL** (§3.4). *Gate:* `/v2v` round-trips through the
v0.8.0 `edge-llm-chat-service` (`test_v2v_server_loop.py`).

**Step 4 — End-to-end service gate (the real acceptance).** Bring up the service
(local or #7 image) and exercise the standard endpoints:
  - ASR WS / offline upload → correct zh + en transcript, `language X` prefix
    stripped, long audio (>6s) segmented OK.
  - TTS WS → intelligible streaming zh + en, `done` event, N=2 burst no 4429
    leak.
  - MOSS (if profile selected) → correct zh speech.
  - `/v2v` → full ASR→LLM→TTS turn.
*Gate:* matches v080-0013 pipeline co-residency numbers; no worker crash in
`docker logs | grep -iE 'error|crash|exit'`.

---

## 5. The biggest risk — the ASR serving worker

**The ASR serving worker is the only true build-from-source gap, and it is a
real adaptation, not just a recompile.**

v080-0012 closed the runtime gap by extracting the decode loop into
`decodeAsrSessionToCompletion` / `getAsrSessionTranscript` and validated it via
the **m6 spike driver** — proving *the v0.8.0 streaming runtime can produce a
transcript* (CER 0.0000 vs golden). But the **serving** `qwen3_asr_worker.cpp` in
this repo (`native/edgellm_voice_worker/`) is the v0.7.1-era worker: it includes
`runtime/llmInferenceSpecDecodeRuntime.h` + `runtime/slotPool.h` and drives its
own `appendPrefillChunk` / `AsrSlot` machinery (`qwen3_asr_worker.cpp:15-17,
227,338-348,1107`). It does **not** call the new v0.8.0 session API.

Two resolution options for Step 1 (ASR):

- **Option A (preferred, true v0.8.0 streaming):** adapt `qwen3_asr_worker.cpp`'s
  per-session slot to drive `AsrStreamingSessionRuntime` + `decodeAsrSessionToCompletion`
  + `getAsrSessionTranscript` (the m6-proven path), keeping the JSONL protocol
  surface byte-identical (`begin_ack`/`partial`/`final`/`segment_rotation`/one-shot
  `requests`). Most engineering; unlocks real sub-WAV streaming later.

- **Option B (lower-risk, production-equivalent):** since production runs
  `stream_mode=accumulate` (the one-shot `requests` path, NOT streaming), it is
  sufficient to make the worker's **`handleOneShot`** path compile + run against
  the v0.8.0 engine. If the v0.7.1 `LLMInferenceSpecDecodeRuntime` still exists in
  the v0.8.0 baseline, a plain recompile may suffice for the live path; the
  streaming branch can stay dormant. **Verify whether
  `llmInferenceSpecDecodeRuntime.h` survives in the v0.8.0 engine** — if removed
  in the re-architecture, even the one-shot path needs porting to the new runtime,
  pushing toward Option A.

**First action for the executor:** grep the v0.8.0 engine baseline (the box
`tensorrt-edge-llm` tree the patches target) for `llmInferenceSpecDecodeRuntime.h`
and `slotPool.h`. Their presence/absence decides Option A vs B and is the single
biggest scoping unknown for Task #2.

Secondary risks (smaller): (1) TTS worker explicit-KV flag coverage (§3.2) if the
highperf-nx engine is single-profile; (2) confirming the TTS **serving worker**
(not the one-shot binary) reproduces the v080-0009 CORRECTED accuracy with
`apply_chat_template=true`.

---

## 6. File:line index (for the executor)

| Concern | File:line |
|---|---|
| ASR adapter spawn | `voxedge/backends/jetson/trt_edge_llm_asr.py:422-469` |
| ASR one-shot request | `…/trt_edge_llm_asr.py:570-591` |
| ASR streaming protocol | `…/trt_edge_llm_asr.py:979-1168` |
| ASR lang-prefix strip | `…/trt_edge_llm_asr.py:545-568` |
| TTS adapter spawn + explicit-KV flags | `…/trt_edge_llm_tts.py:782-831` |
| TTS request (apply_chat_template) | `…/trt_edge_llm_tts.py:1276-1277` |
| MOSS adapter spawn + protocol | `voxedge/backends/jetson/moss_tts_nano.py:146-301,370-390` |
| WorkerIO demux | `voxedge/backends/jetson/worker_io.py` |
| Config builder (ASR/TTS/MOSS) | `server/core/voxedge_backend_config.py:54-225,695-929,932-1000` |
| Path/binary resolvers (★ wiring surface) | `server/core/deploy_paths.py:76-106,124-345` |
| MOSS artifact staging | `server/core/moss_artifacts.py:48-60` |
| LLM HTTP client | `server/core/edge_llm_backend.py:51-103` |
| ASR worker C++ (build target) | `native/edgellm_voice_worker/qwen3_asr_worker.cpp` |
| TTS worker C++ (build target) | `native/edgellm_voice_worker/qwen3_tts_worker.cpp` |
| MOSS worker C++ (build target) | `engine-overlay/addon/cpp/workers/moss_tts_nano_worker.cpp` |
| Worker build instructions | `native/edgellm_voice_worker/README.md:23-34` |
| v0.8.0 ASR runtime hooks proof | `docs/plans/v080-0012-asr-streaming-decode-hook-acceptance.md` |
| v0.8.0 TTS accuracy proof | `docs/plans/v080-0009-tts-customvoice-roundtrip-CORRECTED.md` |
| v0.8.0 MOSS proof | `docs/plans/v080-0011-moss-tts-nano-roundtrip-acceptance.md` |
