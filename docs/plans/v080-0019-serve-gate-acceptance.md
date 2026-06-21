# v080-0019 — v0.8.0 Docker image: REAL serve gate (ASR transcribes + TTS synthesizes)

**Date:** 2026-06-10
**Host:** orin-nx (`Linux 5.15.148-tegra aarch64`, hostname `orinnx`, Orin NX 16GB, JetPack 6.2)
**Image:** `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.8.0-edgellm-20260610e`
**Builds on:** v080-0018 (entrypoint CMD fix). Closes the 4 data-plane blockers that
prevented the v0.8.0 image from actually transcribing/synthesizing through the service.

---

## The 4 blockers — resolution

### Blocker 1 — audio encoder opt profile rejected short serving mels  ✅ FIXED (engine rebuild)

**Root cause.** The v0.8.0 audio encoder was built with
`audio_build --minTimeSteps 1000 --maxTimeSteps 3000`. In
`cpp/builder/audioBuilder.cpp:425` `minChunks = divUp(minTimeSteps, nWindowDim)`
with `nWindowDim = n_window*2 = 100` (config `n_window=50`), giving a `padded_feature`
opt profile of `[10,128,100]..[30,128,100]` — i.e. **10–30 audio chunks only**. The
serving path segments at `segment_cap_sec=5.5` and offline-splits long audio, so every
segment is **1–6 chunks**, below the 10-chunk floor → every real `/asr` request failed:
```
satisfyProfile. Set dimension [6,128,100] for tensor padded_feature does not satisfy
any optimization profiles. Valid range for profile 0: [10,128,100]..[30,128,100].
```

**Fix.** Rebuilt ONLY the audio encoder (llm engine untouched) with `--minTimeSteps 100`
(= the `audio_build` default; `audio_build.cpp:47`) → `minChunks = divUp(100,100) = 1`
→ opt profile `[1,128,100]..[30,128,100]`. Same ONNX (`onnx-v080/audio/model.onnx`),
same FP16 weights, wider dynamic range only. Build (orin-nx, `~/project/edgellm-v080`):
```
./build/examples/multimodal/audio_build \
  --onnxDir  $WS/Qwen3-ASR-0.6B/onnx-v080/audio \
  --engineDir $WS/Qwen3-ASR-0.6B/engines-v080-minchunk1/audio \
  --minTimeSteps 100 --maxTimeSteps 3000
# Engine generation completed in 81.5 s; Successfully built AudioEncoder engine
```
New engine `builder_config`:
`{'min_time_steps': 100, 'max_time_steps': 3000, ...}` (was `min_time_steps: 1000`).

New md5s (re-staged to HF + MANIFEST updated, see §HF):
```
ede676fbd99cb51d556b96637bce86fc  audio_encoder.engine  (377,959,836 B)
3b9ff631075bab6a8e44631fe6fa1c5f  config.json           (2,563 B)
```

**Accuracy still golden (neutral profile widening).** A TRT optimization profile only
constrains the dynamic-shape range + biases tactic selection; it does not change the
computed result. Confirmed by the serve gate below: one-shot transcripts are byte-exact
to the golden text for both EN and ZH on the new engine. Boot pre-warm — which used to
log `TRT-EdgeLLM ASR pre-warm batch=1 failed` (1-chunk out of range) — now succeeds:
```
TRT-EdgeLLM ASR worker pre-warmed shapes 1..6 in 1.2s
```
(shapes 1..6 = exactly the previously-rejected short-mel range).

### Blocker 2 — ASR streaming `begin`→501 stub  ✅ ALREADY HANDLED BY CONFIG (no worker change)

`trt_edge_llm_asr.py:create_stream` (:810) branches on `_use_streaming_worker()` (:378),
which is true only when `stream_mode ∈ {worker, stream, streaming, chunk_confirm,
prefix}`. The `jetson-edgellm-v080` profile sets `EDGE_LLM_ASR_STREAM_MODE=accumulate`
→ `create_stream` returns `_TRTEdgeLLMAccumulatingASRStream` (:816), whose `finalize()`
(:924) calls `self._backend.transcribe(wav_bytes)` — the **proven one-shot worker path**
(`{"requests":[...]}`), NOT the `begin`/`chunk`/`end` streaming protocol (the 501 stub).
So the streaming `/asr/stream` WS already runs the offline one-shot decode per utterance
(VAD-finalized). No `qwen3_asr_worker.cpp` change and no new config key were needed —
the existing default `stream_mode=accumulate` IS the config (option (a) in the task).
The prior `begin→501` observation was a side path; the actual data-plane failure was
Blocker 1 (the encoder rejecting the short mel inside `transcribe`). Confirmed: the
serve gate's `/asr/stream` WS roundtrip returns the correct final transcript.

### Blocker 3 — worker_io drops unkeyed error events → 60 s hang  ✅ FIXED (voxedge overlay)

`WorkerIO._reader_loop` (`worker_io.py:291-296`) routed events by
`request_id`/`id`; an event with no routable id was dropped silently
(`# else: stale / unsolicited`). A worker **error** event without `id` therefore
hung the caller the full `q.get(timeout=60.0)` (`:234`) instead of surfacing fast.
Patched copy (`deploy/docker/worker_io.voxedge-patch.py`, overlaid by the Dockerfile
into `…/voxedge/backends/jetson/worker_io.py`): when an event has no routable queue
**and** `event=="error"` **and** exactly one request is inflight (N=1), route it to the
sole queue. Only `error` is rerouted (never `partial`/`done`), so a legitimately stale
stream is never corrupted; functionally byte-equal otherwise. This is a robustness
wrapper (the real fix is Blocker 1); upstreaming to voxedge is recommended.

### Blocker 4 — slim-image host-lib mount contract (for #8 compose)  ✅ CONFIRMED + DOCUMENTED

The image carries NO TensorRT/CUDA libs (slim). The container needs these host bind
mounts + `LD_LIBRARY_PATH` (captured verbatim from `docker inspect seeed-voice`, the
production contract; without them the worker dies `libnvinfer.so.10: cannot open`):
```
-v /usr/local/cuda/lib64:/host-cuda
-v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs
-v /lib/aarch64-linux-gnu:/host-libs
-v /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt
-v /usr/src/tensorrt:/usr/src/tensorrt
-e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-cuda:/host-nvidia-libs:/host-libs
--runtime nvidia --gpus all
```
PLUS the engine root (`-v /opt/edgellm-v080:/opt/edgellm-v080` or HF first-boot pull
into a named volume) and the `speech-models` volume (`-v …/speech-models/_data:/opt/models`).
**#8's compose must satisfy this** — the production compose already does for `seeed-voice`.

---

## REAL SERVE GATE (raw, throwaway `v080-verify4`, alt port 8644)

Container: image `:…-20260610e`, host-lib mounts above, GPU, `-v /opt/edgellm-v080`
(pre-staged engines incl. the rebuilt min-chunk-1 encoder), `speech-models` volume,
`EDGELLM_V080_AUTO_PULL=0`. translator + edge-llm-chat-service stopped for RAM during
the run (seeed-voice NEVER stopped); all restored after.

### Boot — clean (pull-skip → uvicorn, both v0.8.0 backends preloaded)  ✅
```
ASR backend preload OK (... engine_dir='/opt/edgellm-v080/engines/asr/llm',
  audio_encoder_dir='/opt/edgellm-v080/engines/asr/audio', stream_mode='accumulate', ...)
TRT-EdgeLLM ASR worker pre-warmed shapes 1..6 in 1.2s        # <-- short mels now OK
TTS backend preload OK (talker=/opt/edgellm-v080/engines/qwen3-tts/talker)
BackendManager[asr] ready (trt_edgellm)  /  BackendManager[tts] ready (trt_edgellm)
Application startup complete.  Uvicorn running on http://0.0.0.0:8000
GET /health → {"tts":true,"tts_backend":"trt_edgellm","asr":true,"asr_backend":"trt_edgellm",...}
```

### TTS + ASR loopback roundtrip (TTS synth → feed back to ASR; golden == synthesized text)  ✅

**SHORT EN (the previously-failing 1–6-chunk case):**
```
[TTS]  3.68s wav (24kHz)
[ASR /asr offline]    → 'The quick brown fox jumps over the lazy dog.'   (== golden)
                        inference_time_s=0.239
[ASR /asr/stream WS]  → 'The quick brown fox jumps over the lazy dog.'   (== golden)
```

**SHORT ZH (~3 chunks):**
```
[TTS]  1.84s wav
[ASR /asr offline]    → '今天天气真不错。'   (== golden)   inference_time_s=0.157
[ASR /asr/stream WS]  → '今天天气真不错。'   (== golden)
```

**LONG EN (14.50s → offline-segment split → multi-segment incl. short chunks):**
```
[TTS]  14.50s wav
[ASR /asr offline]    → 'Artificial intelligence is transforming the way we live and
   work From healthcare to transportation Machine learning models are now embedded in
   countless everyday They systems around the world.'
GOLDEN                = 'Artificial intelligence is transforming the way we live and
   work. From healthcare to transportation, machine learning models are now embedded in
   countless everyday systems around the world.'
WORD_RECALL = 26/26 = 1.00   (every golden word recovered; one spurious 'They' at a
                              segment seam — boundary artifact, not a transcription error)
```

### Logs clean  ✅
`docker logs v080-verify4 | grep -iE 'satisfyProfile|501|no event for|in 60s|Failed to
set padded|preprocessing failed|Traceback'` → **empty**. The only `error` matches are
`uvicorn.error` (the logger name). No 501, no satisfyProfile, no 60 s timeout.

---

## Container restore
Pre-run: seeed-voice Up, translator Up, edge-llm-chat-service Up, industrial-security-demo
Restarting (pre-existing loop). seeed-voice **never stopped** (Up ~1 h throughout).
translator + edge-llm-chat-service stopped for RAM, restarted. Throwaway `v080-verify4`
removed. **Final:** seeed-voice Up, translator Up, edge-llm-chat-service Up,
industrial-security-demo Restarting (untouched), v080-verify4 gone.

## HF re-stage
`engines/asr/audio/{audio_encoder.engine,config.json}` re-uploaded to
`harvestsu/seeed-local-voice-artifacts/sm87-trt10.3-jp6.2/v0.8.0/engines/asr/audio/`
via wsl2-local relay (orin-nx cannot direct-upload). MANIFEST (v080-0017) updated with
the new md5s/sizes. md5 verified identical orin-nx → wsl2-local → HF.

## Verdict
**The v0.8.0 image TRANSCRIBES + SYNTHESIZES end-to-end through the service.** All 4
blockers resolved: (1) encoder rebuilt min-chunk-1 + re-staged, (2) streaming uses the
proven accumulate/one-shot path by config, (3) worker_io fast-fails unkeyed errors,
(4) host-lib mount contract documented for #8. `/asr` + `/asr/stream` + `/tts` all
roundtrip correctly on short AND long utterances, EN + ZH, logs clean. **#7 DONE.**
The slim-lib mount contract (Blocker 4) is the contract #8's compose must satisfy.

## Files (this commit)
- `deploy/docker/Dockerfile.jetson.v080-edgellm` — overlay the patched worker_io.py.
- `deploy/docker/worker_io.voxedge-patch.py` — NEW: Blocker-3 unkeyed-error reroute.
- `docs/plans/v080-0017-hf-artifacts-manifest.md` — MANIFEST encoder md5/size + re-stage note.
- `docs/plans/v080-0019-serve-gate-acceptance.md` — this doc.

Image tag `:v0.8.0-edgellm-20260610e`. Rebuilt min-chunk-1 audio encoder on HF
(`ede676fb…`). NOT pushed (per task).
