# v080-0018 — v0.8.0 Docker image: entrypoint CMD fix + boot/serve acceptance

**Date:** 2026-06-10
**Host:** orin-nx (`Linux 5.15.148-tegra aarch64`, hostname `orinnx`, Orin NX 16GB, JetPack 6.2)
**Image:** `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.8.0-edgellm-20260610c` (1.26GB, thin overlay FROM `:prod-unified-v8`)
**Scope:** Task #7 — make the v0.8.0 edgellm Docker image actually boot **and serve** after the first-boot HF pull (the prior image pulled engines then exited 0, never starting uvicorn).

---

## 1. Root cause of "pulls then exits, never serves"

The prior image (`:v0.8.0-edgellm-20260610`, no suffix) overrode `ENTRYPOINT` to the v080
first-boot script but **did not re-declare `CMD`**. Docker resets the inherited `CMD`
to null whenever `ENTRYPOINT` is (re)declared in a child build. So the chain

```
entrypoint.jetson.v080.sh
  → /opt/speech/pull_v080_artifacts.sh   (pulls engines — works)
  → exec /opt/speech/entrypoint.jetson.sh "$@"   ($@ EMPTY, no CMD)
       canonical entrypoint ends with: exec "$@"   → exec nothing → EXIT 0
```

`docker inspect` of the broken image confirmed `CMD=null`. The entrypoint script
itself already had the correct `exec .../entrypoint.jetson.sh "$@"` chain — the bug
was purely the missing CMD.

### Fix (Dockerfile.jetson.v080-edgellm)

```dockerfile
ENTRYPOINT ["/opt/speech/entrypoint.jetson.v080.sh"]
CMD ["python3", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Re-declaring the uvicorn CMD makes it flow through the chain so uvicorn actually runs
after the pull. `entrypoint.jetson.v080.sh` unchanged (its `exec ...jetson.sh "$@"`
was already correct).

---

## 2. Profile + env wiring (so the migrated stack selects v0.8.0 backends + paths)

`server/core/asr_backend.py:create_asr_backend` resolves the backend **class** from
`current_profile()["asr_backend"]` and RAISES if no profile is loaded. So a profile
IS required. The image now bakes `OVS_PROFILE=jetson-edgellm-v080` (new profile,
added in this commit).

### Why a NEW profile, not `jetson-multilang-highperf-nx`

The v0.8.0 `qwen3_tts_worker` is the **generic `LLMEngineRunner`**. Its CLI accepts
ONLY `--talkerEngineDir/--codePredictorEngineDir/--tokenizerDir/--code2wavEngineDir`.
The highperf-nx profile sets the **explicit-KV** keys (`EDGE_LLM_TTS_TALKER_BACKEND=
qwen3_tts_explicit_kv`, `..._TALKER_ENGINE`, `..._CODE_PREDICTOR_BACKEND`,
`..._TEXT_PROJECTION`, `..._PROMPT_KV_CACHE`) + `OVS_TTS_WORKER_CONCURRENCY=2`. The
voxedge wheel emits `--qwen3TtsTalkerBackend` / `--qwen3TtsTalkerEngine` / `--max_slots`
from those, and the v0.8.0 worker aborts:

```
RuntimeError: TTS worker failed to start:
  /opt/jv-workers/qwen3_tts_worker: unrecognized option '--qwen3TtsTalkerBackend'
```

`jetson-edgellm-v080` declares `asr_backend/tts_backend=jetson.trt_edge_llm`,
`LANGUAGE_MODE=multilanguage`, and DELIBERATELY OMITS the 5 explicit-KV keys + sets
`OVS_TTS_WORKER_CONCURRENCY=1` → all-empty → generic-runner path (no flags) →
worker starts. (Wheel docstring: "All-empty (default) → no flags … byte-equivalent at N=1".)

### deploy_paths env precedence (verified at runtime)

`profile_loader._snapshot_operator_keys()` marks every `EDGE_LLM_*`/`QWEN3_*` env key
present at import (non-empty) as operator-owned; `apply_profile` will not overwrite
operator-owned keys. So the image-baked v0.8.0 paths WIN over any profile value.
The `jetson-edgellm-v080` profile carries NO old-layout engine-path keys, so there is
no conflict at all (clean apply, 0 override warnings):

```
[INFO] profile_loader: Applied profile jetson-edgellm-v080 from
       /opt/speech/configs/profiles/jetson-edgellm-v080.json (40 env keys; 0 stale cleared)
```

Baked engine-DIR env (all point at the runtime-pulled `/opt/edgellm-v080/engines`):
`EDGE_LLM_ASR_ENGINE_DIR=/opt/edgellm-v080/engines/asr/llm`,
`EDGE_LLM_ASR_AUDIO_ENC_DIR=/opt/edgellm-v080/engines/asr/audio`,
`EDGE_LLM_TTS_TALKER_DIR / _CP_DIR / _CODE2WAV_DIR` under `engines/qwen3-tts/`,
`EDGELLM_PLUGIN_PATH = EDGE_LLM_ASR_PLUGIN_PATH = .../plugins/libNvInfer_edgellm_plugin.so`
(shared ASR+TTS — matches the slim pull-set). The explicit-KV / engine-FILE keys
(`EDGE_LLM_TTS_TALKER_ENGINE` etc.) are intentionally NOT baked.

The backend preload confirms every path resolves to v0.8.0:
```
ASR backend preload OK (... worker_binary='/opt/jv-workers/qwen3_asr_worker',
  engine_dir='/opt/edgellm-v080/engines/asr/llm',
  audio_encoder_dir='/opt/edgellm-v080/engines/asr/audio',
  plugin_path='/opt/edgellm-v080/plugins/libNvInfer_edgellm_plugin.so', ...)
TTS backend preload OK (binary=/opt/jv-workers/qwen3_tts_worker
  talker=/opt/edgellm-v080/engines/qwen3-tts/talker)
```

---

## 3. Runtime requirements discovered (slim-image host-lib mounts)

The image is slim — it carries NO TensorRT/CUDA libs. They come from host bind-mounts
exactly as the production `seeed-voice` container (`docker inspect seeed-voice`):

```
-v /usr/local/cuda/lib64:/host-cuda
-v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs
-v /lib/aarch64-linux-gnu:/host-libs
-v /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt
-v /usr/src/tensorrt:/usr/src/tensorrt
-e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-cuda:/host-nvidia-libs:/host-libs
```

Without them the worker fails: `qwen3_tts_worker: error while loading shared libraries:
libnvinfer.so.10: cannot open shared object file`. These mounts + LD_LIBRARY_PATH MUST
be set in the #8 compose (the production compose already does this for seeed-voice).

---

## 4. BOOT + SERVE verification (raw)

Throwaway `v080-verify2`, alt port 8642, image `:...-20260610c`, production host-lib
mounts + `speech-models` volume + `/opt/edgellm-v080` engine-cache mount, GPU.
(translator + edge-llm-chat-service stopped for RAM during the run, restored after.)

### (a) Coherent boot — pull → uvicorn, no crash-loop  ✅ PASS
```
[v080-pull] v0.8.0 engines already present (.../asr/llm/llm.engine) — skipping HF pull.
[INFO] uvicorn.error: Started server process [1]
[INFO] profile_loader: Applied profile jetson-edgellm-v080 (40 env keys; 0 stale cleared)
[INFO] asr_backend: Creating ASR backend jetson.trt_edge_llm
[INFO] trt_edge_llm_asr: ASR backend preload OK (...v0.8.0 paths...)
[INFO] trt_edge_llm_tts: TTS backend preload OK (...v0.8.0 paths...)
[INFO] backend_manager: BackendManager[asr] ready (trt_edgellm)
[INFO] backend_manager: BackendManager[tts] ready (trt_edgellm)
[INFO] server.main: Speech service ready.
[INFO] uvicorn.error: Application startup complete.
[INFO] uvicorn.error: Uvicorn running on http://0.0.0.0:8000
```
First-boot HF pull (clean run, empty cache) also verified end-to-end: 23 files,
`Fetching 23 files: 100%`, `snapshot_download → /tmp/v080-hf`, staged into
`/opt/edgellm-v080/engines/...`, then chained to uvicorn (no exit).

### (b) /asr/capabilities + /tts/capabilities = edgellm backend  ✅ PASS
```
GET /health  → {"tts":true,"tts_backend":"trt_edgellm","asr":true,"asr_backend":"trt_edgellm",...}
GET /asr/capabilities → {"backend":"trt_edgellm","capabilities":["multi_language","streaming","offline"],"sample_rate":16000}
GET /tts/capabilities → {"backend":"trt_edgellm","model_id":"qwen3-tts","supports_voice_cloning":true,"sample_rate":24000,...}
```

### (c) ASR roundtrip (real audio → transcript)  ❌ FAIL — engine-artifact defect (NOT entrypoint/wiring)

Every real `/asr` request hangs 60s → 500. Root cause traced to the **v0.8.0 audio
encoder engine optimization profile**, NOT the image:

```
[ERROR] [TensorRT] IExecutionContext::setInputShape: Error Code 3: API Usage Error
  (condition: satisfyProfile. Set dimension [6,128,100] for tensor padded_feature does
   not satisfy any optimization profiles. Valid range for profile 0: [10,128,100]..[30,128,100].)
[ERROR] [audioRunner.cpp:420] Failed to set padded features input shape
[ERROR] [llmInferenceRuntime.cpp:1007] Audio preprocessing failed. This request cannot be handled.
```

The audio encoder accepts only **10..30 audio chunks** (~10–30 s). But:
- the service segments audio at `segment_cap_sec=5.5` and offline-splits long audio,
  so every segment yields **1–6 chunks** → below the 10-chunk minimum;
- the boot pre-warm uses a 1-chunk synthetic mel → also out of range
  (`TRT-EdgeLLM ASR pre-warm batch=1 failed`, non-fatal);
- a 12.6 s real WAV is split into 3.8 s / 1.6 s segments →
  `TRT-EdgeLLM ASR offline segment failed (3.8s)` repeatedly.

The worker IS functional (loads engine, `{"event":"ready","init_ms":~7300,"max_slots":1}`,
returns `{"event":"done",...,"total_ms":378}` for a trivial input), but cannot serve
real utterances with this engine. **The v0.8.0 `asr/audio/audio_encoder.engine` must be
rebuilt with a min audio-chunk profile of 1 (or the service must pad short audio to
≥10 chunks).** Separate engine-rebuild workstream.

Secondary robustness issue: when the worker emits an error event without `id`, voxedge
`worker_io._reader_loop` drops it silently ("stale/unsolicited") → the caller hangs the
full 60 s instead of surfacing the error. Worth a Python-side fix (route an unkeyed
event to the sole inflight queue at N=1, or have the worker echo `id` on every error)
but it is a robustness wrapper around the real engine defect, not the cause.

### (d) /tts roundtrip  ⚠️ NOT VERIFIED (blocked by RAM)
TTS preloads OK and `/tts/capabilities` serves, but a full `/tts` synth roundtrip was
not captured: the stuck ASR request loop + TTS worker + production trio exhausted the
16 GB (the throwaway was OOM-killed, exitCode 137, once during the run). TTS is
independent of the broken audio encoder so it is expected to work, but this is unproven.

---

## 5. Container restore

Pre-run snapshot: seeed-voice Up, translator Up (healthy), edge-llm-chat-service Up
(healthy), industrial-security-demo Restarting (pre-existing loop). **seeed-voice was
NEVER stopped** (Up 57 min throughout). translator + edge-llm-chat-service were stopped
for RAM and restarted. Throwaway `v080-verify2` (+ a stray `friendly_wilbur`, old
`v080-verify`) removed. **Final:** seeed-voice Up, translator Up (healthy),
edge-llm-chat-service Up (healthy), industrial-security-demo Restarting (untouched).

---

## 6. Verdict

- **Entrypoint CMD fix: DONE + PROVEN.** The image boots coherently (pull → uvicorn,
  no crash-loop), preloads both v0.8.0 backends with all-v0.8.0 paths, and serves
  `/health`, `/asr/capabilities`, `/tts/capabilities` as `trt_edgellm`. Gates (a),(b) PASS.
- **NOT yet a deployable end-to-end stack:** the v0.8.0 ASR audio-encoder engine cannot
  process real (short) utterances — its TRT optimization profile requires 10–30 audio
  chunks while the serving path produces 1–6. ASR roundtrip (gate c) FAILS on an engine
  artifact, not on the Docker/entrypoint/profile wiring (which is correct). TTS roundtrip
  (gate d) unverified (RAM).
- **Ready for #8 compose?** The image + entrypoint + profile + the required host-lib
  mounts are correct and can be wired into compose. BUT a working ASR roundtrip first
  needs the audio_encoder engine rebuilt with a min-chunk-1 profile (or a service-side
  short-audio pad). Recommend gating #8 on that engine fix.

### Files (this commit)
- `deploy/docker/Dockerfile.jetson.v080-edgellm` — CMD re-declared; OVS_PROFILE +
  v0.8.0 engine-DIR env; explicit-KV/engine-FILE keys removed; new profile COPY.
- `deploy/docker/entrypoint.jetson.v080.sh` — unchanged (chain was already correct).
- `deploy/docker/pull_v080_artifacts.sh` — unchanged (pull verified end-to-end).
- `configs/profiles/jetson-edgellm-v080.json` — NEW generic-runner v0.8.0 profile.
