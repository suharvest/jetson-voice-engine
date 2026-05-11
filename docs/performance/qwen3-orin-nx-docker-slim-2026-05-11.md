# Qwen3 Orin NX Docker Slim Image — 2026-05-11

Status: **§2 prototype works** — `jetson-voice-qwen3:slim` at **991 MB**
(vs `jetson-voice-speech:v3.5-clean` at **18.2 GB**, **94.6 % smaller**),
TTS streaming + voice clone over HTTP both verified end-to-end on
`orin-nx`.

## Image numbers

| Image | Size | Source |
|---|---|---|
| `jetson-voice-speech:v3.5-clean` | 18.2 GB | Pre-existing production image, bundles CUDA + TRT 10.4 |
| **`jetson-voice-qwen3:slim`** | **991 MB** | New multi-stage build, host-mount JetPack libs |

## Build

Dockerfile: `Dockerfile.slim.qwen3` in the jetson-voice repo.

- Builder stage = `jetson-voice-speech:v3.5-clean` (already pulled).
  Copies out `onnxruntime`, `sherpa_onnx`, `huggingface_hub`,
  `safetensors` python packages.
- Runtime stage = `docker.m.daocloud.io/library/ubuntu:22.04` (109 MB
  local copy; NX docker hub access through aliyun mirror blocks
  `docker.io/library`).
- `apt`: `python3 python3-pip libpython3.10 libasound2 libsndfile1 sox curl`.
- `pip`: `fastapi`, `uvicorn[standard]`, `python-multipart`, `soundfile`,
  `websockets`, `requests`, `numpy<2` (ABI for ort 1.20),
  `piper-phonemize`.
- SONAME symlink for `libonnxruntime.so.1.20.0 → .1`.
- Application source from build context: `app/`, `configs/`, `voices/`.
- Empty mount points: `/opt/edgellm-bin`, `/opt/models/qwen3-edgellm`,
  `/opt/qwen3-edgellm-jetson`.

## Runtime mounts (mandatory)

```bash
docker run -d --name jetson_voice_slim \
  --runtime nvidia --ipc host -p 18092:8000 \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -v ~/project/TensorRT-Edge-LLM/build:/opt/edgellm-bin:ro \
  -v ~/qwen3-models:/opt/models/qwen3-edgellm:ro \
  -v ~/project/qwen3-edgellm-jetson:/opt/qwen3-edgellm-jetson:ro \
  -e LD_LIBRARY_PATH=/host-libs:/host-cuda:/usr/local/lib/python3.10/dist-packages/onnxruntime/capi \
  -e JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx \
  -e QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm \
  -e QWEN3_EDGELLM_JETSON_ROOT=/opt/qwen3-edgellm-jetson \
  -e EDGE_LLM_BASE=/opt/edgellm-bin \
  -e EDGE_LLM_BUILD_DIR=. \
  -e EDGE_LLM_TTS_WORKER_BIN=/opt/edgellm-bin/examples/omni/qwen3_tts_worker \
  -e EDGELLM_PLUGIN_PATH=/opt/edgellm-bin/libNvInfer_edgellm_plugin.so \
  -e EDGE_LLM_TTS_STATEFUL_CODE2WAV=1 \
  -e EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR=/opt/models/qwen3-edgellm/engines/orin-nx/highperf/code2wav_stateful \
  jetson-voice-qwen3:slim
```

Key design decisions:

1. **Host TRT 10.3 mount** instead of bundling TRT in the image. Avoids
   the 10.3-vs-10.4 mismatch that broke `v3.5-clean` (and was the actual
   gotcha behind the 2026-05-11 clean-room TRT misdiagnosis).
2. **`/opt/edgellm-bin` is a bind mount** instead of `COPY` — the
   built worker + plugin live in the EdgeLLM source tree, and rebuilding
   them is much faster than rebuilding the image.
3. **`/opt/qwen3-edgellm-jetson` is a bind mount** — `app/model_downloader`
   invokes `deploy_qwen3_artifacts.py` from there, no need to duplicate
   the deploy script in the image.

## Verified

- `GET /health` returns `200` 9 s after `docker run`:
  ```json
  { "tts": true,
    "tts_backend": "trt_edgellm",
    "tts_capabilities": ["streaming","multi_language","voice_clone","basic_tts"],
    "asr": false,
    "asr_capabilities": [] }
  ```
- `POST /tts/stream` → 215 044 B PCM body (~4.5 s audio @ 24 kHz).
- `POST /tts/clone/stream` with `speaker_embedding_b64` → 222 724 B PCM
  body. Uses the same speaker embedding extracted from
  `bench/wavs/S1.wav` via `tts/speaker_encoder/speaker_encoder.onnx`
  in the §5 gate.
- TTS worker warmup time inside the container: ~13 s (similar to the
  bare-host smoke; stateful Code2Wav + CUDA graph capture dominate).
- Container runtime size (writable layer): 377 kB (image 733 MB virtual,
  991 MB on disk including manifests).

## Known gaps (next step)

- **ASR backend disabled** (`asr: false`): `qwen3_asr_worker` is built
  by the jetson-voice native worker build at
  `~/project/repro-qwen3/jetson-voice/build/edgellm_voice_worker/workers/`
  and is not mounted into the slim container yet. Once that build path
  is added to the bind mount (or the ASR worker baked in), `/asr`
  endpoints come up.
- **Non-streaming `/tts`** (without `stream=true`) returns 500 with
  `Stateful Code2Wav currently requires stream=true`. The Python TTS
  backend should auto-set `stream=true` when the worker is in stateful
  mode; not a blocker for clients that use `/tts/stream` directly.
- **Image not yet pushed** to any registry. Tag and push are deferred
  until the ASR worker is wired in.

## Reproduce

```bash
# On Orin NX with the fork artifacts already in place:
cd ~/project/jetson-voice
docker build -f Dockerfile.slim.qwen3 -t jetson-voice-qwen3:slim .

# (or substitute the upstream docker hub address for the ubuntu base if
# your daemon can reach docker.io directly — see the FROM line in
# Dockerfile.slim.qwen3 for the daocloud override that NX needs.)
```

## Files

- `jetson-voice/Dockerfile.slim.qwen3` (3.1 KB)
- `jetson-voice/scripts/...` — operator may use `run_slim.sh` style
  helper (sample in this report).
