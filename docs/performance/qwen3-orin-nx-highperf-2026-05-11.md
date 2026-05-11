# Qwen3 Orin NX Highperf Real Smoke — 2026-05-11

Status: **PIPELINE OK end-to-end** with the highperf product line.

This is the §1 real-smoke deliverable that the 2026-05-11 clean-room
report blocked on. Together with the FP8 / MHA / CuTe DSL fixes pushed
to the EdgeLLM fork, the published `orin-nx-highperf-2026-05-11`
artifact set now boots cleanly on a fresh Orin NX checkout.

## Environment

- Host: Jetson Orin NX 16GB, JetPack 6 (CUDA 12.6), nvpmodel 40 W (mode 4)
- Host TensorRT: 10.3.0.30 (system `/usr/lib/aarch64-linux-gnu/libnvinfer.so.10`)
- EdgeLLM fork: `suharvest/TensorRT-Edge-LLM`, branch
  `qwen3-tts-highperf-runtime-w8a16` at commit `5cc6060`
  (FP8 text-embedding scale fix + SM87 + CuTe DSL gemm + EMBEDDED_TARGET defaults)
- Worker: `qwen3_tts_worker` md5 `dfe80e62312bc56600d9f67f3e4592cc`
- Plugin: `libNvInfer_edgellm_plugin.so` md5 `079f27b1db5a42cc08017cea66a54f90`
- Artifact set: `orin-nx-highperf-2026-05-11` (25 files, SHA-256 verified)

## Smoke request

Direct `qwen3_tts_worker` JSON-line protocol (no jetson-voice service
wrapper) — proves the underlying runtime is correct without conflating
the docker image issue:

```json
{
  "id": "smoke1",
  "text": "今天我们继续验证低延迟流式生成的效果。",
  "stream": true,
  "stream_only": true,
  "first_chunk_frames": 7,
  "chunk_frames": 10,
  "max_chunk_frames": 10
}
```

Run mode: stateful Code2Wav (`EDGE_LLM_TTS_STATEFUL_CODE2WAV=1`), explicit
KV Talker, native CodePredictor backend.

## Result

```json
{
  "event": "done",
  "audio_complete": true,
  "audio_s": 4.8,
  "samples": 115200,
  "sample_rate": 24000,
  "chunk_count": 7,
  "frames": 60,
  "first_chunk_ms": 583.499,
  "total_ms": 3595.780,
  "generation_ms": 3585.672,
  "code2wav_ms": 150.664,
  "rtf": 0.749,
  "stateful_code2wav": true,
  "stream_only": true,
  "ok": true
}
```

Worker init (cold start): **12 210 ms** (engine deserialize + CUDA graph
capture for Talker + 15 CodePredictor lm_head heads).

Audio file: `docs/audio-evidence/nx-highperf-2026-05-11.wav` (24 kHz mono,
4.8 s).

## What each fix unblocked, in order

1. `code2wav.engine not found` — env `EDGE_LLM_TTS_STATEFUL_CODE2WAV=1` +
   `EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR=.../code2wav_stateful`. The
   non-stateful worker default looks for `code2wav.engine`.
2. `scales must be provided for FP8 embedding table` — fixed in
   EdgeLLM fork commit `c248f73` (`fix(qwen3-tts): pass FP8 dequant
   scales to text embedding lookup`). `loadTalkerWeights` now does
   name-based safetensors lookup and forwards the scale to both
   `embeddingLookup()` call sites.
3. `There must be one kernel to implement the MHA` — fixed in
   `eacfefc` (default `CMAKE_CUDA_ARCHITECTURES=80;86;87;89` for the
   highperf fork branch). The FMHA cubin registry now contains SM87.
4. `CuTe DSL GEMM not compiled` — fixed across `190d977`, `de3939c`,
   `5cc6060` (default `ENABLE_CUTE_DSL=gemm` plus `EMBEDDED_TARGET=
   jetson_orin` + `CUTE_DSL_ARTIFACT_TAG=sm_87` so `cute_dsl_setup()`
   resolves the prebuilt sm_87 artifact tarball without operator-side
   flags).
5. `Stateful Code2Wav currently requires stream=true` — request must use
   `"stream": true`. Documented in the smoke request schema above.
6. `Failed to reshape stateful Code2Wav input codes` — the request must
   declare `first_chunk_frames` / `chunk_frames` / `max_chunk_frames`
   matching the engine's optimization profile (7/10/10 for the
   2026-05-11 set). Worker default of 25 frames exceeds `maxCodeLen`.

After all six, the worker accepts the request and emits 7 connected PCM
chunks ending with `is_final=true` + a `done` event with the metrics
above.

## Reproduce

```bash
# All three repos are public:
cd ~/project
git clone https://github.com/suharvest/jetson-local-voice.git jetson-voice
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
git clone https://github.com/suharvest/TensorRT-Edge-LLM.git
cd jetson-voice && git checkout qwen3tts-accurate-20260507 && cd ..
cd TensorRT-Edge-LLM && git checkout qwen3-tts-highperf-runtime-w8a16
git submodule update --init --recursive
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr -DCUDA_DIR=/usr/local/cuda-12.6 -DCUDA_CTK_VERSION=12.6
cmake --build build --target edgellmCore NvInfer_edgellm_plugin qwen3_tts_worker -j1
cd ../qwen3-edgellm-jetson
python3 scripts/deploy_qwen3_artifacts.py \
  --set orin-nx-highperf-2026-05-11 \
  --root "$HOME/qwen3-models" --verify-sha256

ROOT="$HOME/qwen3-models"
WORKER=~/project/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_worker
PLUGIN=~/project/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
export EDGELLM_PLUGIN_PATH=$PLUGIN
export EDGE_LLM_TTS_STATEFUL_CODE2WAV=1
export EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR=$ROOT/engines/orin-nx/highperf/code2wav_stateful

echo '{"id":"smoke","text":"...","stream":true,"stream_only":true,"first_chunk_frames":7,"chunk_frames":10,"max_chunk_frames":10}' | \
  $WORKER \
    --talkerEngineDir="$ROOT/tts/talker" \
    --qwen3TtsTalkerBackend=qwen3_tts_explicit_kv \
    --qwen3TtsTalkerEngine="$ROOT/engines/orin-nx/highperf/talker_w8a16_outputk/talker_decode_w8a16_outputk.engine" \
    --codePredictorEngineDir="$ROOT/engines/orin-nx/highperf/code_predictor/cp_dir" \
    --codePredictorBackend=qwen3_tts_native \
    --code2wavEngineDir="$ROOT/engines/orin-nx/highperf/code2wav_stateful" \
    --tokenizerDir="$ROOT/tts/tokenizer"
```

## Pending follow-ups

- **§5 voice clone real-embedding gate** — `speaker_encoder.onnx` is now
  on HF (`tts/speaker_encoder/`); next step is to extract a real 1024-d
  embedding and feed it through `/tts/clone`. Worker protocol already
  accepts `speaker_embedding_b64`.
- **§2 docker image rebuild** — `jetson-voice-speech:v3.5-clean`
  bundles TRT 10.4 and conflicts with the TRT 10.3 engines. The slim
  pattern (host TRT mount, 17.7 GB → 898 MB pre-qwen3) needs to be
  re-applied with the qwen3 worker + plugin baked in.
- **Listening audit** — `nx-highperf-2026-05-11.wav` not yet auditioned
  for content correctness vs the prompt.
