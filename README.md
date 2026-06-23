# jetson-voice-engine

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano%20%7C%20Orin%20NX-76b900.svg" alt="Platform">
  <img src="https://img.shields.io/badge/ASR-Qwen3--ASR%20int4%20%7C%20Paraformer-2f80ed.svg" alt="ASR">
  <img src="https://img.shields.io/badge/TTS-Qwen3--TTS%20%7C%20Matcha%20TRT%20%7C%20Kokoro-f97316.svg" alt="TTS">
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
</p>

<p align="center">
  Qwen3-TTS · Matcha TRT · Qwen3-ASR on Jetson Orin Nano/NX — <strong>RTF 0.63 · 28ms TTFA · voice clone · no cloud</strong>
</p>

<!-- TODO: Add demo GIF — record a ~15s terminal session showing:
     1. bash scripts/reproduce_qwen3_highperf.sh running (one command)
     2. verify_reproduction.sh output: "All checks passed"
     3. curl /tts/stream → PCM output (28ms TTFA)
     Suggested tool: asciinema + agg -->

## What is this?

jetson-voice-engine is the **Jetson speech engine build toolkit** for [OpenVoiceStream](https://github.com/suharvest/openvoicestream). It takes official Hugging Face model weights (Qwen3-TTS/ASR, Matcha, Kokoro, Paraformer) and produces validated TensorRT engines + runtime artifacts that OpenVoiceStream deploys on Jetson Orin Nano and NX.

One script handles the entire chain — ONNX export, engine build, HF artifact download, Docker start, and loopback verification. Exit 0 means the whole stack is working.

> **For deployment:** use [OpenVoiceStream](https://github.com/suharvest/openvoicestream)'s `deploy/install.sh --target jetson`. This repo is consumed as its `third_party/qwen3-edgellm-jetson` submodule and is intended for engine contributors and platform engineers.

## Features

- **Qwen3-TTS int4-AWQ** — RTF 0.63 on Orin NX, ~10% faster than FP16; voice clone via pre-extracted speaker embedding; 52-language
- **Matcha TRT** — 28ms TTFA on Orin NX via `/tts/stream`; RTF 0.015–0.021 (50–70× real-time); TRT vocos vocoder
- **Qwen3-ASR int4** — ~55ms latency on Orin NX, ~60ms on Orin Nano; 52-language; int4 confirmed stable (FP16 hangs on NX)
- **Kokoro TRT** — hybrid prefix TRT + ORT suffix; RTF 0.54 on Orin Nano; speaker 52
- **One-script reproduction** — `reproduce_qwen3_highperf.sh` from HF weights to verified service with SHA-256 gate, symbol check, and TTS→ASR loopback
- **Thin engine-overlay** — NVIDIA TRT-Edge-LLM pinned at `v0.7.1` + 8 theme patches as a thin overlay, no vendored source tree
- **JetPack 6 / TensorRT 10 / SM87** — production-validated on Orin Nano Super 8 GB and Orin NX 16 GB

## Quick Start — Qwen3 one-shot on Orin NX

```bash
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
bash qwen3-edgellm-jetson/scripts/reproduce_qwen3_highperf.sh
# add: --reference path/to/24kHz_mono.wav   to also verify voice clone
```

`reproduce_qwen3_highperf.sh` is idempotent and handles the whole Qwen3 chain:
clones repos at validated branches, builds EdgeLLM, downloads + SHA-256-verifies
HF artifacts, builds the slim Docker image, starts the service, then runs
`verify_reproduction.sh` which gates on:

1. W8A16 plugin symbol set
2. Artifact SHA-256 integrity
3. HTTP TTS→ASR loopback on 3 Chinese prompts (LCS ≥ 0.7, up to 3 retries)
4. Voice clone via a real reference WAV

Exit 0 = the whole chain is verified. On failure it prints which check failed.

**Prerequisites:** Jetson Orin NX, JetPack 6 (TensorRT 10.3.0.30, CUDA 12.6),
`docker` + `--runtime nvidia`, ~10 GB free disk for HF artifacts.

See `docs/reproduce-from-zero.md` for the manual step-by-step fallback.

## Table of Contents

- [Performance](#performance)
- [Repository layout](#repository-layout)
- [engine-overlay/ — TRT-Edge-LLM fork overlay](#engine-overlay----trt-edge-llm-fork-overlay)
- [Per-model build scripts](#per-model-build-scripts-models)
- [ONNX export](#onnx-export)
- [HF artifact repo](#hf-artifact-repo)
- [Jetson Voice integration](#jetson-voice-integration)
- [Qwen3 runtime profiles & EdgeLLM branches](#qwen3-runtime-profiles--edgellm-branches)
- [Qwen3 voice clone](#qwen3-voice-clone)
- [Current Qwen3 baseline](#current-qwen3-baseline-2026-05-11)
- [Benchmarking](#benchmarking)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Performance

Measured 2026-06-22. JetPack 6, TensorRT 10. All RTF = synthesis time / audio duration
(lower is better; < 1.0 means faster than real-time). TTFA = time to first audio chunk
via HTTP `/tts/stream` (true streaming, measured with `bench/bench_http_tts_stream.py`).
Direct-backend latency (no service) measured with `bench/bench_*.py`, 3 trials after 1 warmup.

**Devices**:
- **Nano** — Jetson Orin Nano Super 8 GB, SM87
- **NX** — Jetson Orin NX 16 GB, SM87

### TTS — all backends

| Backend | Languages | Nano TTFA | Nano RTF | NX TTFA | NX RTF |
|---------|-----------|:---------:|:--------:|:-------:|:------:|
| **Matcha + Vocos TRT** | zh, en | ❌ no vocos TRT | — | **28–75 ms** | **0.015–0.021** |
| **Kokoro TRT** (hybrid) | en | — | 0.54–0.55 | ❌ no artifact | — |
| **Qwen3-TTS FP16** (v0.8.0) | zh, en, 52 lang | ~1.1 s¹ | 0.73 | ~1.2 s¹ | 0.70–0.71 |
| **Qwen3-TTS int4-AWQ** (v0.8.0) | zh, en, 52 lang | ~0.7 s¹ | **0.65** | ~0.65 s¹ | **0.63** |

> ¹ Qwen3-TTS TTFA estimated from IPC worker total-synthesis time ÷ first-chunk fraction.
> True HTTP streaming TTFA for Qwen3-TTS not yet measured (Matcha is active on NX).
> int4-AWQ is **~10–11% faster** than FP16. fp8 text embedding blocked (missing scale tensors).

> **Matcha** — NX: full-ORT acoustic + TRT vocos, 16 kHz, preload 2.9 s. Nano: ❌ missing
> `vocos_fp16.engine` (build: `trtexec --onnx=vocos-16khz-univ.onnx --fp16 --saveEngine=...`).

> **Qwen3-TTS** — v0.8.0 worker (`voxedge explicit-KV` confirmed via `strings`).
> FP16: Nano preload 7.5 s, NX preload 6.1 s. int4-AWQ: talker 235 MB vs 526 MB FP16.

#### Matcha TRT — NX HTTP streaming detail (`/tts/stream`)

| Text | TTFA mean | TTFA min | Total mean | RTF |
|------|:---------:|:--------:|:----------:|:---:|
| "你好，世界！" | 32 ms | 28 ms | 36 ms | 0.020 |
| "今天天气怎么样？" | 32 ms | 29 ms | 37 ms | 0.019 |
| "请告诉我最近的新闻。" | 44 ms | 43 ms | 49 ms | 0.021 |
| "Hello, how are you today?" | 38 ms | 34 ms | 44 ms | 0.017 |
| "欢迎使用语音合成服务，希望您有一个愉快的体验。" | 56 ms | 42 ms | 70 ms | 0.015 |
| "TensorRT accelerates deep learning…" | 75 ms | 52 ms | 88 ms | 0.015 |

#### Qwen3-TTS — direct backend detail (IPC worker, total synthesis time)

| Text | Nano FP16 | Nano int4 | NX FP16 | NX int4 |
|------|:---------:|:---------:|:-------:|:-------:|
| "你好，世界！" | 1065 ms / RTF 0.74 | 1311 ms / RTF 0.655 | 1197 ms / RTF 0.713 | 1259 ms / RTF 0.629 |
| "今天天气怎么样？" | 1064 ms / 0.739 | 1257 ms / 0.655 | 1137 ms / 0.711 | 1259 ms / 0.630 |
| "请告诉我最近的新闻。" | 1527 ms / 0.734 | 1413 ms / 0.655 | 1531 ms / 0.709 | 1407 ms / 0.628 |
| "Hello, how are you today?" | 1816 ms / 0.732 | 1672 ms / 0.653 | 1979 ms / 0.707 | 1209 ms / 0.630 |
| "欢迎使用语音合成服务…" | 3149 ms / 0.729 | 2968 ms / 0.651 | 2926 ms / 0.704 | 1358 ms / 0.629 |
| "TensorRT accelerates…" | 3555 ms / 0.729 | 4884 ms / 0.650 | 4099 ms / 0.702 | 3303 ms / 0.626 |

### ASR — all backends

| Backend | Languages | Nano latency | NX latency |
|---------|-----------|:------------:|:----------:|
| **Qwen3-ASR int4** (v0.8.0 worker) | 52 | ~60 ms¹ | ~55 ms¹ |
| **Qwen3-ASR FP16** (v0.8.0 worker) | 52 | ~60 ms¹ | ❌ hangs |
| **Paraformer TRT** | zh, en, ja, ko | ❌ no engine | ❌ no decoder engine |

> ¹ Synthetic 2 s WAV (440 Hz tone, 16 kHz). Empty transcript; latency = worker roundtrip overhead.
> Real-speech latency scales with audio length.

> **Qwen3-ASR int4** — Nano: worker v0.8.0 (`be7bee91`), int4 engines, preload 7.96 s.
> NX: same worker, int4 engines (LLM 526 MB + audio encoder 364 MB).
> NX FP16 hangs silently — device-specific TRT runtime issue; int4 confirmed stable on both.

## Repository layout

| Dir / file | What it is |
|---|---|
| `engine-overlay/` | **TRT-Edge-LLM fork overlay** (formerly the `voxedge-engine` repo). Carries the NVIDIA upstream pin + our additions *without* vendoring the full source tree. See below. |
| `models/qwen3/` | Qwen3 ASR/TTS export, engine build, vocab pruning, CustomVoice binary build, and roundtrip/contract verification scripts. |
| `models/kokoro/` | Kokoro TRT engine build (direct / hybrid-prefix / long-bucket / split-generator) + experiments. |
| `models/matcha/` | Matcha acoustic + estimator engine build and ONNX surgery scripts. |
| `models/moss-tts-nano/` | MOSS-TTS-Nano engine build, codec ONNX `If`-rank fix, paged-KV FP16 patch, vendored assets. |
| `models/paraformer/` | Paraformer ASR TRT build + decoder ONNX surgery. |
| `models/common/` | Cross-model helpers: ONNX subgraph/STFT surgery, engine-bundle builder, parity gate, model downloader, perf setup, v2v debug tools. |
| `patches/product/` | Product-level source patches applied on top of the engine build (qwen3-tts text-embedding FP8 variants, paraformer EOF fix). |
| `configs/profiles/` | Jetson Voice deployment profiles (`multilanguage-qwen-{highperf,highperf-nx,official}.json`), all using `${QWEN3_ARTIFACT_ROOT}` so they stay portable. |
| `deploy/artifacts/` | Qwen3 HF artifact manifest (`qwen3_manifest.json`) + SHA-256/size sidecar (`qwen3_checksums.json`). |
| `deploy/audio_preprocessing/` | Shared mel filters + Whisper feature-extractor config used by the ASR worker. |
| `native/edgellm_voice_worker/` | Resident C++ ASR/TTS worker sources (mel extractor, VAD split, qwen3 ASR/TTS workers, kissfft) used by Jetson Voice. |
| `scripts/` | Qwen3 orchestration + reproduction entry points (`reproduce_qwen3_highperf.sh`, `verify_reproduction.sh`, artifact packager/downloader, ONNX export, speaker-embedding extraction). |
| `docs/` | Export guides, reproduction-from-zero, performance records + listenable audio evidence, open issues, plans/negative results. |
| `bench/` | Benchmark scripts: direct-backend (RTF/latency) and HTTP streaming (TTFA). |
| `tests/` | `golden_mels/` fixtures for mel parity. |
| `HF_ARTIFACTS.md` | What belongs in the HF artifact repo + stage/upload commands. |
| `AGENTS.md` | Concise operating guide for coding agents in this repo. |

For the end-to-end picture across product + engine + artifacts, see
OpenVoiceStream's `docs/REPRODUCE.md`.

## engine-overlay/ — TRT-Edge-LLM fork overlay

A thin **overlay** of the NVIDIA TensorRT-Edge-LLM engine. It reconstructs the
full patched source tree at build time instead of vendoring it:

```
engine-overlay/
  UPSTREAM_PIN       # exact NVIDIA commit (364769036… = tag v0.7.1)
  upstream.remote    # https://github.com/NVIDIA/TensorRT-Edge-LLM.git
  addon/             # 40 new files upstream does not have, at original relative paths
  patches/           # 8 theme patches (0001..0008) applied over the pin → 27 modified files
  build.sh           # clone upstream@pin → copy addon → apply patches → build (Jetson host)
  manifests/         # build-reproduction manifests (qwen3-tts / qwen3-asr / customvoice)
  DIVERGENCE.md      # per-topic (a) upstreamable / (b) carried classification + PR/retirement plan
  README.md          # overlay model, addon-vs-patch discipline, reproduction
```

Patches `0001..0008` cover: Orin/Tegra build compat, weight-streaming budget,
ASR streaming session, TTS slot-pool concurrency, CustomVoice language
conditioning, server SSE-disconnect + OpenAI API, OpenAI API docs, and misc
example registration. Manifests pin reproducible builds for `qwen3-tts-highperf-sm87`,
`qwen3-asr-sm87`, and `customvoice-v071`.

Reproduce the patched tree (no CUDA needed) or do a full Jetson build:

```bash
cd engine-overlay
./build.sh --apply-only                              # materialize patched source only
./build.sh manifests/qwen3-tts-highperf-sm87.toml    # full build on Orin / sm_87
```

> **Build-verify is DEFERRED** until run on a Jetson CUDA/TensorRT host (Orin,
> sm_87). `build.sh` refuses to compile on non-aarch64. See
> `engine-overlay/README.md` and `engine-overlay/DIVERGENCE.md` for details,
> including the **SSE/client-disconnect fix (patch `0006`) which is
> PR-pending — do NOT auto-submit**.

## Per-model build scripts (`models/`)

Each `models/<model>/` directory holds the export + engine-build scripts for
that backend. They take official model weights / ONNX and emit the TensorRT
engines and sidecars the product service loads:

- `qwen3/` — ASR thinker + audio-encoder engines, TTS Talker (W8A16) /
  CodePredictor / stateful Code2Wav engines, vocab pruning, CustomVoice binary,
  and `verify_*` roundtrip/contract gates.
- `kokoro/` — Kokoro TRT engines (direct / hybrid-prefix / long-bucket /
  split-generator paths).
- `matcha/` — Matcha acoustic + estimator engines.
- `moss-tts-nano/` — MOSS-TTS-Nano engines (incl. codec ONNX `If`-rank fix and
  paged-KV FP16 patch).
- `paraformer/` — Paraformer ASR engines + decoder surgery.
- `common/` — shared ONNX surgery, engine-bundle builder, parity gate, and
  the model downloader.

## ONNX export

Start from official Qwen3 ASR/TTS Hugging Face snapshots and generate ONNX locally:

```bash
bash scripts/setup_trt_export_env.sh
scripts/export_qwen3_asr_onnx.sh --model-dir /models/Qwen3-ASR-0.6B --out /tmp/qwen3-asr-onnx
scripts/export_qwen3_tts_onnx.sh --model-dir /models/Qwen3-TTS-0.6B --out /tmp/qwen3-tts-onnx
```

See `docs/export-from-official-weights.md` for the uv environment, Qwen package
dependencies, and highperf post-processing details.

## HF artifact repo

Runtime TensorRT/embedding artifacts live in the HF model repo
<https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts>; ONNX files
are reproducible intermediate build products generated locally from official
Qwen weights. Expected layout and required files are in
`deploy/artifacts/qwen3_manifest.json`. The shared `tts/tokenizer/` directory
must include `tokenizer.json`, `tokenizer_config.json`, and
`processed_chat_template.json`; missing sidecars break the C++ tokenizer load
on a from-zero device.

See `HF_ARTIFACTS.md` for what belongs in the artifact repo and the
stage/upload flow (`scripts/package_qwen3_artifacts.py` + `hf upload`).

Current Qwen3 HF publication status:

| Artifact set | Status | Notes |
|---|---|---|
| `orin-nano-highperf-2026-05-10` | complete | Product highperf Nano artifact set. |
| `orin-nx-highperf-2026-05-11` | complete | Product highperf NX-native artifact set. |
| `orin-nano-official-2026-05-10` | complete | Official/minimal Nano artifact set. |

## Jetson Voice integration

Preferred: let `scripts/reproduce_qwen3_highperf.sh` build and start
everything. For manual deployment of an existing image:

```bash
JETSON_VOICE_PROFILE=multilanguage-qwen-highperf-nx \
QWEN3_HF_REPO_ID=harvestsu/qwen3-edgellm-jetson-artifacts \
docker compose -f deploy/docker-compose.yml up -d
```

(`multilanguage-qwen-highperf` for Nano, `multilanguage-qwen-highperf-nx` for NX.)

Sanity-check the running service in one line:

```bash
bash scripts/verify_reproduction.sh \
    --plugin /opt/edgellm-bin/libNvInfer_edgellm_plugin.so \
    --artifact-root /opt/models/qwen3-edgellm \
    --service-url http://localhost:18092 \
    [--embedding /tmp/precomputed_speaker_emb.b64]
```

## Qwen3 runtime profiles & EdgeLLM branches

Two Qwen profiles are maintained:

- `official`: minimal-diff EdgeLLM-compatible path for correctness and upstream review.
- `highperf`: product path for low-latency Qwen3 ASR + Qwen3 TTS dual residency on Orin.

Jetson Voice consumes these via the JSON profiles under `configs/profiles/` and
the artifact manifest `deploy/artifacts/qwen3_manifest.json`.
`multilanguage-qwen-highperf` targets the Nano artifact set;
`multilanguage-qwen-highperf-nx` targets NX-native engines.

The corresponding TensorRT-Edge-LLM fork branches:

- `official-qwen3-tts-upstream-runtime`: minimal-diff correctness/runtime branch.
- `qwen3-tts-highperf-runtime-w8a16`: product high-performance branch for the
  current Orin highperf artifacts (explicit Qwen3-TTS backend, W8A16 plugin/runtime,
  CP runtime optimizations + GPU CP kernels, stateful Code2Wav runner). These
  runtime changes are also captured as `engine-overlay/` patches above.

Do not deploy highperf artifacts against EdgeLLM `main`.

## Qwen3 voice clone

> **Note:** voice cloning is the **base Qwen3-TTS** path only. **Qwen3-CustomVoice**
> (`customvoice-v071`) does **not** support voice cloning — it serves a fixed set
> of built-in speakers. Use the base Qwen3-TTS engines for `/tts/clone/stream`.

Pre-extract the speaker embedding on a workstation (librosa + onnxruntime)
once per voice and pass the base64 to the worker. Don't run the speaker encoder
per request.

```bash
python3 scripts/extract_speaker_embedding.py \
    reference.wav speaker_encoder.onnx speaker_emb.b64
curl -X POST http://localhost:18092/tts/clone/stream \
    -H 'content-type: application/json' \
    -d "{\"text\":\"中文文本\",\"speaker_embedding_b64\":\"$(cat speaker_emb.b64)\",\"first_chunk_frames\":7,\"chunk_frames\":10,\"max_chunk_frames\":10}" \
    -o clone.pcm
```

The mel pipeline inside `extract_speaker_embedding.py` matches the official
Qwen3-TTS `modeling_qwen3_tts.mel_spectrogram` exactly (magnitude not power,
librosa slaney-norm mel, reflect-pad `(n_fft-hop)/2`, `log(clip(x, 1e-5, None))`).
Skip any of those four and the Talker collapses to filler tokens (see
`docs/performance/qwen3-orin-nx-voice-clone-pass-2026-05-11.md`).

## Current Qwen3 baseline (2026-05-11)

End-to-end loopback evidence in
`docs/performance/qwen3-orin-nx-loopback-pass-2026-05-11.md` and
`docs/performance/qwen3-orin-nx-voice-clone-pass-2026-05-11.md` — both report
exact-match ASR on three Chinese prompts via the slim docker container on Orin
NX. Performance numbers in
`docs/performance/qwen3-orin-profiles-2026-05-10.md` (V2V `EOS → first audio`
611–637 ms).

## Benchmarking

The `bench/` directory has two kinds of scripts:

- **Direct-backend** (`bench_kokoro`, `bench_matcha`, `bench_qwen3_tts`, `bench_qwen3_asr`,
  `bench_paraformer`) — load voxedge in-process, no HTTP service needed. Measure total
  synthesis time and RTF; TTFA cannot be measured here (IPC worker batches all audio).
- **HTTP streaming** (`bench_http_tts_stream`) — hits a running seeed-voice service via
  `/tts/stream`, measures true streaming TTFA (first PCM chunk) and RTF via chunked HTTP.

### Prerequisites

```bash
# Direct-backend: set PYTHONPATH to the voxedge checkout
export PYTHONPATH=~/voxedge-test

# HTTP streaming: start the service first (port 8621 by default)
docker compose -f deploy/docker-compose.yml up -d
```

### `bench/bench_http_tts_stream.py` — HTTP streaming TTFA

```bash
python3 bench/bench_http_tts_stream.py \
    --url http://localhost:8621 \
    --repeat 3
```

Measures true TTFA via `/tts/stream` (chunked PCM). Requires a running service.
`--api-key` if the service has authentication enabled. `--output-jsonl` to save results.

### `bench/bench_kokoro.py` — Kokoro TRT TTS

```bash
python3 bench/bench_kokoro.py \
    --model-base ~/ovs-models/kokoro-multi-lang-v1_0 \
    --hybrid-dir ~/ovs-models/kokoro-multi-lang-v1_0/hybrid2 \
    --runtime-mode hybrid \
    --repeat 3
```

Key args: `--runtime-mode` (hybrid/split_generator/engine/ort_cpu),
`--speaker-id`, `--text`, `--output-jsonl`.

### `bench/bench_qwen3_tts.py` — Qwen3-TTS worker TTS

```bash
python3 bench/bench_qwen3_tts.py \
    --worker-build ~/path/to/jetson-workers \
    --plugin-path ~/path/to/libNvInfer_edgellm_plugin.so \
    --talker-dir ~/path/to/engines/talker \
    --cp-dir ~/path/to/engines/code_predictor \
    --tokenizer-dir ~/path/to/engines/talker \
    --code2wav-dir ~/path/to/engines/code2wav \
    --repeat 3
```

Key args: `--qwen3-profile` (highperf/official), `--text`, `--output-jsonl`.
Worker binary must be compiled against the same runtime version as the engines
(check the `[version.cpp] Model version X does not match runtime version Y` warning).

### `bench/bench_matcha.py` — Matcha TRT TTS

```bash
python3 bench/bench_matcha.py \
    --model-base ~/voice_test/models/matcha-icefall-zh-en \
    --split-encoder-onnx ~/matcha_split/onnx/matcha_encoder_trt.onnx \
    --split-estimator-engine ~/matcha_split/engines/matcha_estimator_step0_bf16.engine \
    --acoustic-ep SPLIT_TRT \
    --skip-if-no-vocos \
    --repeat 3
```

Key args: `--acoustic-ep` (SPLIT_TRT or empty for full-ORT),
`--vocos-engine` (TRT engine path), `--skip-if-no-vocos`, `--text`, `--output-jsonl`.

### `bench/bench_qwen3_asr.py` — Qwen3-ASR worker

```bash
python3 bench/bench_qwen3_asr.py \
    --worker-build ~/path/to/jetson-workers \
    --plugin-path ~/path/to/libNvInfer_edgellm_plugin.so \
    --engine-dir ~/path/to/engines/asr_thinker_full_fp8embed \
    --audio-enc-dir ~/path/to/engines \
    --audio test.wav \
    --repeat 3
```

`--audio-enc-dir` must be the **parent** of the `audio/` subdirectory (i.e., the
directory containing `audio/config.json` and `audio/audio_encoder.engine`).
If `--audio` is omitted, a synthetic 2 s 440 Hz tone at 16 kHz is used.

### `bench/bench_paraformer.py` — Paraformer TRT ASR

```bash
python3 bench/bench_paraformer.py \
    --model-dir ~/path/to/paraformer_build \
    --decoder-engine ~/path/to/paraformer_decoder.plan \
    --audio test.wav \
    --repeat 3
```

Requires a decoder TRT engine (`--decoder-engine`). Encoder falls back to ORT
CUDA EP if TRT encoder produces NaN (auto-detected). `--encoder-engine` and
`--encoder-onnx` are both optional.

## Contributing

This repo is the Jetson engine build component of [OpenVoiceStream](https://github.com/suharvest/openvoicestream). Bug reports and pull requests are welcome.

- **Engine bugs / build failures** — open an issue here
- **Product / Docker / deployment issues** — open an issue in [openvoicestream](https://github.com/suharvest/openvoicestream)
- **Agent operating guide** — see `AGENTS.md` for coding agent conventions in this repo
- **Patch / overlay changes** — read `engine-overlay/DIVERGENCE.md` before modifying; some patches are PR-pending upstream and must not be force-submitted

## Acknowledgements

- [NVIDIA TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) — the upstream engine framework this repo forks and extends
- [Qwen3-TTS / Qwen3-ASR](https://github.com/QwenLM/Qwen3-TTS) — Alibaba Qwen3 speech models
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — non-autoregressive TTS acoustic model
- [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) — lightweight neural TTS
- [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS) — compact TTS with paged-KV support

## License

[Apache-2.0](LICENSE)
