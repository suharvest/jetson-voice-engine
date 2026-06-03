# jetson-voice-engine

The **build-time engine repo** for the Jetson voice stack. It produces *all*
TensorRT engines and runtime artifacts that the `seeed-local-voice` product
service deploys on Jetson Orin — across every supported model — and publishes
the large outputs to Hugging Face.

> **Repo name note:** the GitHub remote is still
> `suharvest/qwen3-edgellm-jetson` (rename to `jetson-voice-engine` pending).
> The directory is consumed as the `third_party/qwen3-edgellm-jetson`
> submodule of `seeed-local-voice`.

Post-restructure, this repo was merged from two former sources:

1. the **voxedge-engine overlay** of the NVIDIA TensorRT-Edge-LLM fork
   (`engine-overlay/`), and
2. the **seeed Jetson engine-build scripts** for every model
   (`models/<model>/`).

The deployable product service (API, frontend/backend selection, Docker,
device deployment) lives in `seeed-local-voice`; this repo owns export, engine
build, runtime overlay, validation gates, and lessons learned.

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
| `tests/` | `golden_mels/` fixtures for mel parity. |
| `HF_ARTIFACTS.md` | What belongs in the HF artifact repo + stage/upload commands. |
| `AGENTS.md` | Concise operating guide for coding agents in this repo. |

For the end-to-end picture across product + engine + artifacts, see the
seeed `docs/REPRODUCE.md` in the parent `seeed-local-voice` repo.

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
> including the **SSE/client-disconnect fix (in patch `0006`) which is
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

## Quick start — Qwen3 one-shot on Orin NX

```bash
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
bash qwen3-edgellm-jetson/scripts/reproduce_qwen3_highperf.sh
# add: --reference path/to/24kHz_mono.wav   to also verify voice clone
```

`scripts/reproduce_qwen3_highperf.sh` is idempotent and handles the whole
Qwen3 chain: clones the repos at the validated branches, builds EdgeLLM,
downloads + SHA-256-verifies the HF artifact set, builds the slim docker
image, brings the service up, then runs `scripts/verify_reproduction.sh`,
which gates on (1) W8A16 plugin symbol set, (2) artifact integrity, (3) HTTP
TTS→ASR loopback on three Chinese prompts (LCS-similarity ≥ 0.7 across up to 3
retries), and (4) voice clone via a real reference WAV. Exit 0 means the whole
chain is verified; on failure it prints which check failed and why.

Prerequisites: Jetson Orin NX with JetPack 6 (TensorRT 10.3.0.30, CUDA 12.6),
docker + `--runtime nvidia`, ~10 GB free disk for HF artifacts.

`scripts/verify_reproduction.sh` is reusable gate-only (symbol set / artifact
SHA-256 / TTS+ASR loopback / clone) for CI. See
`docs/reproduce-from-zero.md` for the manual step-by-step fallback.

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
