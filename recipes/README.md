# Reproducible Recipes — our own models & engine artifacts

This directory is the **single discoverable index** for every artifact *we*
produce on top of NVIDIA TensorRT-Edge-LLM (TRT-Edge-LLM) for the Jetson voice
stack: quantized exports (int4 / fp8), per-device TRT engines, and the
slot-pool worker binaries. Each recipe is a self-contained "how to regenerate
this artifact from public weights" note.

Sources for these recipes were previously scattered across the fork
(`TensorRT-Edge-LLM` wip branches), the `edgellm-v080-migration` repo, Mac
backups, and the `engine-overlay/`. They are consolidated here so the
export/build steps are reproducible **and findable**.

## How to read a recipe

Every recipe states:

- **Artifact** — what comes out (engine / ONNX / quantized checkpoint).
- **Inputs** — public weights + which driver scripts (with a **pinned fork
  commit / branch**).
- **Commands** — the exact invocation sequence (driver flags + `llm_build`).
- **Pin** — fork SHA/branch the drivers live on.
- **md5** — recorded digest of the known-good artifact.
- **HF** — where the prebuilt artifact is published.
- **Build device** — where it was produced (all engines: **SM 87 Jetson Orin**,
  NX or Nano).

> **TRT engines are NOT byte-reproducible.** TensorRT autotuning is
> host/driver/version sensitive, so a rebuilt `*.engine` will almost never
> match the recorded md5. **The md5 is a download-integrity check for the
> published artifact only.** The reproducibility judgement for engines is
> **functional equivalence** (sane ASR transcription / TTS audio roundtrip),
> not a byte match. Quantized **checkpoints / ONNX** are more stable but still
> not guaranteed bit-identical across toolkit versions.

## Index

| Recipe | Artifact | Driver pin | Build device |
|---|---|---|---|
| [int4-tts-export.md](int4-tts-export.md) | int4-AWQ Talker + CodePredictor ONNX; fp8 `text_embedding` | fork `wip/native-int4-talker` `ff2318e`, `wip/fp8-embedding` `873ca22` | host w/ CUDA GPU (export); engine on SM 87 Jetson |
| [asr-int4-export.md](asr-int4-export.md) | int4-AWQ ASR decoder checkpoint/ONNX | fork `wip/asr-int4-decoder` `c80bcc0` | host w/ CUDA GPU |
| [asr-b2-engine.md](asr-b2-engine.md) | Qwen3-ASR thinker engine, `maxBatchSize=2` (N>1 ASR) | (engine build, no fork driver) | orin-nx (SM 87) |
| [talker-b2-engine.md](talker-b2-engine.md) | Qwen3-TTS Talker engine, `maxBatchSize=2` (N>1 TTS **batch-lane** — DEFERRED alt, `v080-0010`; NOT the production slot-pool path) | (engine build, no fork driver) | orin-nx (SM 87) |
| [slot-pool-worker.md](slot-pool-worker.md) | `qwen3_tts_streaming_worker` + `qwen3_asr_worker` (N>1) binaries — **production N>1 path** (slot-pool + shared-engine ctor) | overlay `UPSTREAM_PIN` `a361221` | SM 87 Jetson |

See also: [`../docs/asr-thinker-engine-build-recipe.md`](../docs/asr-thinker-engine-build-recipe.md)
(canonical maxBatch=1 ASR thinker + `--fp8_embedding`), and
[`../HF_ARTIFACTS.md`](../HF_ARTIFACTS.md).

## Published HF artifact locations

| HF repo | Contents |
|---|---|
| `harvestsu/seeed-local-voice-artifacts` | `sm87-trt10.3-jp6.2/v0.8.0/` engine sets + `asr-b2` (N>1 ASR thinker engine) + `talker-b2` |
| `harvestsu/qwen3-tts-0.6b-base` | Qwen3-TTS-12Hz-0.6B-Base re-export bundle |
| `harvestsu/customvoice-...int4fp8` | CustomVoice int4 Talker + fp8 embedding bundle |
| `harvestsu/qwen3-edgellm-jetson-artifacts` | per-device highperf engine sets (ASR/TTS), manifests/checksums (see `HF_ARTIFACTS.md`) |

`sm87` = Jetson Orin Ampere compute capability 8.7 (NX **and** Nano).
`trt10.3` = TensorRT 10.3. `jp6.2` = JetPack 6.2 / CUDA 12.6.
