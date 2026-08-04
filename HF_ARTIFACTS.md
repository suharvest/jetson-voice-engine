# Hugging Face artifact contract

This repository builds Jetson artifacts. OpenVoiceStream resolves and deploys
them through its profiles and its fail-closed
`deploy/artifacts/v091-release-lock.json`.

## Repository map

| Model family | Generated-artifact repository |
|---|---|
| Qwen3 ASR / Qwen3 TTS | `harvestsu/qwen3-edgellm-jetson-artifacts` |
| SparkTTS | `harvestsu/sparktts-0p5b-jetson-artifacts` |
| MOSS-TTS-Nano | `harvestsu/seeed-local-voice-artifacts` |
| Qwen3.5 GDN/MTP | `harvestsu/Qwen3.5-4B-AWQ-GDN-MTP-TensorRT-EdgeLLM-engine` |
| SenseVoice | `harvestsu/sensevoice-rknn` |

The table names generated artifact stores, not official weight sources. The
official model repo IDs and the proven immutable revisions are recorded in the
r5 entry of `deploy/artifacts/qwen3_manifest.json`. Spark is pinned to model
snapshot `642071559bfc6346c2359d19dcb6be3f9dd8a05d` and source commit
`2f1ea9082400547242641f5271b6f941c9f439d1`. For the remaining models, the
historical files in this repository do not prove one immutable snapshot; the
outer OVS release lock must supply it. A floating `main` is not a release pin.

## v0.9.1 assembly set

The legacy aggregate staging candidate is
`orin-nx-edgellm-v091-jp62-trt103-sm87-20260803-r5`, under `v091/` in the
Qwen3 artifact repository. It is bound to:

- NVIDIA TensorRT-Edge-LLM v0.9.1 at `7f061f21f0a581ba234a1e233c9315b89d8e47d6`;
- 7 locked proposed-upstream bug patches plus 35 local product patches;
- Jetson Orin NX, SM87, JetPack 6.2 / L4T R36.4.3, CUDA 12.6,
  TensorRT 10.3, aarch64 Release;
- the required file list, manifest, provenance, and SHA-256 inventory in
  `deploy/artifacts/qwen3_manifest.json`.

It remains `published_to_hf=false`. Do not describe it as released or use the
repository-level `revision: main` as a production lock. This aggregate manifest
is retained only for build/staging provenance.

The only release source of truth is OVS
`deploy/artifacts/v091-release-lock.json`. Its schema records:

- `schema_version` and `artifact_set`;
- `target.platform`, `target.sm`, `target.jetpack`, `target.cuda`,
  `target.tensorrt`, and `target.onnxruntime`;
- `source.upstream_sha`, `source.build_outer_sha`,
  `source.engine_overlay_sha`, and `source.formal_diff_sha256`;
- each `model_artifacts.<model>.repo`, immutable `revision`,
  `payload_sha256`, and `payload_size`;
- each `artifacts.<path>` digest, size, mode, and any other required file
  metadata.

## Publication gate

All required files must exist and verify. Missing worker binaries, plugin,
engines, metadata, provenance, or checksum files are fatal; optional/best-effort
semantics are forbidden for anything listed in `required_files`.

Use the mirror and preserve resumable HF cache state:

```bash
test "$(bash -c 'printf %s "$HF_ENDPOINT"')" = https://hf-mirror.com

python3 scripts/package_qwen3_artifacts.py \
  --set orin-nx-edgellm-v091-jp62-trt103-sm87-20260803-r5 \
  --source-root /opt/edgellm-v091 \
  --out /tmp/edgellm-v091-r5-stage
```

Upload requires explicit release approval. After upload, record the immutable
HF commit in the OVS release lock, download that revision into a clean path,
and verify every byte against `SHA256SUMS` before changing
`published_to_hf` to true.
