# Reproduce the Jetson engine build from zero (v0.9.1)

This guide reproduces **build outputs**. OpenVoiceStream owns profile
composition, images, service startup, and the final release lock. In
particular, Matcha engines can be built here, but only an OVS profile combines
Matcha with ASR and exposes it through the product service.

## Locked identity

| Input | Required value |
|---|---|
| NVIDIA upstream | `https://github.com/NVIDIA/TensorRT-Edge-LLM.git` |
| Upstream commit | `7f061f21f0a581ba234a1e233c9315b89d8e47d6` (v0.9.1) |
| Proposed-upstream fixes | 7 files in `engine-overlay/patches/upstream-v091-prs/series` |
| Local product patches | 35 files in `engine-overlay/patches/v091-candidate/series` |
| Target | Orin NX, SM87, JP6.2/L4T 36.4.3, CUDA 12.6, TRT 10.3, aarch64 Release |

The four TOML manifests repeat this identity and
`engine-overlay/validate-manifest.py` rejects any mismatch before cloning or
building.

## 1. Prepare the mirror-backed environment

Large downloads must use the HF mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
test "$(bash -c 'printf %s "$HF_ENDPOINT"')" = https://hf-mirror.com
command -v hf
```

Use `hf download --revision <immutable-sha>` and keep the HF cache so an
interrupted `.incomplete` download can resume. Do not construct a
`huggingface.co` URL with curl/wget.

## 2. Clone this build repository

```bash
git clone https://github.com/suharvest/jetson-voice-engine.git
cd jetson-voice-engine
git checkout <commit-recorded-by-OVS-release-lock>
```

For an OVS release, the only source of truth is OVS
`deploy/artifacts/v091-release-lock.json`. It binds `schema_version`,
`artifact_set`, the exact target tuple (including ONNX Runtime), source SHAs and
formal-diff hash, per-model HF repo/revision/payload digest+size, and per-file
artifact digest/size/mode. This repository's aggregate manifest is legacy
build/staging provenance and must not replace that outer lock.

## 3. Verify and materialize the overlay

This step works without CUDA and proves the 7+35 boundary:

```bash
cd engine-overlay
bash tests/verify-patch-stack.sh
SKIP_AUTOCLONE=1 bash tests/test-provenance-negative.sh
./build.sh --apply-only
```

The resulting checkout is NVIDIA v0.9.1 plus exactly the locked upstream bug
series, addon files, and sparse local product series.

## 4. Materialize model inputs

The model-to-artifact repository map is in `HF_ARTIFACTS.md`. Every official
model download needs an immutable revision. Spark is the only current input
whose historical revision is fully proven in this repository:

```bash
bash engine-overlay/drivers/fetch-sparktts-v091-inputs.sh /path/to/input-root
```

That driver pins Spark-TTS source
`2f1ea9082400547242641f5271b6f941c9f439d1` and model snapshot
`642071559bfc6346c2359d19dcb6be3f9dd8a05d`.

Qwen3 ASR, Qwen3 TTS Base/CustomVoice, MOSS, and Qwen3.5 GDN/MTP builds must
receive the immutable revisions recorded by the OVS release lock. Existing
repository history does not justify substituting a guessed SHA or floating
`main`. Base TTS additionally requires `TTS_BASE_MODEL_REVISION`; its INT4 path
also requires `TTS_BASE_STAGE2_CHECKPOINT` and `TTS_BASE_STAGE2_REVISION`.

## 5. Build on the qualified Jetson

On Orin NX with JP6.2/L4T 36.4.3, CUDA 12.6 and TRT 10.3:

```bash
cd engine-overlay
./build.sh manifests/qwen3-asr-sm87.toml
./build.sh manifests/qwen3-tts-highperf-sm87.toml
./build.sh manifests/customvoice-v091.toml
./build.sh manifests/sparktts-sm87-v091.toml
```

`build.sh` probes SM/platform, L4T, CUDA, and TensorRT and rejects any target
other than the qualified tuple before CMake. It also fails if required
voice-worker sources, the MOSS worker, the plugin, or other mandatory outputs
are absent. A warning plus exit zero is not an acceptable release result.

Per-family engine drivers are under `models/` and
`engine-overlay/build-engines-for-device.sh`. Matcha scripts under
`models/matcha/` only produce engine assets; OVS owns the Matcha profile and
service integration.

## 6. Stage and verify artifacts

Stage the exact r5 required-file set:

```bash
cd ..
python3 scripts/package_qwen3_artifacts.py \
  --set orin-nx-edgellm-v091-jp62-trt103-sm87-20260803-r5 \
  --source-root /opt/edgellm-v091 \
  --out /tmp/edgellm-v091-r5-stage
```

The command must fail on every missing `required_files` entry. Verify the
staged `manifest.json`, `PROVENANCE.md`, and `SHA256SUMS`; then publish only
with explicit approval and lock the returned immutable HF revision in OVS.
The current manifest says `published_to_hf=false`, so a from-zero **deployment**
is intentionally blocked until that outer lock and remote snapshot exist.

## 7. Hand off to OVS

OVS must fail closed on all of the following before starting the service:

- `source.upstream_sha`, `source.build_outer_sha`,
  `source.engine_overlay_sha`, and `source.formal_diff_sha256`;
- 7 proposed-upstream + 35 local patch identities;
- every `model_artifacts` repo, immutable revision, payload SHA-256 and size;
- `target.platform`, SM87, JetPack 6.2, CUDA 12.6, TensorRT 10.3, and the
  locked ONNX Runtime version;
- container digest and selected OVS profile;
- every locked artifact path's SHA-256, size, mode, and required metadata.

Run runtime, HTTP, concurrency, cancellation, and Matcha-composition gates from
OVS. Passing this repository's build tests alone is not a production release.
