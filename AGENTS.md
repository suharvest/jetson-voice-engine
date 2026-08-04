# Agent Guide

This repository is the Jetson engine **build component** consumed by
OpenVoiceStream (OVS). It is not the OVS service and it is not VoxEdge.

## Release boundary

- NVIDIA upstream is `TensorRT-Edge-LLM` v0.9.1 at
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6`.
- `engine-overlay/build.sh` applies exactly 7 locked proposed-upstream bug
  patches and then exactly 35 sparse local product patches. Keep those two
  series separate; never fold, renumber, or silently skip them.
- This repository owns export/build drivers, the overlay, native voice workers,
  build manifests, checksums, and artifact-production documentation.
- OVS owns profiles, composition, HTTP/API behavior, image assembly, runtime
  deployment, and `deploy/artifacts/v091-release-lock.json`. That outer lock is
  the only release source of truth. Its schema binds `schema_version`,
  `artifact_set`, `target`, `source`, `model_artifacts`, and `artifacts`.
- Matcha is built here, but an OVS profile composes it with ASR and the service.
  Do not claim that this repository deploys or selects Matcha.

The active build target is fail-closed: Orin NX / SM87, JetPack 6.2 (L4T
R36.4.3), CUDA 12.6, TensorRT 10.3, aarch64, `jetson-orin`, Release. The build
probes the host and `verify-release-target.py` rejects mismatches before CMake.
Changing any field requires a new qualified artifact set, not an edit to the
existing release identity.

## Reproducible inputs

Use `HF_ENDPOINT=https://hf-mirror.com` and verify it in a non-login child
shell before downloading. Use `hf download --revision <immutable-sha>`; never
use a floating branch for a release build. The Spark inputs are currently
locked in `engine-overlay/drivers/fetch-sparktts-v091-inputs.sh`. Other model
revisions must be supplied by, and match, the outer OVS release lock. Do not
invent a revision when historical provenance does not prove it.

Generated artifacts are model-owned in HF. See `HF_ARTIFACTS.md` for the repo
map. The JVE aggregate manifest is legacy build/staging provenance, not a
release lock. `published_to_hf=false` means it must not be presented as a
downloadable release.

## Build

Materialize the exact source tree without CUDA:

```bash
cd engine-overlay
./build.sh --apply-only
```

Build on the qualified Jetson host:

```bash
cd engine-overlay
./build.sh manifests/qwen3-asr-sm87.toml
./build.sh manifests/qwen3-tts-highperf-sm87.toml
./build.sh manifests/customvoice-v091.toml
./build.sh manifests/sparktts-sm87-v091.toml
```

Required workers, plugins, engines, provenance files, and checksums must fail
loud. A release path must not convert missing outputs into a warning or an
exit-zero best-effort result.

## Validation

Before committing:

```bash
bash -n engine-overlay/*.sh engine-overlay/drivers/*.sh scripts/*.sh
python3 -m py_compile engine-overlay/*.py scripts/*.py
python3 -m pytest -q tests/test_v091_overlay_contract.py \
  tests/test_v091_release_download_contract.py \
  tests/test_sparktts_export_compat.py
bash engine-overlay/tests/verify-patch-stack.sh
SKIP_AUTOCLONE=1 bash engine-overlay/tests/test-provenance-negative.sh
```

For model/toolchain builds, retain the manifest, `PROVENANCE.md`,
`DRIVER_REVISION`, and SHA-256 sidecars. A device runtime claim additionally
requires OVS release-lock validation and the OVS hardware qualification gate.
