# TensorRT-Edge-LLM v0.10.0 overlay scaffold

This directory is an isolated migration scaffold for NVIDIA TensorRT-Edge-LLM
v0.10.0. The v0.9.1 `engine-overlay/` remains the rollback baseline and is not
modified by this scaffold.

```text
engine-overlay-v010/
  UPSTREAM_PIN                 # 71dd1bae… (NVIDIA tag v0.10.0)
  upstream.remote              # NVIDIA canonical repository
  patches/upstream-v010-prs/   # four locked, rebased upstream fixes
  patches/PATCH-STATE-v010.md  # retirement and product-rebase ledger
  addon/                       # copied product files, unchanged for now
  patches/v091-candidate/      # legacy product patches, history only
  patches/v010-candidate/      # 32 sparse rebased product patches
  build.sh                     # reproducible materialization/build wrapper
  manifests/                   # v0.10 build contracts
```

## Proposed-upstream migration

The active upstream series has four entries:

1. explicit CUDA architecture preservation (PR #118, rebased);
2. the v0.10-only static-target `INTERFACE` `--wrap=_cudaLaunchKernelEx`
   propagation (PR #118; the shim link propagation is already upstream);
3. linear MRoPE metadata normalization (PR #146, rebased);
4. destination checkpoint dtype preservation (PR #149, rebased).

TensorRT-Edge-LLM v0.10.0 already contains the FP4 guard and pre-10.7 reader
fallback; it removes the legacy embedded-cubin FMHA backend in favor of CuTe
DSL FMHA-v2. The old v0.9.1 patch files are not part of this series. See
[PATCH-STATE-v010.md](patches/PATCH-STATE-v010.md)
for the disposition ledger.

See [RELEASE-NOTES-ADOPTION.md](RELEASE-NOTES-ADOPTION.md) for the product
adoption decision on new v0.10 capabilities.

Each upstream patch is byte-locked by `series`, `LOCK`, and `SHA256SUMS`.
The lock records the local rebased commit/tree provenance; replay always starts
from the official `UPSTREAM_PIN` and applies the patch diffs in series order.

## Full replay

No CUDA or TensorRT is required to materialize the upstream chain:

```bash
EDGELLM_UPSTREAM_REMOTE=/tmp/tensorrt-edge-llm-v010-upstream \
VOXEDGE_WORKDIR=/tmp/tensorrt-edge-llm-v0100-replay \
  ./build.sh --apply-only
```

`--apply-only` applies the four upstream fixes, copies `addon/`, and replays all
32 product patches without requiring CUDA or TensorRT.

The focused integrity/replay gate is:

```bash
tests/verify-patch-stack.sh /tmp/tensorrt-edge-llm-v010-upstream
```

The gate verifies the official v0.10.0 baseline tree, the complete 4+32 forward
patch chain, and reverse replay. Device qualification remains separate.

## Build target

The target remains Jetson Orin NX / SM87, JetPack 6.2 (L4T 36.4.3), CUDA 12.6,
TensorRT 10.3, aarch64, Release. Existing v0.9.1 images and artifacts remain the
rollback path until the v0.10 hardware gates pass.

v0.10.0 always compiles its new CuTe FMHA-v2 runner. NVIDIA ships an SM87
prebuilt only for CUDA 13, so the JetPack 6.2 / CUDA 12.6 build must first
generate the `fmha` group on an Orin device with `nvidia-cutlass-dsl==4.6.1`
and `cupy-cuda12x==12.3.0`. The default build therefore uses
`ENABLE_CUTE_DSL=fmha`; `OFF` is not a valid pristine v0.10 plugin build.
The first qualified artifact metadata and checksum are recorded in
`CUTEDSL-SM87-CUDA12.lock`; a final release must publish and bind that artifact
in the outer OVS v0.10 release lock.

The aggregate `build-engines-for-device.sh` is now a v0.10-only driver. It
verifies the official v0.10.0 base pin, requires immutable model revisions,
uses the native Base/VoiceDesign/CustomVoice exporter, builds both native Base
clone encoders, and uses the upstream MTP `--specBase` / `--specDraft` path.
Qwen3-TTS INT4 remains an explicit product driver because upstream v0.10
supports the Talker in FP16 only: `qwen3-tts` is the native FP16 route and
`qwen3-tts-int4` is the separately qualified CustomVoice extension. The
production-equivalent Base lane is `qwen3-tts-base-int4`; it reuses the
immutable, previously qualified stage-2 checkpoint but always re-exports the
Talker with the v0.10 exporter and plugin v1. `qwen3-tts-base` remains the
native FP16 correctness/gray baseline. The reviewed driver is
`drivers/export-qwen3-tts-int4-v010.sh` and refuses floating stage-2 inputs.
Native VoiceDesign and Qwen3.5 DFlash routes are revision-locked for gray
testing. Qwen3.5 Orin release routes preserve W4A16-AWQ and plugin v1 rather
than using the upstream NVFP4 default.
The aggregate driver always writes to a new v0.10 artifact root; no v0.9.1
ONNX or engine is accepted as an input.

The port makes model regeneration possible but does not by itself switch OVS.
Every v0.10 model artifact must still pass hardware qualification and be bound
by a new outer release lock before profiles, images, or compose identities move.
