# v0.10.0 branch validation — 2026-08-14

This is source/build qualification evidence, not a release qualification. No
v0.9.1 ONNX, TensorRT engine, plugin, or worker is credited to v0.10.

## Source gates

- Official base: `71dd1bae032e70771265917ec74d3ff4cad07a10`.
- Exact source identity: 4 locked proposed-upstream patches, additive `addon/`,
  then 32 locked sparse product patches.
- `build.sh --apply-only`: PASS from a clean official checkout.
- `tests/verify-patch-stack.sh`: 4+32 forward replay PASS; reverse replay
  returns to official tree `6ade7849c470eb2becf669b9470e7385f1454e3d`.
- Four manifests pass `validate-manifest.py`; provenance negative tests PASS.
- `tests/test_v010_overlay_contract.py`: 6 PASS.

## CuTe FMHA-v2 gate

The official source has no SM87/CUDA12 prebuilt; its SM87 archive is CUDA13.
On `orin-nano`, generated all 25 supported `fmha` variants with:

- host CUDA 12.6.68;
- `nvidia-cutlass-dsl==4.6.1` (cu12 runtime);
- `cupy-cuda12x==12.3.0`;
- target `aarch64/sm_87`, artifact CUDA major 12.

The resulting tarball SHA-256 and metadata are locked in
`CUTEDSL-SM87-CUDA12.lock`.

## Device compile gates

### orin-nano — official baseline

JetPack 6.2.1 / CUDA 12.6 / TensorRT 10.3 / SM87:

- CMake with `ENABLE_CUTE_DSL=fmha`: PASS;
- `NvInfer_edgellm_plugin`: PASS;
- `llm_inference`: PASS;
- `ldd -r` and `--help` smoke: PASS.

### orin-nx — complete patched tree

JetPack 6.2 / CUDA 12.6 / TensorRT 10.3 / SM87:

- CMake with `ENABLE_CUTE_DSL=fmha`: PASS;
- `NvInfer_edgellm_plugin`: PASS;
- `llm_inference`: PASS;
- `qwen3_tts_streaming_worker`: PASS;
- `moss_tts_nano_worker`: PASS;
- `ldd -r` on plugin and all three executables: PASS;
- `--help` smoke on all three executables: PASS.

The first patched archive compile lacked initialized `3rdParty/` content
because it was packaged from an `--apply-only` tree. Restoring the official
submodule payload fixed the packaging-only failure; normal full `build.sh`
initializes submodules before CMake.

## Gates still required before an overall OVS upgrade

- Port model-specific v0.10 export/build drivers.
- Regenerate every ONNX and TensorRT engine with v0.10 identity.
- Publish model-owned immutable artifacts and a new outer v0.10 release lock.
- Run ASR, Qwen3-TTS Base/CustomVoice/native clone, MOSS, Spark, Qwen3.5,
  cancellation/recovery, N=1/N=2, byte/parity, co-residency, and RSS gates.
- Build new v0.10 runtime images/profiles/compose identities while preserving
  v0.9.1 as rollback.

Until those gates pass, `build-engines-for-device.sh` remains deliberately
fail-closed and outer OVS v0.9.1 release identities must not be changed.
