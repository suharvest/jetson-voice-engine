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
- `tests/test_v010_overlay_contract.py`: 7 PASS.

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

### orin-nano — complete patched tree

JetPack 6.2.1 / CUDA 12.6 / TensorRT 10.3 / SM87:

- CMake with `ENABLE_CUTE_DSL=fmha`: PASS;
- `NvInfer_edgellm_plugin`: PASS;
- `llm_build` and `audio_build`: PASS;
- `qwen3_tts_streaming_worker` and `qwen3_tts_inference`: PASS.
- `ldd -r` on both TTS executables reports no missing libraries or undefined
  symbols; both `--help` smokes expose the native clone-engine contract.

Nano build artifact SHA-256 values:

- plugin: `2c4afd46718fd2e15afa85d542d4cbc1590bb3e677cabccc03c4afce5a5da330`;
- `llm_build`: `71f691eaff604b4e970faa981459dea2dec926597eb263ee4c71b854cec21c6f`;
- `audio_build`: `01409dcbdecaba063506c2474072df2516613ce5ec9303859166a4fec8c528e3`;
- streaming worker: `0fe6d16b22e15104108de36feccafca7f8ec6a97c61ed982a3cf24d6f94092f4`;
- inference CLI after the streaming PCM dtype fix:
  `735369ddfbb07117554f6e628a7659fa91d0ca9d4d4039370424055f012aa316`.

The first validation archive excluded `cpp/builder/` because its transport
filter used the over-broad name pattern `build*`. Restoring that tracked
directory and reconfiguring CMake fixed the validation-package failure; it was
not a source or CMake defect.

## Engine-driver port (2026-08-15)

- The aggregate engine driver is enabled for the exact v0.10.0 source pin.
- Qwen3-ASR, Qwen3-TTS, Qwen3.5 base, and native MTP source checkpoints are
  pinned to immutable Hugging Face revisions.
- Qwen3-TTS Base uses the native exporter and builds both
  `speaker_encoder.engine` and `speech_tokenizer_encoder.engine`.
- The legacy v0.9.1 Base exporter and standalone speaker-encoder surgery are
  retired. Code2Wav uses the v0.10 engine-root layout.
- The opaque MTP hook is retired in favor of native `--mtp`, `--specBase`, and
  `--specDraft` contracts.

## Qwen3-TTS Base gray export (2026-08-15)

- Model payload came from the immutable ModelScope revision
  `fda1995a3162ad0488393cc366b82ca59c91c08e`; the two large weight files match
  the official Hugging Face blob SHA-256 values exactly:
  `180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6`
  and `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`.
- The exporter was installed editable from the complete patched v0.10 replay
  tree, not from pristine upstream.
- Talker, code predictor, Code2Wav, speaker encoder, and speech-tokenizer
  encoder all export successfully and pass `onnx.checker`.
- Gray export found two upstream host-portability defects. Patch 0025 now
  selects CUDA only when available for Code2Wav export and forces eager Mimi
  attention during static clone-tokenizer export. This preserves Jetson CUDA
  behavior and fixes CPU/Apple export plus the Transformers symbolic SDPA
  guard failure.
- Frozen ONNX transport tar SHA-256:
  `12f70daf92287296ad8357a87ceb9a20c838b74a49c2ccb5d2795d581ff31622`.

## Qwen3-TTS Base gray runtime on orin-nano (2026-08-15)

Nano is an 8 GB resource-constrained gray lane, not the formal release
profile. Talker and code predictor use the release dimensions; Code2Wav is
limited to `min=1,opt=64,max=128` frames, and the speaker encoder accepts
1--10 seconds (`min=24000,opt=67200,max=240000` samples). All five engines
were built natively on Nano:

- Talker: `6897b813448eee183d6b43c04be6f51c5b06b62b473c4824fbad2a448c29a862`;
- code predictor: `bd1ffd2a71a38d90e1f57f10b75414845af4382e432721e33aa1610deaa3732c`;
- Code2Wav: `7197ddabbd3f51b3ff5daafad2d824abd3ba801650cab3baf70779fc09ed7408`;
- speaker encoder: `964ae2b971415a13da630bedcc0b4d65d5e50ce11fcc007812190783f9b572d0`;
- speech-tokenizer encoder:
  `8c997858e89bec1d817f9623626963fd888bd52f36195412405aba418db146cc`.

A real 2.8-second, 24 kHz reference WAV
(`66fdc78ffa0b59b11846dc3e1db3e067ecfc0d462f43067d68afe4d0d8fe1273`)
and a Chinese Base clone request produced 64 frames / 5.12 seconds of 24 kHz,
16-bit mono PCM. Non-streaming inference passed 1/1 requests; its output WAV
SHA-256 is
`b2088a2cde5dd4475c9256f421a6b851fde9e146a8f503daac31a97a153ca3a7`.

The first streaming run exposed an upstream v0.10 CLI defect: Qwen3-TTS
Code2Wav emits FP32, but the chunk accumulator unconditionally reinterpreted
the buffer as FP16. RVQ remained bit-exact while the PCM saturated
(`RMS=26326`). Patch 0025 now normalizes FP16 or FP32 chunks to FP32 before WAV
encoding. The rebuilt CLI SHA-256 is
`735369ddfbb07117554f6e628a7659fa91d0ca9d4d4039370424055f012aa316`.
After the fix, streaming passed 1/1 requests with 9 callbacks, 64 frames,
TTFC 106.8 ms, TTFPA 548.4 ms, peak reported unified memory 1025.32 MB, and a
valid 5.12-second PCM WAV. Its RMS is 313.29 (non-streaming 343.47), peak is
2951, RVQ SHA-256 is identical to non-streaming
(`6d0145738e13bc68a10ec95b6641a386db89936c214f6915f3a2bc22c6867874`),
and WAV SHA-256 is
`7a9dbb0ff50c4d543c9f5fe6bbe6c7832a271ed51218f65f23624f614d22a314`.
Chunked and one-shot vocoder waveforms are not expected to be sample-identical
because chunk boundaries reset vocoder context; listening/quality parity is a
later release gate.

The production `qwen3_tts_streaming_worker` was also exercised with a real
Base clone request rather than credited from the CLI result. Its event sequence
was `ready`, 9 `chunk` events, then `done`; it emitted 122880 PCM-S16LE samples
(245760 bytes), RMS 1827.37, peak 13736, and PCM SHA-256
`1781a97361e5b33dee0275380efac84fe05941eb5505aec8b3e2bbfcf27dbcce`.
The worker already branches on the actual FP16/FP32 waveform dtype and did not
share the CLI bug.

## Qwen3-TTS Base formal-profile runtime on orin-nx (2026-08-15)

The 16 GB Orin NX built the full release-shaped Base set with Code2Wav
`min=1,opt=128,max=512` and speaker audio
`min=24000,opt=240000,max=960000` (1--40 seconds):

- Talker: `42c5adf8a84cf4d9b9749da61852af89cf62be119f6fb64186c98b066896ad4d`;
- code predictor: `26b63eb030590fe2cf0f1f4d09d9812350cf002983f1034362e679200d9bd063`;
- Code2Wav: `9979795d0e0e05e21b3090a1f674acbcb18ef72172e665f7d97d34a85a5f7424`;
- speaker encoder: `e91083f4719168cd4f0357482b504dc0a7d92678b49f5ee7ac2746a1d5ac89f7`;
- speech-tokenizer encoder:
  `be7be4c96916e90873175720fcc29e71a41c8a7d948baa67db57ac191f9cc25d`.

The same checked reference WAV and Chinese clone request passed non-streaming
and streaming with 64 frames / 122880 samples / 5.12 seconds of valid 24 kHz,
16-bit mono PCM. Non-streaming stage GPU time was 5.72 seconds and peak
reported unified memory was 1030.23 MB; its WAV SHA-256 is
`c0fd331dc17b71a5e55d2d3598c1ee78831788fe732f7c74947773a270276019`.
Streaming emitted 9 callbacks with TTFC 76.3 ms, TTFPA 494.9 ms, peak reported
unified memory 1032.60 MB, and WAV SHA-256
`b4fee5443ba4695a04f8fe856974b6fe0371bbbf037a0e1571ec42b946764add`.
Its RMS/peak (352.66/4215) are comparable to one-shot (386.78/4231), and the
two modes have the same RVQ SHA-256
`6258d8b62d8f98f7c3b0c0f678debaca7b3fcaa4951f89e8a709d74bce6ca7eb`.

## Gates still required before an overall OVS upgrade

- Regenerate every ONNX and TensorRT engine with v0.10 identity.
- Publish model-owned immutable artifacts and a new outer v0.10 release lock.
- Run ASR, Qwen3-TTS Base/CustomVoice/native clone, MOSS, Spark, Qwen3.5,
  cancellation/recovery, N=1/N=2, byte/parity, co-residency, and RSS gates.
- Build new v0.10 runtime images/profiles/compose identities while preserving
  v0.9.1 as rollback.

Until those gates pass, the driver may produce only isolated v0.10 candidate
artifacts and outer OVS v0.9.1 release identities must not be changed.
