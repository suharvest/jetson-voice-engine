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

## Qwen3-ASR FP16 gray runtime on orin-nano (2026-08-15)

This is a gray functional lane, not the formal INT4 release artifact. The
checkpoint is `Qwen/Qwen3-ASR-0.6B` at immutable revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. It was downloaded through
`https://hf-mirror.com`; `model.safetensors` SHA-256 is
`79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`.
The checked checkpoint tar SHA-256 is
`673d68f455b118f54955eeb29baea163ba44a2a8149c52d492040e77d17db747`.

The complete patched v0.10 exporter produced FP16 thinker and audio-encoder
ONNX on the host; both pass `onnx.checker`. The checked ONNX transport tar
SHA-256 is
`cafe29ab00c9b191b966aba7d452ef23582af3cd251b69eb855b155d36b7541b`.
The v0.10 builders then produced on Nano:

- thinker b1:
  `7894bdf9207edf6b55b169735cc9fe52e84b4cb8907d1fda7d17e87e3325abf9`;
- thinker b2 (`max_batch_size=2`):
  `8c5efe7647ce6abc3464ec86417095b46365aabb3a3e5952cb613cae503a6586`;
- audio encoder:
  `84c41bf89e394cf76f59db472db4d4bab4d1ad2266ee23bd9f68ed8e1fd961b1`.

The ASR worker was rebuilt from this branch's vendored source and linked to
the same v0.10 `libedgellmCore.a` and plugin. After the native-batch safety
guard described below, source SHA-256 is
`07aadf3a9b76835cfe2bb43886fb132687bc9e3f2f89c426f7aa0f7583b91bd7`
and binary SHA-256 is
`3381a9ce35d32c199ac37a65cecfe772b056678a48796d296d4f2520e752eab2`.

The b1 worker transcribed a checked 3.2-second real 24 kHz Chinese WAV
(`40609737c5c1d290f6bfe8e45ccebdb316a44c61606d286754b62b229bf753eb`)
through the production `pcm_b64` path as exactly
`一二三四五六七八九十。`. Three repeated finalizations took 558--584 ms
engine time. The b1 engine correctly clamps `--max_slots=2` to one lane; a
second begin returns `pool_saturated` / status 4429, and the lane is reusable
after release. A 33-sample `tegrastats` run observed peak RAM 4311/7620 MB and
peak GPU busy 72%.

The b2 engine exposes lanes 0 and 1, rejects a third begin with status 4429,
and reuses a released lane. Two simultaneously resident sessions finalized
serially with exact isolated transcripts `今天天气真好。` and
`人工智能改变了世界。` in 614 ms and 166 ms in the final guarded run. Peak
RAM was 4381/7620 MB and peak GPU busy 96%.

However, a single native multimodal request containing those two distinct WAVs
did **not** preserve isolation: before the guard, row 0 returned row 1's text
and row 1 hallucinated while the runtime reported `ok=true`. This is positive
evidence that v0.10 does not yet provide usable ASR continuous batching. The
worker now fails that unsafe request shape closed with
`native_audio_batch_unsafe_v010`; N=2 currently means two sessions may
co-reside while their full-audio finalizations execute serially. It is neither
mid-decode admission/refill nor true continuous batching. A two-distinct-WAV
native isolation gate must pass before this guard can be removed.

The v0.10 audio runner logs two non-fatal constraints in this configuration:
online GPU fbank needs `ENABLE_CUTE_DSL=gemm` (the release build currently
enables `fmha`, so CPU mel fallback is used), and the generic Omni runner probes
an optional `action.engine` that ASR does not use. Neither prevented the checked
ASR outputs above.

## WSL model-input reuse audit (2026-08-15)

`wsl2-local` has complete Hugging Face caches for all three immediate voice
inputs at the exact immutable revisions frozen by the v0.10 driver:

- Qwen3-ASR 0.6B: `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`
  (1.8 GB cache; `model.safetensors` SHA-256 matches the Nano gray input above);
- Qwen3-TTS 0.6B Base: `5d83992436eae1d760afd27aff78a71d676296fc`
  (2.4 GB cache);
- Qwen3-TTS 0.6B CustomVoice:
  `85e237c12c027371202489a0ec509ded67b5e4b5` (2.4 GB cache).

It also retains a 227 MB MOSS-TTS-Nano source directory and a 3.7 GB Spark-TTS
source directory for their model-specific v0.10 re-export lanes. The 8.7 GB
Qwen3.5 directory is an older AWQ workspace, not the official
`Qwen/Qwen3.5-4B` revision required by the v0.10 NVFP4 driver, so it is not
accepted as that lane's source checkpoint.

The WSL cache may replace network download only. Quantization keeps the
previously qualified per-model recipe, while every v0.10 ONNX and TensorRT
engine is regenerated. Old ONNX/engines remain mechanism-only evidence and
cannot be promoted into the v0.10 artifact set.

## Qwen3-ASR formal INT4 checkpoint and ONNX gate (2026-08-15)

The exact ASR revision above was quantized on `wsl2-local` with the complete
patched v0.10 Python tree and its pinned toolchain (Torch 2.13.0, Transformers
5.14.1, ModelOpt 0.45.0). The calibration contract is the previously qualified
W4A16-AWQ recipe: LibriSpeech multimodal audio+transcript calibration, 128
samples, group size 128, no zero point, pre-quant scale enabled, audio tower
FP16, and tied embedding/LM head excluded. Quantization completed in 370.7 s;
the checkpoint `model.safetensors` SHA-256 is
`8bb301e19569ba8c41224e9513190c5fa76073de2c1d5d2caf109fe9fdc69e30`.

v0.10 changes the INT4 exporter default to CuTe-DSL
`Int4GroupwiseGemmPluginV2`. That is a different weight layout/kernel path
from the AWQ-swizzled v1 plugin qualified on Orin. The formal export therefore
sets `--int4-gemm-plugin-version 1`; the aggregate driver and manifests now
freeze that choice for both ASR and the product CustomVoice INT4 driver.

The fail-closed `validate-asr-onnx.py` gate passes with 1,136 thinker nodes,
515 audio nodes, exactly 196 `Int4GroupwiseGemmPlugin` v1 nodes, zero v2 nodes,
all required sidecars present, and both ONNX models accepted by
`onnx.checker`. Core SHA-256 values are:

- thinker `model.onnx`:
  `2fc79431026a8d84002a48a27825e6153e5618634cb4d1976d1285b7b33d5ee6`;
- thinker external weights:
  `d1f0b7f602c96e5fd7293f28b2ab2b40622a77d5850d1afaa536e2ed0cc78a35`;
- full-vocab embedding:
  `70fb4840066b259c8a90aea869a9ed204dd7f13240665ab8bbb7980046dc8964`;
- audio encoder `model.onnx`:
  `50d4de338fcd867308867600af37ed193a8161bac62f338328f66045df96fcef`.

This closes the checkpoint/export gate. The engine and small-corpus PCM gray
gate below also passes, but the full production greedy/force-language ZH CER
and EN WER corpus remains required before replacing the FP16 gray set.

## Qwen3-ASR formal INT4 engine and PCM gray gate (2026-08-15)

All three engines were built natively on `orin-nx` (JetPack 6.2, CUDA 12.6,
TensorRT 10.3, SM87) from the checked ONNX above. The builder and plugin are
the already-qualified v0.10 binaries recorded in the device compile gate. The
release profiles are unchanged from v0.9.1: thinker b1/b2 use maximum input
1,024 and KV capacity 1,536; the audio encoder uses 100--3,000 time steps.

The resulting SHA-256 values are:

- thinker b1 engine:
  `b1dd878acc5ee7f045cf274d525f1ede997f32f92b823b48d7892c85409ebf1b`;
- thinker b1 config:
  `6d6c1a6c307e394349aabb234c241c44a78e3b5885d2190cdbf9714f6f2fb545`;
- thinker b2 engine:
  `62085451b8d416db2d11838583a9ce3d62d47c03e25581f8a56011b91659b0d2`;
- thinker b2 config:
  `0a4c0caf39905f0deaf0b726caedab97c93975dd338040b8d4368c79a507289e`;
- audio encoder engine:
  `a359ac5d35a0ad9d1f4666da0e4b10d13ee676bd8e6c09a6b571951392ab506a`;
- audio encoder config:
  `30e489e9c3982f42fe3976ad6e5391adffda3aae5b23ff0998d53d7159e78193`.

Each engine directory was copied to `wsl2-local`, checked file by file, then
restored to NX and checked again before runtime. No x86-built TensorRT engine
is used. The worker is the same guarded v0.10 binary used by the FP16 Nano gray
lane, SHA-256
`3381a9ce35d32c199ac37a65cecfe772b056678a48796d296d4f2520e752eab2`.

The production `pcm_b64` path passed all eight existing requantization samples
(five Chinese and three English) on both b1 and b2. After punctuation
normalization every result has LCS 1.0 against the previous validated text.
Final-chunk latency ranges were 78.9--206.8 ms on b1 and 79.3--208.1 ms on b2.

The b2 session gate also passes: lanes 0 and 1 can co-reside, a third begin
fails closed with `pool_saturated` / status 4429, the two distinct Chinese and
English samples finalize serially with isolated LCS 1.0 text, and lane 0 is
reusable afterward. The checked finalizations took 589.8 ms and 149.3 ms from
the client side. This confirms the current two-resident-session contract; it
does not provide continuous batching or mid-decode admission.

The old test driver's precomputed `mel_path` scenarios fail with the v0.10
formal export (`TensorRT Edge LLM cannot handle this request`) while the actual
product `pcm_b64` path passes. That legacy compatibility path is not credited
to the gray gate and must not be used as the release health check. The failure
is retained as a compatibility difference rather than hidden by the PCM pass.

## Gates still required before an overall OVS upgrade

- Regenerate every ONNX and TensorRT engine with v0.10 identity.
- Publish model-owned immutable artifacts and a new outer v0.10 release lock.
- Replace the ASR FP16 gray set with formal INT4 artifacts; keep native audio
  batch disabled until its two-WAV isolation defect is fixed.
- Run Qwen3-TTS CustomVoice, MOSS, Spark, Qwen3.5, cancellation/recovery,
  byte/parity, cross-model co-residency, and release-profile RSS gates.
- Build new v0.10 runtime images/profiles/compose identities while preserving
  v0.9.1 as rollback.

Until those gates pass, the driver may produce only isolated v0.10 candidate
artifacts and outer OVS v0.9.1 release identities must not be changed.
