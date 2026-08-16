# Nemotron-3.5-ASR v0.10 validation (2026-08-16)

## Decision

Do not add Nemotron-3.5-ASR to a production profile and do not publish its
engines or an image. The TensorRT steady-state performance gate passed, but the
multilingual quality gate failed: Chinese CER was 26.03% versus the 10% limit
and the incumbent Qwen v0.10 result of 6.58%. English WER passed the absolute
8% limit at 6.33%, but was also behind Qwen at 4.88%.

The official Transformers reference reproduced the same quality: 26.34% Chinese
CER and 6.33% English WER. TensorRT matched 17 of 20 transcripts exactly; the
other three differed by only a few Chinese characters, with TensorRT's aggregate
CER slightly better. This clears the HF-to-ONNX-to-TensorRT conversion parity
question and attributes the failed Chinese gate to the checkpoint/domain rather
than v0.10 conversion loss.

This evaluation does not change or fail the qualified v0.10 release. Nemotron
remains an experimental lane outside its release lock.

## Pinned inputs

- Model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- HF revision: `1c8deaecc64b91f034d73e08dd8b64625eb3395d`
- `model.safetensors`: `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174`
- TensorRT-Edge-LLM: v0.10.0
- Target: fleet `orin-nano`, JetPack 6.2, CUDA 12.6, TensorRT 10.3,
  SM87, `MAXN_SUPER`
- Corpus: 20 pinned Chinese/English short and long WAVs; archive SHA-256
  `df555d34a1c454c3c1324a68121b7088674736a66d89a2601fd129df788d880e`
- Prompt IDs: `zh-CN=4`, `en-US=0`

The model checkpoint advertises cache-aware streaming, but the v0.10
experimental Edge-LLM runner exercised here is offline, batch 1, and does not
expose the checkpoint's cache-aware streaming state. No streaming claim is made
from this result.

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| v0.10 Edge-LLM plugin | `e6f445c3c471c8ff49736fe835c586524d28eef23d1851f57f12edaf5a1783f2` |
| `audio_encoder.engine` (FP16, max 8192 mel frames) | `f03953e6c586a2e851a6056f52bc101ae4556946cb32a39cd00b7a9eb5974efa` |
| `rnnt_step.engine` | `b214ef51165561300208f7778fb9eaf3d4e1542d33697dcc68132cf036ee0fe8` |
| ONNX archive | `09a58bff9feaac40f0bc15c1475c4c4fd7dad991ce187d19a6e2972c11f66622` |
| Audio ONNX / data | `af6d5b714340eabe246c85359f45d7429b759d95493639885eb8b31736f2d4c3` / `e6856bb1a92e20b63ccddaafd7aca844741262493e9547726e59d2ceb999540c` |
| RNNT decoder ONNX / data | `dd9e7081ac366cbcdeea32896d421d8255c9513434a4c37a4c8a9e60f1d4a3ea` / `cb9d944d0f562c34ef1a459f8881eca7d98a82ca4dcb33e37da3d375c52e88d5` |

The engine build took 122.8 seconds for the encoder. The serialized encoder is
1,284,901,860 bytes and the RNNT step engine is 47,558,252 bytes. The maximum
8192-mel-frame profile covers about 81.9 seconds of audio.

## Results

| Path | Chinese CER | English WER | Notes |
|---|---:|---:|---|
| TensorRT v0.10 FP16 | 26.03% | 6.33% | steady RTF 0.035 (`zh_short_02`) and 0.030 (`en_long_03`) |
| Transformers FP32 | 26.34% | 6.33% | PyTorch 2.13.0+cu130, Transformers 5.14.1 |
| Qwen v0.10 incumbent | 6.58% | 4.88% | same 20 WAVs and normalization |

The performance threshold was RTF <= 0.08. Nemotron passed on both steady-state
representatives. The one-process-per-sample cold wall time was about 6.7 seconds
because it includes engine deserialization and process startup; it is not a
service latency result and must not be substituted for first-token latency.

The machine-readable summary is
`bench/perf/baselines/edgellm-v010-nemotron-asr-orin-nano-evaluation.json`.
The reproducible runners are `bench/perf/nemotron_asr_offline_gate.py` and
`bench/perf/nemotron_asr_hf_reference.py`.

## Evaluation-only build accommodations

The upstream v0.10 experimental CMake target aggregates unrelated experimental
models. For this device check it was narrowed to Nemotron only. Also,
`ENABLE_CUTE_DSL=OFF` still globbed the two CuTe FMHA runner sources; the device
copy conditionally excluded those sources because NVIDIA's shipped SM87 CuTe
prebuilt is CUDA 13 while this target is CUDA 12.6. AppleDouble files in the
transferred source were removed from the device copy.

These accommodations were limited to the evaluation tree and are not part of
the production overlay, release lock, or published image identity. Before any
future product adoption, they must be converted into reviewed overlay patches
and the cache-aware streaming runtime must receive its own correctness,
cancellation, recovery, and concurrency gates.
