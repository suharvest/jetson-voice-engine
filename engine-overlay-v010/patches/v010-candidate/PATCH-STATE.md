# TensorRT Edge-LLM v0.10.0 product patch state

Official base: NVIDIA tag `v0.10.0`
(`71dd1bae032e70771265917ec74d3ff4cad07a10`).

The active product series contains 32 sparse patches after four generic
upstream fixes and additive files from `addon/`.

## Retired on v0.10.0

- old `0024` CustomVoice 9-row language conditioning: v0.10.0 provides native
  Qwen3-TTS language and instruction conditioning;
- old `0035` Base export guard relaxation: v0.10.0 natively supports Base,
  CustomVoice, VoiceDesign, and clone encoders;
- old `0036` `codec_language_id` export: v0.10.0 carries the language metadata.

The former external-speaker-embedding path is replaced by the native
`cloneEncoderDir` plus `ref_audio` / `ref_text` contract. The product worker
temporarily accepts `speaker_embedding_b64` for protocol compatibility, warns,
and ignores it; callers must migrate to the native reference-audio contract.

## Retained product capability

- Orin TTS fallback GEMM/GEMV and executable linkage;
- ASR lane/session concurrency;
- SparkTTS BF16/W4A16 modeling, export, kernels, and plugins;
- Qwen3-TTS streaming, empty-prefill guard, FP8 table loader, language wiring,
  cooperative cancellation, shared-engine slots, and chunk aggregation;
- MOSS-TTS-Nano runtime, build integration, concurrent dispatch, and cancel.

Plain AWQ/ModelOpt export supports the v0.10.0 INT4 V1/V2 dispatch. The BF16
output extension requires `--int4-gemm-plugin-version 1` and fails clearly for
V2 rather than producing an incompatible graph.

The complete 4+32 chain passes integrated forward and reverse replay against
the official v0.10.0 tree. Device compile, engine regeneration, and runtime
qualification remain release gates.
