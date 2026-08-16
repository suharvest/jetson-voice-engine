# TensorRT-Edge-LLM v0.10.0 feature adoption

Source of truth: NVIDIA `CHANGELOG.md` and the v0.10.0 support matrix at
`UPSTREAM_PIN`. This ledger separates useful product work from model-list
churn; it does not turn experimental upstream features on by default.

## Adopt in the v0.10 migration

- **Qwen3-TTS language, instruction, VoiceDesign, and on-device clone.** Use
  the native request/config/clone-encoder interfaces. This retires our old
  9-row language, Base-export, and codec-language patches and replaces the
  external embedding implementation. Keep a temporary OVS compatibility
  adapter; callers must migrate from `speaker_embedding_b64` to reference
  audio/text.
- **CuTe DSL FMHA-v2.** This replaces the deleted legacy embedded-cubin FMHA.
  It is a mandatory runtime migration, not an optional product switch. JP6.2
  needs a locally generated SM87/CUDA12 `fmha` artifact because NVIDIA ships
  the v0.10 SM87 prebuilt only for CUDA13.
- **Paged KV-cache reuse.** Enable only after A/B tests on our Qwen3.5 chat
  path cover MTP/GDN, 4K/8K context, concurrent sessions, cancellation, memory
  watermark, byte parity, and repeated-prefix hit/miss behavior.

Paged KV, per-iteration active batch sizing, and finished-slot compaction are
clear prerequisites for continuous batching, but v0.10 does not yet refill a
running decode batch with newly arrived requests. Its server `RequestBatcher`
only forms a fixed micro-batch of compatible non-streaming requests and waits
for that runtime call to finish; streaming requests do not use this batcher.
Track upstream scheduler work, but do not advertise continuous batching yet.

## Opt-in evaluation after the base upgrade

- **Nemotron-3.5-ASR: evaluated, not adopted.** The offline batch-1 v0.10
  runner passed the Orin steady-state RTF gate but failed Chinese quality
  (26.03% CER versus Qwen's 6.58% on the same corpus). The official
  Transformers path reproduced the loss, so it is not an ONNX/TensorRT parity
  regression. English passed the absolute gate but remained behind Qwen. Do
  not publish engines, add a production profile, or claim cache-aware
  streaming from this experimental runner; see
  `NEMOTRON-ASR-VALIDATION-20260816.md`.
- **Nemotron-3.5 Lightning, MTP/DFlash, and DSpark.** Potentially useful for
  chat latency, but Orin memory and the JP6.2 compatibility tier make these
  canary-only until measured. Do not replace the existing Qwen3.5 default.
- **Qwen3-Omni streaming audio and OpenAI `/v1/audio/speech` / transcription
  endpoints.** Useful as an internal compatibility surface. Preserve the OVS
  public protocol through an adapter instead of exposing upstream API changes
  directly.
- **Experimental direct builder without ONNX.** Evaluate build time and engine
  parity in CI, but keep ONNX export as the release source of truth for v0.10.

## Not on the current voice roadmap

Cosmos3-Edge, DiffusionGemma, Gemma4, video input, and Claude/OpenClaw
integration do not solve a current OVS voice-path requirement. They stay out
of the release profile unless a product scenario is approved separately.

## Compatibility constraints

The Orin target remains SM87 / JetPack 6.2 / CUDA 12.6 / TensorRT 10.3 and is
listed by upstream as compatible rather than the primary tested platform.
Orin supports FP16/INT8/INT4, not FP8/FP4 engine execution. Regenerate all ONNX
and TensorRT engines for the v0.10 source/plugin identity; never mix v0.9.1
engines, plugins, or workers with v0.10 artifacts.
