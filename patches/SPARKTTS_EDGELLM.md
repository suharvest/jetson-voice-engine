# SparkTTS edge-llm framework changes — organization

**All** SparkTTS edge-llm framework changes live on the single engine-overlay
baseline branch (the pin), NOT split across fork-main + jve patches. This keeps
upstream maintenance to ONE rebase.

## Where
- Branch: `suharvest/TensorRT-Edge-LLM` `integration/v080-sparktts`
  (= `c48c0de` Base+CustomVoice baseline + ①②③, all opt-in / default byte-identical).
- Pinned by `engine-overlay/UPSTREAM_PIN` = `8437f027`.
- ① bf16/fp16 mixed-precision · ② W4A16 INT4-AWQ BF16-output · ③ shared-engine
  ctor (EngineExecutor + LLMInferenceRuntime). See UPSTREAM_PIN for details.

## Why one branch (not fork-main + patches)
An upstream NVIDIA bump rebases this ONE branch onto the new release; jve only
re-pins (a SHA bump) — no patch re-conflicts, no per-repo churn. The Base /
CustomVoice variants are unaffected (the changes are opt-in). SparkTTS is a
product variant = this baseline + its own worker + config, not a separate pin.

## Upstream-PR candidates (independent of the build pin)
① (and follow-ups ②③) are generic, opt-in, upstream-PR candidates; a clean
`sparktts/bf16-mixed-precision` branch (off NVIDIA 0.8.0) carries ① for that.

## Validation (orin-nx, 2026-06-26)
ASR clone CER 0 / EN WER 0.02; N=2 MD5 byte-identical, 0 CUDA errors;
RTF gen 0.74; VRAM N=1 ~360MB / N=2 ~700MB (cudaMemGetInfo); engine 645MB.
