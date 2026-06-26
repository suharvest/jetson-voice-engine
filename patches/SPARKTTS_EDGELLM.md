# SparkTTS edge-llm framework changes — organization

How our SparkTTS-on-Jetson framework changes to TensorRT-Edge-LLM are split:
generic/upstream → fork **main**; SparkTTS-layered → **patches here**.
(Replaces the one-off `feat/sparktts-converged` branch, which is superseded.)

## ① bf16/fp16 mixed-precision  →  fork `main` (NOT a patch here)
- Branch: `suharvest/TensorRT-Edge-LLM` `sparktts/bf16-mixed-precision` (cut from
  `origin/main` = NVIDIA 0.8.0, 5 cherry-picked commits, +220/-21, 5 files:
  config.py / models/default/modeling_default.py / models/linear.py /
  checkpoint/loader.py / onnx/export.py).
- Generic capability: opt-in `config.mixed_precision` (BF16 residual/MLP +
  FP16 attention island via Cast nodes). **Default unset = byte-identical** to
  upstream → does not affect existing models. **Upstream-PR candidate.**
- This is the EdgeLLM-baseline change; lands on fork main, not carried as a patch.

## ② W4A16 (INT4-AWQ) with BF16 GEMM output  →  patch
- `v080-0030-w4a16-int4-bf16-output.patch` (14 files, +~1100; dev-only
  `bf16_overflow_microbench.cu` intentionally excluded).
- int4 GEMM/GEMV kBF16 output path + plugin `output_dtype` opt-in attr + AWQ
  `mixed_precision_with_quant` + per-linear routing (MLP int4→bf16 out, attn fp16).
- **Depends on ①** (layers config.py / linear.py on the mixed-precision base).
- Opt-in; default int4 path unchanged. Upstream-PR candidate (follow-up to ①).

## ③ shared LLM engine across N slots (borrowed ctor)  →  patch
- `v080-0031-shared-engine-borrowed-ctor.patch` (4 files: cpp/runtime/exec/
  engineExecutor.{cpp,h}, cpp/runtime/llmInferenceRuntime.{cpp,h}).
- `EngineExecutor::createForLLMBorrowed` + `LLMInferenceRuntime(borrowedBaseEngine,…)`
  so N slots share one deserialized engine (path ctor unchanged, opt-in).
- **Depends on the prior voice-runtime baseline** (reuses the existing
  `legacy/llmEngineRunner` + `qwen3OmniTTSRuntime` borrowed-engine pattern). Must
  apply over a baseline that already carries that prior work.

## Apply order (over the EdgeLLM voice-runtime baseline)
The baseline = the fork's voice-runtime checkout (prior qwen3-tts/CustomVoice/ASR
work) **rebased to include ① (fork main)**. Then, in the build setup:
1. (① already present via main)
2. `git apply patches/v080-0030-w4a16-int4-bf16-output.patch`
3. `git apply patches/v080-0031-shared-engine-borrowed-ctor.patch`

## Validation (converged W4A16 stack, orin-nx, 2026-06-26)
ASR clone CER 0 / EN WER 0.02; N=2 MD5 byte-identical, 0 CUDA errors; real-machine
RTF gen 0.74 (clone e2e 0.81 / controllable 0.96); VRAM N=1 ~360 MB / N=2 ~700 MB
(GPU, cudaMemGetInfo); engine 645 MB (−58% vs fp16-hybrid).

## TODO (unified push — owner decision)
- Land `sparktts/bf16-mixed-precision` onto fork `main` (+ rebase voice-runtime
  baseline onto it so the build picks up ①).
- Wire patches 0030/0031 into the EdgeLLM-baseline build-setup apply step.
- Delete the superseded `feat/sparktts-converged` (fork + local).
- Push the prior voice-runtime branches to the fork remote (currently local-only).
