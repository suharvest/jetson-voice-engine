# int4-AWQ ASR decoder export

Quantize the Qwen3-ASR thinker (decoder) to **int4-AWQ (W4A16)** using the
upstream `tensorrt_edgellm` quantization path, with the fork fix that guards
ModelOpt 0.39's `export_hf_checkpoint(extra_state_dict=...)` kwarg for the int4
path. Validated functionally non-degrading (ZH CER ~0%, EN WER ~11% =
near-homophone word drops, not corruption).

- **Artifact:** int4-AWQ ASR decoder HF checkpoint (`W4A16_AWQ`,
  `hf_quant_config.json`) → native ONNX export → TRT engine via `llm_build`
  on Jetson.
- **Inputs (public weights):** `Qwen/Qwen3-ASR-0.6B`.
- **Pin:** fork `suharvest/wip/asr-int4-decoder` @ **`c80bcc0`**
  ("fix(asr-int4): guard export_hf_checkpoint extra_state_dict kwarg for
  modelopt 0.39"). This branch sits on top of `ff2318e` (the int4 drivers) and
  patches `tensorrt_edgellm/quantization/quantize.py` so the standard quantize
  CLI works for `int4_awq` under ModelOpt 0.39.
- **md5:** not pinned (regenerate; engine md5 recorded with the published set).
- **HF:** published alongside ASR engine sets under
  `harvestsu/seeed-local-voice-artifacts` / `harvestsu/qwen3-edgellm-jetson-artifacts`.
- **Build device:** ModelOpt quant + ONNX export on a CUDA GPU host; TRT engine
  on **SM 87 Jetson Orin**.

> Validation pitfalls (do NOT trust a naive harness): match the **production
> decode contract** — greedy `top_k=1` (not sampling), prime the assistant turn
> with the `force_language` scaffold (`language <Lang><asr_text>`), then
> `stripLanguagePrefix`. A sampling/no-scaffold harness falsely reports "int4
> damaged". See memory `qwen3asr_int4_validation_forcelanguage_2026_06_20`.

## Driver / patch (reference — read-only)

```bash
git -C ~/project/TensorRT-Edge-LLM show c80bcc0 --stat
git -C ~/project/TensorRT-Edge-LLM show c80bcc0:tensorrt_edgellm/quantization/quantize.py
```

## Commands

Check out the fork branch (read-only; do not modify), then run the upstream
quantize CLI with the int4_awq format against the ASR weights:

```bash
PY=~/TensorRT-Edge-LLM/.venv/bin/python
ASR=<hf-snapshot-dir of Qwen/Qwen3-ASR-0.6B>
OUT=~/asr-int4

# int4-AWQ quantize (the c80bcc0 fix makes export_hf_checkpoint succeed on
# modelopt 0.39 for this path)
$PY -m tensorrt_edgellm.quantization.quantize \
    --model_dir "$ASR" \
    --output_dir "$OUT/_int4" \
    --qformat int4_awq
# (use the exact CLI flags of the pinned quantize.py — verify with --help)
```

Then native ONNX export + `llm_build` engine build on Jetson, following the
`--fp8_embedding` thinker recipe in
[`../docs/asr-thinker-engine-build-recipe.md`](../docs/asr-thinker-engine-build-recipe.md)
(point the export/build at the int4 checkpoint). For the N>1 (`maxBatchSize=2`)
engine variant see [`asr-b2-engine.md`](asr-b2-engine.md).

## Verification (functional)

Drive a one-shot ASR request through the production worker decode path (greedy +
force_language scaffold). Sane ZH transcription = pass; spot-check EN against a
faster-whisper reference for intelligibility.
