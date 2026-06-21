# int4-AWQ TTS export (Talker + CodePredictor) + fp8 text_embedding

Quantize the Qwen3-TTS Talker backbone **and** the CodePredictor (CP)
transformer to **int4-AWQ (W4A16, group_size 128)**, and optionally quantize
the `text_embedding` table to **fp8 (e4m3)** for the runtime's native FP8
embedding-lookup path. Weight-only int4 on the backbone Linears; `lm_head`
(EOS / codec heads) and CP `down_proj` stay FP16/FP32 to preserve EOS and
codec-token fidelity.

- **Artifact:** int4-AWQ Talker ONNX (`Int4GroupwiseGemmPlugin` nodes) + int4
  CP ONNX + fp8 `text_embedding.safetensors`. Engines are then built from these
  ONNX dirs with `llm_build` on a Jetson (see `asr-b2`/`talker-b2` for the
  `llm_build` step pattern).
- **Inputs (public weights):** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (or
  `…-CustomVoice` for the CV variant). The drivers rebuild the talker/CP as a
  vanilla `transformers` `Qwen3ForCausalLM`, run ModelOpt `INT4_AWQ_CFG`, and
  re-assemble a checkpoint the native `tensorrt-edgellm` ONNX export consumes.
- **Pin:**
  - int4 drivers — fork `suharvest/wip/native-int4-talker` @ **`ff2318e`**
    (`quantize_talker_stage1.py`, `stage2_export.py`,
    `quantize_cp_stage1.py`, `cp_stage2_export.py`).
  - fp8 embedding — fork `suharvest/wip/fp8-embedding` @ **`873ca22`**
    (`scripts/quantize_text_embedding_fp8.py`).
  - Larger-calibration talker variant `quantize_talker_stage1_bigcalib.py` +
    the derisk harness (`fp16_stage{1,2}.py`, `check_onnx_nodes.py`, run
    scripts) are in the Mac backup tarball
    `~/project-backups/20260621-133543/wsl2-recovered/wsl2-int4-drivers-uncommitted.tar.gz`
    (extracted at `…/extracted/{project/edgellm-v080-export,cv-int4-derisk}`).
- **md5:** export checkpoints/ONNX are not pinned here (regenerate); engine md5s
  are recorded per published engine set on HF.
- **HF:** `harvestsu/customvoice-...int4fp8` (CV int4 Talker + fp8 embedding),
  `harvestsu/qwen3-tts-0.6b-base` (Base bundle).
- **Build device:** ModelOpt quant/ONNX export on any CUDA GPU host (the
  recovered runs used a WSL2 box, fork venv `~/TensorRT-Edge-LLM/.venv`); the
  TRT engine build from the ONNX is on an **SM 87 Jetson Orin**.

> Background / pitfalls (why lm_head & CP down_proj stay high-precision; the
> "wrong tokenizer" runaway saga; int4 is functionally clean on Base): see the
> project memory notes `customvoice_talker_int4_eos_fail_2026_06_20` and
> `qwen3asr_int4_validation_forcelanguage_2026_06_20`.

## Driver scripts (reference — do not copy the full bodies)

Read the canonical sources read-only from the fork:

```bash
git -C ~/project/TensorRT-Edge-LLM show ff2318e:quantize_talker_stage1.py
git -C ~/project/TensorRT-Edge-LLM show ff2318e:stage2_export.py
git -C ~/project/TensorRT-Edge-LLM show ff2318e:quantize_cp_stage1.py
git -C ~/project/TensorRT-Edge-LLM show ff2318e:cp_stage2_export.py
git -C ~/project/TensorRT-Edge-LLM show wip/fp8-embedding:scripts/quantize_text_embedding_fp8.py
```

## Commands

Set the fork venv python + export-driver workdir on `PYTHONPATH` (so the
drivers can `import tensorrt_edgellm`):

```bash
PY=~/TensorRT-Edge-LLM/.venv/bin/python
export PYTHONPATH=<dir-containing-the-driver-scripts>   # e.g. project/edgellm-v080-export
BASE=<hf-snapshot-dir of Qwen3-TTS-12Hz-0.6B-Base or -CustomVoice>
OUT=~/tts-int4
```

### Talker — Stage 1 (int4-AWQ quantize backbone)

```bash
$PY quantize_talker_stage1.py \
    --model_dir   "$BASE" \
    --output_dir  "$OUT"            # writes $OUT/_hf_unified (W4A16 + hf_quant_config.json)
    # --num_samples 256  --max_len 64   (calibration defaults)
```

### Talker — Stage 2 (re-assemble int4 ckpt + native ONNX export)

```bash
$PY stage2_export.py \
    --orig_model_dir "$BASE" \
    --unified_dir    "$OUT/_hf_unified" \
    --stage2_ckpt    "$OUT/_int4_talker_ckpt" \
    --onnx_out       "$OUT/llm"          # final int4 Talker ONNX
```

### CodePredictor — Stage 1 + Stage 2 (same two-stage pattern)

```bash
$PY quantize_cp_stage1.py \
    --model_dir  "$BASE" \
    --output_dir "$OUT/cp"               # --num_samples 256 --seq_len 16

$PY cp_stage2_export.py \
    --orig_model_dir "$BASE" \
    --unified_dir    "$OUT/cp/_hf_unified_cp" \
    --stage2_ckpt    "$OUT/_int4_cp_ckpt" \
    --onnx_out       "$OUT/cp_llm"
```

### fp8 text_embedding (memory-only; no speed/quality cost)

```bash
$PY ~/TensorRT-Edge-LLM/scripts/quantize_text_embedding_fp8.py \
    text_embedding.safetensors \
    text_embedding.fp8.safetensors \
    --block 128        # block must be a multiple of 16 and divide hidden
```

Produces two tensors: `text_embedding` (FP8 e4m3 table) + `text_embedding_scales`
(FP32 per-group). The runtime classifies by dtype and dequantizes in
`embeddingKernels.cu` `Fp8EmbeddingLoader`.

### Build engines from the int4 ONNX (on Jetson)

Same `llm_build` invocation pattern as [`asr-b2-engine.md`](asr-b2-engine.md) /
[`talker-b2-engine.md`](talker-b2-engine.md), pointed at `$OUT/llm` and
`$OUT/cp_llm`. ModelConfig auto-detects `W4A16_AWQ` from `hf_quant_config.json`
and emits the int4 GEMM plugin nodes.

## Verification (functional, not byte)

ASR-roundtrip the synthesized audio with a known-good reference (faster-whisper)
and confirm intelligible output. Stage-2 CP fix is load-bearing:
`hf_quant_config.json` **must** be present in the CP export temp dir or the
loader silently builds FP16 and int4 tensors are dropped → q_proj reshape
failure (the inline export in `cp_stage2_export.py` handles this).
