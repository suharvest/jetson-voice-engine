# Export ONNX from Official Qwen3 Weights

This repo does not require users to download our ONNX files. A user can start
from the official Qwen3 ASR/TTS Hugging Face snapshots, export ONNX with the
TensorRT-Edge-LLM fork, then build device-specific TensorRT engines on Jetson.

## 1. Prepare the EdgeLLM fork

```bash
git clone https://github.com/suharvest/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout qwen3-tts-highperf-runtime-w8a16
git submodule update --init --recursive
```

Use `official-qwen3-tts-upstream-runtime` instead if you only want the
minimal-diff upstream-review path.

## 2. Create the WSL2/x86 export uv environment

The export environment should run on x86/WSL2 with an NVIDIA GPU. It creates a
separate uv project, installs TensorRT-Edge-LLM as editable, and applies the
Qwen/transformers compatibility patches used by our exports.

```bash
cd qwen3-edgellm-jetson

TRT_SRC=$HOME/project/TensorRT-Edge-LLM \
TRT_EXPORT_PROJECT=/tmp/trt-export \
PYTHON=3.12 \
PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple \
bash scripts/setup_trt_export_env.sh
```

If your network needs a local proxy, leave `DISABLE_PROXY` unset and configure
`HTTP_PROXY`/`HTTPS_PROXY` before running the script. If you explicitly want no
proxy, set `DISABLE_PROXY=1`.

The script expects the Qwen Python packages to be available in the user Python
site-packages or via explicit paths:

```bash
QWEN_ASR_PKG_DIR=/path/to/qwen_asr \
QWEN_TTS_PKG_DIR=/path/to/qwen_tts \
QWEN_OMNI_UTILS_PKG_DIR=/path/to/qwen_omni_utils \
bash scripts/setup_trt_export_env.sh
```

The resulting environment lives at `/tmp/trt-export` by default. Re-run with
`RESET=0` to reuse the directory without deleting it.

## 3. Export Qwen3-ASR ONNX

Assume the user has downloaded the official Qwen3-ASR snapshot to
`/models/Qwen3-ASR-0.6B`:

```bash
scripts/export_qwen3_asr_onnx.sh \
  --model-dir /models/Qwen3-ASR-0.6B \
  --out /tmp/qwen3-asr-onnx \
  --export-project /tmp/trt-export \
  --device cuda \
  --fp8-embedding
```

Output:

- `/tmp/qwen3-asr-onnx/llm`: ASR thinker/LLM ONNX and sidecars.
- `/tmp/qwen3-asr-onnx/audio`: ASR audio encoder ONNX and config.
- `/tmp/qwen3-asr-onnx/export_manifest.json`: export metadata.

Optional knobs:

- `--export-models thinker` if you want to force the LLM export filter.
- `--reduced-vocab-dir <dir>` for vocab-pruned experiments.
- `--audio-quantization fp8` for audio encoder quantization experiments.

Production default for our current profile is full vocab; pruning is not
enabled by default.

## 4. Export Qwen3-TTS ONNX

Assume the user has downloaded the official Qwen3-TTS snapshot to
`/models/Qwen3-TTS-0.6B`:

```bash
scripts/export_qwen3_tts_onnx.sh \
  --model-dir /models/Qwen3-TTS-0.6B \
  --out /tmp/qwen3-tts-onnx \
  --export-project /tmp/trt-export \
  --trt-src $HOME/project/TensorRT-Edge-LLM \
  --device cuda
```

Output:

- `/tmp/qwen3-tts-onnx/official/llm`: official Talker + CodePredictor ONNX.
- `/tmp/qwen3-tts-onnx/official/audio`: official tokenizer decoder ONNX.
- `/tmp/qwen3-tts-onnx/highperf/talker_w8a16`: W8A16 Talker ONNX variants.
- `/tmp/qwen3-tts-onnx/highperf/code_predictor`: optimized/pretransposed CP ONNX.
- `/tmp/qwen3-tts-onnx/highperf/code2wav_stateful`: stateful Code2Wav ONNX.
- `/tmp/qwen3-tts-onnx/export_manifest.json`: export metadata.

Use `--official-only` to skip all high-performance transforms. Use
`--no-w8a16-talker`, `--no-fp8-text-embedding`, or `--no-stateful-code2wav` to
disable individual transforms.

Stateful Code2Wav export validates against RVQ codes. If `--code2wav-codes` is
not provided, the wrapper creates a tiny synthetic `rvq_codes` safetensors file
for interface export. For quality validation, pass real RVQ codes captured from
Talker output.

## 5. Build TensorRT engines on Jetson

ONNX export is platform-independent. TensorRT engines are device/tactic
specific and should be built on the target Jetson class.

For the current highperf path, use the build helpers in this repo:

```bash
scripts/build_qwen3_nx_native_engines.sh
```

For deployment, upload only the runtime artifacts to the HF artifact repo:
`.engine`, config/tokenizer, embedding/projection sidecars, and checksums. ONNX
files are intermediate build products and are intentionally not part of the
default runtime artifact set.
