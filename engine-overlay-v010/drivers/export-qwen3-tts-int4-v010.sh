#!/usr/bin/env bash
# Export a previously qualified Qwen3-TTS INT4 stage-2 Talker with the exact
# TensorRT-Edge-LLM v0.10.0 exporter. The quantization checkpoint remains a
# separate, immutable input; this driver never relabels an older ONNX graph.
set -euo pipefail

PIN="71dd1bae032e70771265917ec74d3ff4cad07a10"
upstream="${TTS_INT4_UPSTREAM:?TTS_INT4_UPSTREAM is required}"
model="${TTS_INT4_MODEL:?TTS_INT4_MODEL is required}"
model_revision="${TTS_INT4_MODEL_REVISION:?TTS_INT4_MODEL_REVISION is required}"
stage2="${TTS_INT4_STAGE2_CHECKPOINT:?TTS_INT4_STAGE2_CHECKPOINT is required}"
stage2_revision="${TTS_INT4_STAGE2_REVISION:?TTS_INT4_STAGE2_REVISION is required}"
output="${TTS_INT4_OUTPUT:?TTS_INT4_OUTPUT is required}"
plugin_version="${TTS_INT4_GEMM_PLUGIN_VERSION:-1}"
driver_sha="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"

fail() {
  echo "ERROR: $*" >&2
  exit 12
}

[ "${TTS_INT4_PRECISION:-int4}" = "int4" ] \
  || fail "TTS_INT4_PRECISION must be int4"
[ "${plugin_version}" = "1" ] \
  || fail "Orin release export requires TTS_INT4_GEMM_PLUGIN_VERSION=1"
[ -e "${upstream}/.git" ] || fail "upstream is not a Git checkout: ${upstream}"
[ "$(git -C "${upstream}" rev-parse HEAD)" = "${PIN}" ] \
  || fail "expected TensorRT-Edge-LLM v0.10.0 ${PIN}"
[ -s "${model}/config.json" ] || fail "model config missing: ${model}/config.json"
[ -s "${stage2}/config.json" ] || fail "stage-2 config missing: ${stage2}/config.json"
[ -s "${model}/model.safetensors" ] || fail "model weights missing: ${model}/model.safetensors"
[ -s "${stage2}/model.safetensors" ] || fail "stage-2 weights missing: ${stage2}/model.safetensors"
[ -s "${stage2}/hf_quant_config.json" ] || fail "stage-2 quant config missing: ${stage2}/hf_quant_config.json"
[ ! -e "${output}" ] || fail "output must be fresh: ${output}"

model_config_sha="$(sha256sum "${model}/config.json" | awk '{print $1}')"
model_weights_sha="$(sha256sum "${model}/model.safetensors" | awk '{print $1}')"
stage2_config_sha="$(sha256sum "${stage2}/config.json" | awk '{print $1}')"
stage2_quant_config_sha="$(sha256sum "${stage2}/hf_quant_config.json" | awk '{print $1}')"
stage2_weights_sha="$(sha256sum "${stage2}/model.safetensors" | awk '{print $1}')"

python3 - "${model}/config.json" "${stage2}/config.json" <<'PY'
import json
import sys

configs = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        configs.append(json.load(handle))
for config in configs:
    if config.get("model_type") != "qwen3_tts":
        raise SystemExit("ERROR: expected model_type='qwen3_tts'")
source_kind = configs[0].get("tts_model_type")
stage2_kind = configs[1].get("tts_model_type")
if source_kind not in {"base", "custom_voice"}:
    raise SystemExit(f"ERROR: unsupported tts_model_type={source_kind!r}")
if stage2_kind != source_kind:
    raise SystemExit(
        f"ERROR: stage-2 tts_model_type={stage2_kind!r} does not match {source_kind!r}"
    )
PY

tmp="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-v010-int4.XXXXXX")"
cleanup() {
  case "${tmp}" in
    "${TMPDIR:-/tmp}"/qwen3-tts-v010-int4.*) rm -rf -- "${tmp}" ;;
  esac
}
trap cleanup EXIT

(
  cd "${upstream}"
  PYTHONPATH="${upstream}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -m tensorrt_edgellm.scripts.export \
      "${stage2}" "${tmp}" \
      --components talker \
      --int4-gemm-plugin-version 1
)

for required in \
  model.onnx model.onnx.data config.json embedding.safetensors \
  text_embedding.safetensors text_projection.safetensors tokenizer.json \
  processed_chat_template.json; do
  [ -s "${tmp}/llm/${required}" ] \
    || fail "v0.10 INT4 Talker export did not produce llm/${required}"
done

python3 - "${tmp}/llm/model.onnx" <<'PY'
import sys
import onnx

graph = onnx.load(sys.argv[1], load_external_data=True)
onnx.checker.check_model(graph)
ops = [node.op_type for node in graph.graph.node]
v1 = ops.count("Int4GroupwiseGemmPlugin")
v2 = ops.count("Int4GroupwiseGemmPluginV2")
if v1 <= 0 or v2 != 0:
    raise SystemExit(f"ERROR: expected plugin-v1-only graph, got v1={v1} v2={v2}")
PY

mkdir -p "${output}"
cp -a "${tmp}/llm/." "${output}/"
printf 'sha256:%s\n' "${driver_sha}" > "${output}/DRIVER_REVISION"
{
  printf '# Qwen3-TTS INT4 v0.10 export provenance\n\n'
  printf 'driver_sha256: `%s`\n\n' "${driver_sha}"
  printf 'upstream_sha: `%s`\n\n' "${PIN}"
  printf 'model_revision: `%s`\n\n' "${model_revision}"
  printf 'model_config_sha256: `%s`\n\n' "${model_config_sha}"
  printf 'model_weights_sha256: `%s`\n\n' "${model_weights_sha}"
  printf 'stage2_revision: `%s`\n\n' "${stage2_revision}"
  printf 'stage2_config_sha256: `%s`\n\n' "${stage2_config_sha}"
  printf 'stage2_quant_config_sha256: `%s`\n\n' "${stage2_quant_config_sha}"
  printf 'stage2_weights_sha256: `%s`\n\n' "${stage2_weights_sha}"
  printf 'int4_gemm_plugin_version: `1`\n'
} > "${output}/PROVENANCE.md"
(
  cd "${output}"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
)

echo "Qwen3-TTS INT4 v0.10 Talker export passed: ${output}"
