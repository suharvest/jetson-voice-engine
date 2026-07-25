#!/usr/bin/env bash
# Export the Seeed Qwen3-TTS Base extension with TensorRT Edge-LLM v0.9.1.
#
# Required environment:
#   TTS_BASE_UPSTREAM       patched v0.9.1 checkout
#   TTS_BASE_MODEL          pinned Qwen3-TTS Base HF snapshot
#   TTS_BASE_MODEL_REVISION immutable HF revision
#   TTS_BASE_OUTPUT         fresh output directory
#   TTS_BASE_PRECISION      fp16 or int4
#
# INT4 additionally requires a reviewed stage-2 checkpoint:
#   TTS_BASE_STAGE2_CHECKPOINT
#   TTS_BASE_STAGE2_REVISION

set -euo pipefail

PIN="7f061f21f0a581ba234a1e233c9315b89d8e47d6"
upstream="${TTS_BASE_UPSTREAM:?TTS_BASE_UPSTREAM is required}"
model="${TTS_BASE_MODEL:?TTS_BASE_MODEL is required}"
model_revision="${TTS_BASE_MODEL_REVISION:?TTS_BASE_MODEL_REVISION is required}"
output="${TTS_BASE_OUTPUT:?TTS_BASE_OUTPUT is required}"
precision="${TTS_BASE_PRECISION:-int4}"
driver_sha="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"

fail() {
  echo "ERROR: $*" >&2
  exit 12
}

run_export() {
  (
    cd "${upstream}"
    PYTHONPATH="${upstream}${PYTHONPATH:+:${PYTHONPATH}}" \
      python3 -m tensorrt_edgellm.scripts.export "$@"
  )
}

[ -e "${upstream}/.git" ] || fail "upstream is not a Git checkout: ${upstream}"
[ -s "${model}/config.json" ] || fail "Base checkpoint has no config.json: ${model}"
[ ! -e "${output}" ] || fail "output must not already exist: ${output}"

actual_pin="$(git -C "${upstream}" rev-parse HEAD)"
[ "${actual_pin}" = "${PIN}" ] \
  || fail "expected official v0.9.1 ${PIN}, got ${actual_pin}"

python3 - "${model}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
if config.get("model_type") != "qwen3_tts":
    raise SystemExit(
        f"ERROR: expected model_type='qwen3_tts', got {config.get('model_type')!r}"
    )
if config.get("tts_model_type") != "base":
    raise SystemExit(
        f"ERROR: expected tts_model_type='base', got {config.get('tts_model_type')!r}"
    )
PY

# The extension patch must have replaced the official hard error. Merely
# pointing this driver at an untouched checkout would otherwise produce a
# misleading "official Base export" claim.
if grep -Fq "Only Qwen3-TTS CustomVoice checkpoints are supported" \
    "${upstream}/tensorrt_edgellm/scripts/export.py"; then
  fail "Base exporter extension is not applied to the v0.9.1 checkout"
fi

mkdir -p "${output}"

case "${precision}" in
  fp16)
    run_export "${model}" "${output}" \
      --components talker,code_predictor,code2wav
    stage2_revision="not-applicable"
    ;;
  int4)
    stage2="${TTS_BASE_STAGE2_CHECKPOINT:?TTS_BASE_STAGE2_CHECKPOINT is required for INT4}"
    stage2_revision="${TTS_BASE_STAGE2_REVISION:?TTS_BASE_STAGE2_REVISION is required for INT4}"
    [ -s "${stage2}/config.json" ] \
      || fail "INT4 stage-2 checkpoint has no config.json: ${stage2}"
    run_export "${stage2}" "${output}" --components talker
    run_export "${model}" "${output}" --components code_predictor,code2wav
    ;;
  *)
    fail "unsupported TTS_BASE_PRECISION=${precision}; expected fp16 or int4"
    ;;
esac

for required in \
  llm/model.onnx llm/config.json \
  code_predictor/model.onnx code_predictor/config.json \
  code2wav/model.onnx code2wav/config.json; do
  [ -s "${output}/${required}" ] || fail "export did not produce ${required}"
done

python3 - "${output}" <<'PY'
import json
import sys
from pathlib import Path

import onnx

root = Path(sys.argv[1])
for component in ("llm", "code_predictor", "code2wav"):
    model_path = root / component / "model.onnx"
    graph = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(graph)
    with (root / component / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if component == "llm" and config.get("tts_model_type") != "base":
        raise SystemExit(
            "ERROR: exported Talker config did not preserve tts_model_type='base'"
        )
PY

printf 'sha256:%s\n' "${driver_sha}" > "${output}/DRIVER_REVISION"
{
  printf '# Qwen3-TTS Base v0.9.1 export provenance\n\n'
  printf 'driver_sha256: `%s`\n\n' "${driver_sha}"
  printf 'upstream_sha: `%s`\n\n' "${PIN}"
  printf 'model_revision: `%s`\n\n' "${model_revision}"
  printf 'precision: `%s`\n\n' "${precision}"
  printf 'stage2_revision: `%s`\n' "${stage2_revision}"
} > "${output}/PROVENANCE.md"

(
  cd "${output}"
  find llm code_predictor code2wav -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS
)

echo "Qwen3-TTS Base v0.9.1 export passed: ${output}"
