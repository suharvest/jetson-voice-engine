#!/usr/bin/env bash
# Fetch the exact SparkTTS inputs needed to rebuild the standalone BiCodec and
# speaker-decoder TensorRT engines. Model downloads are mirror-only and hf's
# local-dir cache is deliberately retained so interrupted downloads resume.
set -euo pipefail

SPARK_SOURCE_SHA=2f1ea9082400547242641f5271b6f941c9f439d1
SPARK_MODEL_REVISION=642071559bfc6346c2359d19dcb6be3f9dd8a05d
SPARK_SOURCE_URL=https://github.com/SparkAudio/Spark-TTS.git
SPARK_MODEL_ID=SparkAudio/Spark-TTS-0.5B

input_root="${SPARKTTS_INPUT_ROOT:?set SPARKTTS_INPUT_ROOT to a new or existing input directory}"
source_dir="${input_root}/Spark-TTS"
model_dir="${input_root}/Spark-TTS-0.5B"

if [ "${HF_ENDPOINT:-}" != "https://hf-mirror.com" ]; then
  echo "ERROR: HF_ENDPOINT must be https://hf-mirror.com" >&2
  echo "       run: fleet bootstrap <device> --profile edge-mirror" >&2
  exit 4
fi
if [ "$(bash -c 'printf %s "${HF_ENDPOINT:-}"')" != "https://hf-mirror.com" ]; then
  echo "ERROR: a non-login child shell does not inherit the required HF mirror" >&2
  exit 4
fi
command -v hf >/dev/null || {
  echo "ERROR: hf CLI is required; provision it through the managed uv environment" >&2
  exit 4
}
command -v git >/dev/null || {
  echo "ERROR: git is required" >&2
  exit 4
}

mkdir -p "${input_root}"
source_created=0
if [ ! -d "${source_dir}/.git" ]; then
  if [ -e "${source_dir}" ]; then
    echo "ERROR: refusing to replace non-git path ${source_dir}" >&2
    exit 5
  fi
  git clone --filter=blob:none --no-checkout "${SPARK_SOURCE_URL}" "${source_dir}"
  source_created=1
fi

actual_source="$(git -C "${source_dir}" rev-parse HEAD 2>/dev/null || true)"
if [ "${actual_source}" != "${SPARK_SOURCE_SHA}" ]; then
  git -C "${source_dir}" fetch --filter=blob:none origin "${SPARK_SOURCE_SHA}"
  if [ "${source_created}" = 1 ]; then
    git -C "${source_dir}" checkout --detach "${SPARK_SOURCE_SHA}"
  fi
  actual_source="$(git -C "${source_dir}" rev-parse HEAD)"
fi
if [ "${actual_source}" != "${SPARK_SOURCE_SHA}" ]; then
  echo "ERROR: ${source_dir} is at ${actual_source}; expected ${SPARK_SOURCE_SHA}" >&2
  echo "       refusing to reset or overwrite an existing checkout" >&2
  exit 5
fi
if ! git -C "${source_dir}" diff --quiet || ! git -C "${source_dir}" diff --cached --quiet; then
  echo "ERROR: Spark-TTS source checkout is dirty" >&2
  exit 5
fi

HF_ENDPOINT=https://hf-mirror.com hf download \
  "${SPARK_MODEL_ID}" \
  BiCodec/config.yaml \
  BiCodec/model.safetensors \
  --revision "${SPARK_MODEL_REVISION}" \
  --local-dir "${model_dir}"

for required in \
  "${model_dir}/BiCodec/config.yaml" \
  "${model_dir}/BiCodec/model.safetensors"; do
  test -s "${required}" || {
    echo "ERROR: required checkpoint input is missing: ${required}" >&2
    exit 6
  }
done

provenance="${input_root}/INPUT_PROVENANCE.txt"
tmp_provenance="${provenance}.tmp.$$"
{
  printf 'spark_source_url: %s\n' "${SPARK_SOURCE_URL}"
  printf 'spark_source_sha: %s\n' "${SPARK_SOURCE_SHA}"
  printf 'spark_model_id: %s\n' "${SPARK_MODEL_ID}"
  printf 'spark_model_revision: %s\n' "${SPARK_MODEL_REVISION}"
  printf 'hf_endpoint: %s\n' "${HF_ENDPOINT}"
  sha256sum \
    "${model_dir}/BiCodec/config.yaml" \
    "${model_dir}/BiCodec/model.safetensors"
} >"${tmp_provenance}"
mv "${tmp_provenance}" "${provenance}"

printf 'SPARKTTS_REPO=%q\n' "${source_dir}"
printf 'SPARKTTS_MODEL_DIR=%q\n' "${model_dir}"
printf 'SPARKTTS_INPUT_PROVENANCE=%q\n' "${provenance}"
