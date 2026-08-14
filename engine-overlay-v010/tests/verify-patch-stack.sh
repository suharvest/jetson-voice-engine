#!/usr/bin/env bash
# Offline integrity/replay gate for the TensorRT-Edge-LLM v0.10.0 overlay.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="$(grep -vE '^[[:space:]]*#' "${HERE}/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')"
UPSTREAM_DIR="${HERE}/patches/upstream-v010-prs"
LOCAL_DIR="${HERE}/patches/v010-candidate"
REPLAY_SOURCE="${1:-${EDGELLM_UPSTREAM_CHECKOUT:-}}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die "sha256sum or shasum is required"
  fi
}

read_series() {
  local dir="$1"
  local series_file="$2"
  local expected="$3"
  local label="$4"
  local line
  SERIES=()
  while IFS= read -r line || [ -n "${line}" ]; do
    case "${line}" in ""|\#*) continue ;; esac
    [[ "${line}" != */* ]] || die "${label}: non-basename series entry ${line}"
    [ -f "${dir}/${line}" ] || die "${label}: missing ${line}"
    SERIES+=("${line}")
  done < "${series_file}"
  [ "${#SERIES[@]}" -eq "${expected}" ] \
    || die "${label}: expected ${expected}, found ${#SERIES[@]}"

  local listed actual
  listed="$(mktemp)"
  actual="$(mktemp)"
  printf '%s\n' "${SERIES[@]}" | sort > "${listed}"
  find "${dir}" -maxdepth 1 -type f -name '*.patch' -exec basename {} \; | sort > "${actual}"
  diff -u "${actual}" "${listed}" \
    || die "${label}: directory differs from explicit series"
  rm -f "${listed}" "${actual}"
}

read_series "${UPSTREAM_DIR}" "${UPSTREAM_DIR}/series" 4 "proposed-upstream"
UPSTREAM_SERIES=("${SERIES[@]}")
[ -f "${LOCAL_DIR}/series" ] || die "local-product: missing required v0.10 series"
read_series "${LOCAL_DIR}" "${LOCAL_DIR}/series" 32 "local-product"
LOCAL_SERIES=("${SERIES[@]}")

verify_sums() {
  local dir="$1"
  local sums_file="$2"
  local label="$3"
  shift 3
  local expected_series=("$@")
  local count=0 expected_sha file extra
  SUM_SHAS=()
  while read -r expected_sha file extra; do
    case "${expected_sha}" in ""|\#*) continue ;; esac
    [ -z "${extra:-}" ] || die "${label}: malformed SHA256SUMS entry"
    [ "${count}" -lt "${#expected_series[@]}" ] \
      || die "${label}: SHA256SUMS has extra entry ${file}"
    [ "${file}" = "${expected_series[count]}" ] \
      || die "${label}: SHA256SUMS order/set differs from series at ${file}"
    [ -f "${dir}/${file}" ] || die "${label}: SHA256SUMS references missing ${file}"
    [ "$(sha256_file "${dir}/${file}")" = "${expected_sha}" ] \
      || die "${label}: ${file} SHA-256 mismatch"
    SUM_SHAS+=("${expected_sha}")
    count=$((count + 1))
  done < "${sums_file}"
  [ "${count}" -eq "${#expected_series[@]}" ] \
    || die "${label}: SHA256SUMS count differs from series"
}

lock_count=0
LOCK_FILES=()
LOCK_COMMITS=()
LOCK_PARENTS=()
LOCK_TREES=()
LOCK_PATCH_IDS=()
LOCK_SHAS=()
while IFS='|' read -r file pr commit parent tree patch_id expected_sha; do
  case "${file}" in ""|\#*) continue ;; esac
  [ "${lock_count}" -lt "${#UPSTREAM_SERIES[@]}" ] \
    || die "LOCK has extra entry ${file}"
  [ "${file}" = "${UPSTREAM_SERIES[lock_count]}" ] \
    || die "LOCK order/set differs from series at ${file}"
  [ -f "${UPSTREAM_DIR}/${file}" ] || die "LOCK references missing ${file}"
  actual_sha="$(sha256_file "${UPSTREAM_DIR}/${file}")"
  [ "${actual_sha}" = "${expected_sha}" ] || die "${file}: SHA-256 mismatch"
  actual_commit="$(sed -n '1s/^From \([0-9a-f]\{40\}\) .*/\1/p' "${UPSTREAM_DIR}/${file}")"
  [ "${actual_commit}" = "${commit}" ] || die "${file}: From commit mismatch"
  actual_patch_id="$(git patch-id --stable < "${UPSTREAM_DIR}/${file}" | awk '{print $1}')"
  [ "${actual_patch_id}" = "${patch_id}" ] || die "${file}: stable patch-id mismatch"
  case "${pr}" in 118|146|149) ;; *) die "${file}: unexpected PR ${pr}" ;; esac
  [ "${#parent}" -eq 40 ] && [ "${#tree}" -eq 40 ] \
    || die "${file}: malformed parent/tree provenance"
  LOCK_FILES+=("${file}")
  LOCK_COMMITS+=("${commit}")
  LOCK_PARENTS+=("${parent}")
  LOCK_TREES+=("${tree}")
  LOCK_PATCH_IDS+=("${patch_id}")
  LOCK_SHAS+=("${expected_sha}")
  lock_count=$((lock_count + 1))
done < "${UPSTREAM_DIR}/LOCK"
[ "${lock_count}" -eq 4 ] || die "LOCK: expected 4 records, found ${lock_count}"

verify_sums "${UPSTREAM_DIR}" "${UPSTREAM_DIR}/SHA256SUMS" \
  "proposed-upstream" "${UPSTREAM_SERIES[@]}"
UPSTREAM_SUM_SHAS=("${SUM_SHAS[@]}")
for ((index=0; index < ${#UPSTREAM_SERIES[@]}; index++)); do
  [ "${UPSTREAM_SUM_SHAS[index]}" = "${LOCK_SHAS[index]}" ] \
    || die "proposed-upstream: LOCK and SHA256SUMS disagree for ${UPSTREAM_SERIES[index]}"
done
verify_sums "${LOCAL_DIR}" "${LOCAL_DIR}/SHA256SUMS" \
  "local-product" "${LOCAL_SERIES[@]}"
echo "local-product: 32-entry v0.10 candidate present"

echo "integrity: 4 exact proposed-upstream + 32 local-product PASS"

if [ -z "${REPLAY_SOURCE}" ]; then
  echo "replay: SKIP (pass a clean upstream checkout or set EDGELLM_UPSTREAM_CHECKOUT)"
  exit 0
fi

git -C "${REPLAY_SOURCE}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "not a Git checkout or worktree: ${REPLAY_SOURCE}"
git -C "${REPLAY_SOURCE}" cat-file -e "${PIN}^{commit}" \
  || die "upstream checkout lacks ${PIN}"
echo "official objects: skipped for overlay-local rebases"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/edgellm-overlay-replay.XXXXXX")"
case "${TMP_ROOT}" in
  "${TMPDIR:-/tmp}"/edgellm-overlay-replay.*) ;;
  *) die "refusing unsafe temporary path ${TMP_ROOT}" ;;
esac
cleanup() {
  rm -rf -- "${TMP_ROOT}"
}
trap cleanup EXIT

REPLAY="${TMP_ROOT}/repo"
mkdir -p "${REPLAY}"
git -C "${REPLAY}" init -q
# Consume only the source's exact object store through a read-only alternate.
# This avoids enumerating unrelated refs while preserving the complete Git
# index/tree, including gitlinks, executable bits, and symlinks.
SOURCE_COMMON_DIR="$(git -C "${REPLAY_SOURCE}" rev-parse --git-common-dir)"
case "${SOURCE_COMMON_DIR}" in
  /*) ;;
  *) SOURCE_COMMON_DIR="$(cd "${REPLAY_SOURCE}/${SOURCE_COMMON_DIR}" && pwd -P)" ;;
esac
SOURCE_OBJECTS="${SOURCE_COMMON_DIR}/objects"
[ -d "${SOURCE_OBJECTS}" ] || die "source object directory is missing: ${SOURCE_OBJECTS}"
mkdir -p "${REPLAY}/.git/objects/info"
printf '%s\n' "${SOURCE_OBJECTS}" > "${REPLAY}/.git/objects/info/alternates"
git -C "${REPLAY}" read-tree "${PIN}"
git -C "${REPLAY}" checkout-index -a
EXPECTED_BASE_TREE="$(git -C "${REPLAY_SOURCE}" rev-parse "${PIN}^{tree}")"
ACTUAL_BASE_TREE="$(git -C "${REPLAY}" write-tree)"
[ "${ACTUAL_BASE_TREE}" = "${EXPECTED_BASE_TREE}" ] \
  || die "replay baseline tree differs from ${PIN}: ${ACTUAL_BASE_TREE}"
echo "exact replay baseline tree: ${ACTUAL_BASE_TREE}"

for file in "${UPSTREAM_SERIES[@]}"; do
  git -C "${REPLAY}" apply --check "${UPSTREAM_DIR}/${file}"
  git -C "${REPLAY}" apply "${UPSTREAM_DIR}/${file}"
done

(cd "${HERE}/addon" && find . -type f -print0 | while IFS= read -r -d '' file; do
  destination="${REPLAY}/${file#./}"
  mkdir -p "$(dirname "${destination}")"
  cp -p "${file}" "${destination}"
done)

if [ "${#LOCAL_SERIES[@]}" -gt 0 ]; then
  for file in "${LOCAL_SERIES[@]}"; do
    git -C "${REPLAY}" apply --check "${LOCAL_DIR}/${file}"
    git -C "${REPLAY}" apply "${LOCAL_DIR}/${file}"
  done
fi
git -C "${REPLAY}" diff --check
echo "forward replay: 4+32 PASS"

if [ "${#LOCAL_SERIES[@]}" -gt 0 ]; then
  for ((index=${#LOCAL_SERIES[@]} - 1; index >= 0; index--)); do
    file="${LOCAL_SERIES[index]}"
    git -C "${REPLAY}" apply --reverse --check "${LOCAL_DIR}/${file}"
    git -C "${REPLAY}" apply --reverse "${LOCAL_DIR}/${file}"
  done
fi
for ((index=${#UPSTREAM_SERIES[@]} - 1; index >= 0; index--)); do
  file="${UPSTREAM_SERIES[index]}"
  git -C "${REPLAY}" apply --reverse --check "${UPSTREAM_DIR}/${file}"
  git -C "${REPLAY}" apply --reverse "${UPSTREAM_DIR}/${file}"
done

git -C "${REPLAY}" diff --quiet || die "tracked worktree differs after reverse replay"
REVERSED_BASE_TREE="$(git -C "${REPLAY}" write-tree)"
[ "${REVERSED_BASE_TREE}" = "${EXPECTED_BASE_TREE}" ] \
  || die "replay index tree differs from ${PIN} after reverse replay"
echo "reverse replay: 4/4; official tracked tree PASS"
