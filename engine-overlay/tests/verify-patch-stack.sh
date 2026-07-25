#!/usr/bin/env bash
# Offline integrity/replay gate for the normalized v0.9.1 overlay.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="$(grep -vE '^[[:space:]]*#' "${HERE}/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')"
UPSTREAM_DIR="${HERE}/patches/upstream-v091-prs"
LOCAL_DIR="${HERE}/patches/v091-candidate"
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

read_series "${UPSTREAM_DIR}" "${UPSTREAM_DIR}/series" 7 "proposed-upstream"
UPSTREAM_SERIES=("${SERIES[@]}")
read_series "${LOCAL_DIR}" "${LOCAL_DIR}/series" 36 "local-product"
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
  case "${pr}" in 118|145|146|147|148|149) ;; *) die "${file}: unexpected PR ${pr}" ;; esac
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
[ "${lock_count}" -eq 7 ] || die "LOCK: expected 7 records, found ${lock_count}"

verify_sums "${UPSTREAM_DIR}" "${UPSTREAM_DIR}/SHA256SUMS" \
  "proposed-upstream" "${UPSTREAM_SERIES[@]}"
UPSTREAM_SUM_SHAS=("${SUM_SHAS[@]}")
for ((index=0; index < ${#UPSTREAM_SERIES[@]}; index++)); do
  [ "${UPSTREAM_SUM_SHAS[index]}" = "${LOCK_SHAS[index]}" ] \
    || die "proposed-upstream: LOCK and SHA256SUMS disagree for ${UPSTREAM_SERIES[index]}"
done
verify_sums "${LOCAL_DIR}" "${LOCAL_DIR}/SHA256SUMS" \
  "local-product" "${LOCAL_SERIES[@]}"

for retired in 0033 0034 0037 0038 0040; do
  if find "${LOCAL_DIR}" -maxdepth 1 -type f -name "${retired}-*.patch" | grep -q .; then
    die "retired local patch ${retired} is still present"
  fi
done

patch_0009="${LOCAL_DIR}/0009-feat-loader-load-weights-into-their-declared-half-dt.patch"
grep -q 'BF16Linear' "${patch_0009}" || die "0009 lost BF16Linear tied-weight behavior"
if grep -q '_set_tensor' "${patch_0009}"; then
  die "0009 still duplicates upstream checkpoint dtype behavior"
fi

patch_0039="${LOCAL_DIR}/0039-fix-cmake-propagate-CuTe-shim-driver-and-wrap-requir.patch"
grep -q 'PUBLIC "${CUDA_DRIVER_LIB}"' "${patch_0039}" \
  || die "0039 lost residual CUDA driver propagation"
if grep -q 'cudart_shim\|wrap=_cudaLaunchKernelEx' "${patch_0039}"; then
  die "0039 still duplicates PR #118 shim/wrap propagation"
fi

echo "integrity: 7 exact proposed-upstream + 36 sparse local patches PASS"

if [ -z "${REPLAY_SOURCE}" ]; then
  echo "replay: SKIP (pass a clean upstream checkout or set EDGELLM_UPSTREAM_CHECKOUT)"
  exit 0
fi

git -C "${REPLAY_SOURCE}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "not a Git checkout or worktree: ${REPLAY_SOURCE}"
git -C "${REPLAY_SOURCE}" cat-file -e "${PIN}^{commit}" \
  || die "upstream checkout lacks ${PIN}"
for ((index=0; index < ${#LOCK_COMMITS[@]}; index++)); do
  commit="${LOCK_COMMITS[index]}"
  git -C "${REPLAY_SOURCE}" cat-file -e "${commit}^{commit}" \
    || die "upstream checkout lacks locked commit ${commit}"
  actual_parent="$(git -C "${REPLAY_SOURCE}" show -s --format='%P' "${commit}")"
  actual_tree="$(git -C "${REPLAY_SOURCE}" show -s --format='%T' "${commit}")"
  [ "${actual_parent}" = "${LOCK_PARENTS[index]}" ] \
    || die "${commit}: actual parent differs from LOCK"
  [ "${actual_tree}" = "${LOCK_TREES[index]}" ] \
    || die "${commit}: actual tree differs from LOCK"
  object_patch_id="$(git -C "${REPLAY_SOURCE}" show --format= "${commit}" | git patch-id --stable | awk '{print $1}')"
  [ "${object_patch_id}" = "${LOCK_PATCH_IDS[index]}" ] \
    || die "${commit}: object patch-id differs from LOCK"
done
echo "official objects: 7/7 commit parent/tree/patch-id PASS"

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
# Consume only the exact locked base tree. Cloning/fetching the source would
# enumerate every source ref and can fail on an unrelated broken/partial ref
# even when PIN and all seven locked commits are present.
git -C "${REPLAY_SOURCE}" archive "${PIN}" | tar -x -C "${REPLAY}"
git -C "${REPLAY}" add -A
git -C "${REPLAY}" -c user.name=overlay-replay \
  -c user.email=overlay-replay@invalid commit -q -m "exact v0.9.1 replay base"

for file in "${UPSTREAM_SERIES[@]}"; do
  git -C "${REPLAY}" apply --check "${UPSTREAM_DIR}/${file}"
  git -C "${REPLAY}" apply "${UPSTREAM_DIR}/${file}"
done

(cd "${HERE}/addon" && find . -type f -print0 | while IFS= read -r -d '' file; do
  destination="${REPLAY}/${file#./}"
  mkdir -p "$(dirname "${destination}")"
  cp -p "${file}" "${destination}"
done)

for file in "${LOCAL_SERIES[@]}"; do
  git -C "${REPLAY}" apply --check "${LOCAL_DIR}/${file}"
  git -C "${REPLAY}" apply "${LOCAL_DIR}/${file}"
done
git -C "${REPLAY}" diff --check
echo "forward replay: 7/7 + 36/36 PASS"

for ((index=${#LOCAL_SERIES[@]} - 1; index >= 0; index--)); do
  file="${LOCAL_SERIES[index]}"
  git -C "${REPLAY}" apply --reverse --check "${LOCAL_DIR}/${file}"
  git -C "${REPLAY}" apply --reverse "${LOCAL_DIR}/${file}"
done
for ((index=${#UPSTREAM_SERIES[@]} - 1; index >= 0; index--)); do
  file="${UPSTREAM_SERIES[index]}"
  git -C "${REPLAY}" apply --reverse --check "${UPSTREAM_DIR}/${file}"
  git -C "${REPLAY}" apply --reverse "${UPSTREAM_DIR}/${file}"
done

git -C "${REPLAY}" diff --quiet || die "tracked tree differs after reverse replay"
expected_addon="$(mktemp)"
actual_addon="$(mktemp)"
(cd "${HERE}/addon" && find . -type f | sed 's#^./##' | sort) > "${expected_addon}"
git -C "${REPLAY}" ls-files --others --exclude-standard | sort > "${actual_addon}"
diff -u "${expected_addon}" "${actual_addon}" \
  || die "post-reverse untracked tree is not addon-only"
rm -f "${expected_addon}" "${actual_addon}"

echo "reverse replay: 36/36 + 7/7; official tracked tree + addon-only PASS"
