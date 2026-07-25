#!/usr/bin/env bash
# Negative provenance gates plus a real core.autocrlf=true clone check.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INNER_ROOT="$(cd "${HERE}/.." && pwd)"
PIN="$(grep -vE '^[[:space:]]*#' "${HERE}/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')"
OFFICIAL_CHECKOUT="${1:-}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/edgellm-provenance-negative.XXXXXX")"
case "${TMP_ROOT}" in
  "${TMPDIR:-/tmp}"/edgellm-provenance-negative.*) ;;
  *) echo "ERROR: unsafe temporary path ${TMP_ROOT}" >&2; exit 1 ;;
esac
cleanup() {
  rm -rf -- "${TMP_ROOT}"
}
trap cleanup EXIT

check_sums() {
  local directory="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "${directory}" && sha256sum -c SHA256SUMS >/dev/null)
  else
    (cd "${directory}" && shasum -a 256 -c SHA256SUMS >/dev/null)
  fi
}

expect_fail() {
  local label="$1"
  shift
  if "$@" >"${TMP_ROOT}/${label}.log" 2>&1; then
    echo "ERROR: negative test unexpectedly passed: ${label}" >&2
    exit 1
  fi
  echo "negative PASS: ${label}"
}

# Must fail before clone/fetch/build.
expect_fail missing_manifest \
  bash "${HERE}/build.sh" "${TMP_ROOT}/does-not-exist.toml"

cp "${HERE}/manifests/qwen3-asr-sm87.toml" "${TMP_ROOT}/stale-hash.toml"
python3 - "${TMP_ROOT}/stale-hash.toml" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
text, count = re.subn(
    r'series_sha256 = "[0-9a-f]+"',
    'series_sha256 = "' + "0" * 64 + '"',
    text,
    count=1,
)
assert count == 1
path.write_text(text)
PY
expect_fail stale_manifest_hash \
  bash "${HERE}/build.sh" "${TMP_ROOT}/stale-hash.toml"

cp "${HERE}/manifests/qwen3-asr-sm87.toml" "${TMP_ROOT}/stale-entry.toml"
python3 - "${TMP_ROOT}/stale-entry.toml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = 'checksums = "patches/upstream-v091-prs/SHA256SUMS"'
assert text.count(old) == 1
path.write_text(text.replace(
    old, 'checksums = "patches/upstream-v091-prs/MISSING"'))
PY
expect_fail stale_manifest_entry \
  bash "${HERE}/build.sh" "${TMP_ROOT}/stale-entry.toml"
for log in missing_manifest stale_manifest_hash stale_manifest_entry; do
  if grep -q 'cloning upstream' "${TMP_ROOT}/${log}.log"; then
    echo "ERROR: ${log} reached clone before provenance rejection" >&2
    exit 1
  fi
done
echo "negative PASS: manifest failures rejected before clone"

cp -R "${HERE}" "${TMP_ROOT}/lock-order-overlay"
python3 - "${TMP_ROOT}/lock-order-overlay/patches/upstream-v091-prs/LOCK" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text().splitlines()
records = [i for i, line in enumerate(lines) if line and not line.startswith("#")]
lines[records[0]], lines[records[1]] = lines[records[1]], lines[records[0]]
path.write_text("\n".join(lines) + "\n")
PY
expect_fail lock_order_set \
  bash "${TMP_ROOT}/lock-order-overlay/tests/verify-patch-stack.sh"

cp -R "${HERE}" "${TMP_ROOT}/sums-order-overlay"
python3 - "${TMP_ROOT}/sums-order-overlay/patches/upstream-v091-prs/SHA256SUMS" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text().splitlines()
lines[0], lines[1] = lines[1], lines[0]
path.write_text("\n".join(lines) + "\n")
PY
expect_fail sha_order_set \
  bash "${TMP_ROOT}/sums-order-overlay/tests/verify-patch-stack.sh"

mkdir -p "${TMP_ROOT}/not-a-repository"
expect_fail non_git_replay_source \
  bash "${HERE}/tests/verify-patch-stack.sh" \
  "${TMP_ROOT}/not-a-repository"

MISSING_PIN="${TMP_ROOT}/missing-pin-repository"
mkdir -p "${MISSING_PIN}"
git -C "${MISSING_PIN}" init -q
expect_fail missing_target_pin \
  bash "${HERE}/tests/verify-patch-stack.sh" "${MISSING_PIN}"

if [ "${SKIP_AUTOCLONE:-0}" = "1" ]; then
  echo "autocrlf=true clone: SKIP"
  exit 0
fi

AUTOCLONE="${TMP_ROOT}/autocrlf-clone"
git -c core.autocrlf=true clone --no-local "${INNER_ROOT}" "${AUTOCLONE}" >/dev/null
for relative in \
  engine-overlay/patches/upstream-v091-prs/series \
  engine-overlay/patches/upstream-v091-prs/LOCK \
  engine-overlay/patches/upstream-v091-prs/SHA256SUMS \
  engine-overlay/patches/v091-candidate/series \
  engine-overlay/patches/v091-candidate/SHA256SUMS; do
  cmp "${INNER_ROOT}/${relative}" "${AUTOCLONE}/${relative}"
done
while IFS= read -r file; do
  cmp "${INNER_ROOT}/engine-overlay/patches/upstream-v091-prs/${file}" \
    "${AUTOCLONE}/engine-overlay/patches/upstream-v091-prs/${file}"
done < "${HERE}/patches/upstream-v091-prs/series"
while IFS= read -r file; do
  cmp "${INNER_ROOT}/engine-overlay/patches/v091-candidate/${file}" \
    "${AUTOCLONE}/engine-overlay/patches/v091-candidate/${file}"
done < "${HERE}/patches/v091-candidate/series"
check_sums "${AUTOCLONE}/engine-overlay/patches/upstream-v091-prs"
check_sums "${AUTOCLONE}/engine-overlay/patches/v091-candidate"

if [ -n "${OFFICIAL_CHECKOUT}" ]; then
  gitlink_count="$(git -C "${OFFICIAL_CHECKOUT}" ls-tree -r "${PIN}" \
    | awk '$1 == "160000" {count++} END {print count + 0}')"
  [ "${gitlink_count}" -eq 3 ] \
    || { echo "ERROR: expected three v0.9.1 gitlink fixtures, found ${gitlink_count}" >&2; exit 1; }
  echo "exact-tree fixture: ${gitlink_count} gitlinks"

  MISSING_OBJECT_CLONE="${TMP_ROOT}/missing-object-clone"
  git clone --no-local "${OFFICIAL_CHECKOUT}" \
    "${MISSING_OBJECT_CLONE}" >/dev/null
  expect_fail missing_locked_object \
    bash "${HERE}/tests/verify-patch-stack.sh" \
    "${MISSING_OBJECT_CLONE}"

  OFFICIAL_CLONE="${TMP_ROOT}/official-clone"
  git clone --no-local "${OFFICIAL_CHECKOUT}" "${OFFICIAL_CLONE}" >/dev/null
  while IFS='|' read -r file pr commit parent tree patch_id expected_sha; do
    case "${file}" in ""|\#*) continue ;; esac
    git -C "${OFFICIAL_CLONE}" fetch "${OFFICIAL_CHECKOUT}" "${commit}" \
      >/dev/null
  done < "${HERE}/patches/upstream-v091-prs/LOCK"
  bash "${HERE}/tests/verify-patch-stack.sh" "${OFFICIAL_CLONE}" >/dev/null
  echo "ordinary Git clone replay source: PASS"

  python3 - "${OFFICIAL_CLONE}/.git/refs/heads/unrelated-broken" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("0" * 40 + "\n")
PY
  bash "${HERE}/tests/verify-patch-stack.sh" \
    "${OFFICIAL_CLONE}" >/dev/null
  echo "unrelated broken ref replay source: PASS"

  OFFICIAL_WORKTREE="${TMP_ROOT}/official-worktree"
  git -C "${OFFICIAL_CHECKOUT}" worktree add --detach \
    "${OFFICIAL_WORKTREE}" "${PIN}" >/dev/null
  [ -f "${OFFICIAL_WORKTREE}/.git" ] \
    || { echo "ERROR: regression fixture is not a standard linked worktree" >&2; exit 1; }
  bash "${HERE}/tests/verify-patch-stack.sh" "${OFFICIAL_WORKTREE}" >/dev/null
  git -C "${OFFICIAL_CHECKOUT}" worktree remove "${OFFICIAL_WORKTREE}"
  echo "linked Git worktree replay source: PASS"

  bash "${AUTOCLONE}/engine-overlay/tests/verify-patch-stack.sh" \
    "${OFFICIAL_CHECKOUT}" >/dev/null
else
  bash "${AUTOCLONE}/engine-overlay/tests/verify-patch-stack.sh" >/dev/null
fi
echo "autocrlf=true clone: locked bytes and integrity PASS"
