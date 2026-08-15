#!/usr/bin/env python3
"""Fail-closed source-provenance validation for an overlay build manifest."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 on JetPack 6
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as error:
        raise SystemExit(
            "ERROR: manifest provenance: Python tomllib/tomli is required"
        ) from error


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: manifest provenance: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        fail(f"missing [{name}] table")
    return value


def value(entry: dict[str, Any], table_name: str, key: str, kind: type) -> Any:
    result = entry.get(key)
    if not isinstance(result, kind):
        fail(f"[{table_name}].{key} must be {kind.__name__}")
    return result


def locked_path(root: Path, raw: str, expected: str, field: str) -> Path:
    if raw != expected:
        fail(f"{field} must be {expected!r}, got {raw!r}")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        fail(f"{field} escapes overlay root")
    if not path.is_file():
        fail(f"{field} does not exist: {path}")
    return path


def series_entries(path: Path) -> list[str]:
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if Path(line).name != line:
            fail(f"{path}: non-basename series entry {line!r}")
        entries.append(line)
    if len(entries) != len(set(entries)):
        fail(f"{path}: duplicate series entries")
    return entries


def verify_hash(actual_path: Path, expected_hash: str, field: str) -> None:
    actual_hash = sha256(actual_path)
    if actual_hash != expected_hash:
        fail(f"{field} stale: expected {expected_hash}, got {actual_hash}")


def verify_release_target(data: dict[str, Any]) -> None:
    """Bind release manifests to the only qualified Jetson toolchain."""
    target = table(data, "target")
    expected: dict[str, Any] = {
        "device": "jetson-orin-nx",
        "sm": "87",
        "jetpack": "6.2",
        "l4t": "36.4.3",
        "cuda": "12.6",
        "tensorrt": "10.3",
        "embedded_target": "jetson-orin",
        "aarch64_build": True,
    }
    for key, wanted in expected.items():
        actual = target.get(key)
        if actual != wanted:
            fail(f"[target].{key} must be {wanted!r}, got {actual!r}")

    build = table(data, "build")
    if build.get("type") != "Release":
        fail(f"[build].type must be 'Release', got {build.get('type')!r}")


def verify_voice_clone_contract(data: dict[str, Any]) -> None:
    """Lock the Base persistent-voice ABI without changing CustomVoice."""
    model = data.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("base"), dict):
        return
    compatibility = table(data, "compatibility")
    voice = compatibility.get("voice_clone")
    if not isinstance(voice, dict):
        fail("[compatibility.voice_clone] must be a table")
    expected: dict[str, Any] = {
        "native_fields": ["ref_audio", "ref_text"],
        "legacy_field": "speaker_embedding_b64",
        "legacy_encoding": "base64(le-f32[1024])",
        "legacy_decoded_bytes": 4096,
        "legacy_requires_external_encoder": False,
        "legacy_mutually_exclusive_with": ["speaker", "speaker_id", "ref_audio", "ref_text"],
        "finite_values_required": True,
        "customvoice_unchanged": True,
    }
    for key, wanted in expected.items():
        if voice.get(key) != wanted:
            fail(f"[compatibility.voice_clone].{key} must be {wanted!r}, got {voice.get(key)!r}")


def main() -> None:
    if len(sys.argv) != 4:
        fail("usage: validate-manifest.py OVERLAY_ROOT MANIFEST UPSTREAM_PIN")
    root = Path(sys.argv[1]).resolve()
    manifest = Path(sys.argv[2]).resolve()
    pin = sys.argv[3]
    if not manifest.is_file():
        fail(f"manifest does not exist: {manifest}")

    try:
        with manifest.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        fail(f"cannot parse {manifest}: {error}")

    upstream = table(data, "upstream")
    if value(upstream, "upstream", "pin", str) != pin:
        fail("[upstream].pin does not match UPSTREAM_PIN")
    if upstream.get("version") != "v0.10.0":
        fail("[upstream].version must be 'v0.10.0'")
    if upstream.get("remote") != "https://github.com/NVIDIA/TensorRT-Edge-LLM.git":
        fail("[upstream].remote must be the NVIDIA repository")

    verify_release_target(data)
    verify_voice_clone_contract(data)

    proposed = table(data, "proposed_upstream_patches")
    if value(proposed, "proposed_upstream_patches", "directory",
             str) != "patches/upstream-v010-prs":
        fail("[proposed_upstream_patches].directory is not the locked directory")
    proposed_series = locked_path(
        root,
        value(proposed, "proposed_upstream_patches", "series", str),
        "patches/upstream-v010-prs/series",
        "[proposed_upstream_patches].series",
    )
    proposed_lock = locked_path(
        root,
        value(proposed, "proposed_upstream_patches", "lock", str),
        "patches/upstream-v010-prs/LOCK",
        "[proposed_upstream_patches].lock",
    )
    proposed_sums = locked_path(
        root,
        value(proposed, "proposed_upstream_patches", "checksums", str),
        "patches/upstream-v010-prs/SHA256SUMS",
        "[proposed_upstream_patches].checksums",
    )
    verify_hash(
        proposed_series,
        value(proposed, "proposed_upstream_patches", "series_sha256", str),
        "[proposed_upstream_patches].series_sha256",
    )
    verify_hash(
        proposed_lock,
        value(proposed, "proposed_upstream_patches", "lock_sha256", str),
        "[proposed_upstream_patches].lock_sha256",
    )
    verify_hash(
        proposed_sums,
        value(proposed, "proposed_upstream_patches", "checksums_sha256", str),
        "[proposed_upstream_patches].checksums_sha256",
    )
    proposed_entries = series_entries(proposed_series)
    proposed_count = value(proposed, "proposed_upstream_patches", "count",
                           int)
    if proposed_count != 4 or proposed_count != len(proposed_entries):
        fail("[proposed_upstream_patches].count must be 4 and match series")

    local = table(data, "patches")
    if value(local, "patches", "directory",
             str) != "patches/v010-candidate":
        fail("[patches].directory is not the locked directory")
    local_series = locked_path(
        root,
        value(local, "patches", "series", str),
        "patches/v010-candidate/series",
        "[patches].series",
    )
    local_sums = locked_path(
        root,
        value(local, "patches", "checksums", str),
        "patches/v010-candidate/SHA256SUMS",
        "[patches].checksums",
    )
    verify_hash(local_series, value(local, "patches", "series_sha256", str),
                "[patches].series_sha256")
    verify_hash(local_sums,
                value(local, "patches", "checksums_sha256", str),
                "[patches].checksums_sha256")
    local_entries = series_entries(local_series)
    local_count = value(local, "patches", "count", int)
    if local_count != 32 or local_count != len(local_entries):
        fail("[patches].count must be 32 and match series")
    if local.get("sparse_numbering") is not True:
        fail("[patches].sparse_numbering must be true")

    print(f"manifest provenance: PASS ({manifest})")


if __name__ == "__main__":
    main()
