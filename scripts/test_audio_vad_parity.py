#!/usr/bin/env python3
"""Parity test: AudioVadSplitter (C++) vs split_audio_into_chunks (Python).

Runs the C++ test binary on a WAV (produces a JSON dump of boundaries),
then runs the reference Python implementation (verbatim from
Qwen3-ASR/qwen_asr/inference/utils.py:246) on the same WAV, and compares
boundary positions within ±1 sample tolerance.

The reference is inlined so this script works on any machine with numpy
and scipy.io.wavfile (i.e. no qwen_asr package required).

Usage:
    python scripts/test_audio_vad_parity.py \
        --wav docs/audio-evidence/zh-long-04-2026-05-13.wav \
        --binary native/edgellm_voice_worker/build/test_audio_vad_split \
        [--max-chunk-sec 13.0] [--search-expand-sec 2.0]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np


MIN_ASR_INPUT_SECONDS = 0.5  # mirror Qwen3-ASR constant


def split_audio_into_chunks_reference(
    wav: np.ndarray,
    sr: int,
    max_chunk_sec: float,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> List[Tuple[np.ndarray, float]]:
    """Verbatim port of qwen_asr/inference/utils.py:246 (numpy reference)."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)

    total_len = int(wav.shape[0])
    total_sec = total_len / float(sr)
    if total_sec <= max_chunk_sec:
        return [(wav, 0.0)]

    max_len = int(max_chunk_sec * sr)
    expand = int(search_expand_sec * sr)
    win = max(4, int((min_window_ms / 1000.0) * sr))

    chunks: List[Tuple[np.ndarray, float]] = []
    start = 0
    offset_sec = 0.0

    while (total_len - start) > max_len:
        cut = start + max_len
        left = max(start, cut - expand)
        right = min(total_len, cut + expand)

        if right - left <= win:
            boundary = cut
        else:
            seg = wav[left:right]
            seg_abs = np.abs(seg)
            window_sums = np.convolve(
                seg_abs, np.ones(win, dtype=np.float32), mode="valid"
            )
            min_pos = int(np.argmin(window_sums))
            wstart = min_pos
            wend = min_pos + win
            local = seg_abs[wstart:wend]
            inner = int(np.argmin(local))
            boundary = left + wstart + inner

        boundary = int(max(boundary, start + 1))
        boundary = int(min(boundary, total_len))

        chunk = wav[start:boundary]
        chunks.append((chunk, offset_sec))
        offset_sec += (boundary - start) / float(sr)
        start = boundary

    tail = wav[start:total_len]
    chunks.append((tail, offset_sec))

    min_len = int(MIN_ASR_INPUT_SECONDS * sr)
    padded: List[Tuple[np.ndarray, float]] = []
    for c, off in chunks:
        if c.shape[0] < min_len:
            pad = min_len - int(c.shape[0])
            c = np.pad(c, (0, pad), mode="constant", constant_values=0.0).astype(
                np.float32
            )
        padded.append((c, off))
    return padded


def reference_boundaries(wav: np.ndarray, sr: int, max_chunk_sec: float,
                          search_expand_sec: float) -> List[int]:
    """Return boundary sample indices including trailing total_len.

    Re-runs the boundary search standalone (not from chunk lengths, since
    chunks are post-pad), to mirror what AudioVadSplitter::boundaries emits.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)

    total_len = int(wav.shape[0])
    total_sec = total_len / float(sr)
    out: List[int] = []
    if total_sec <= max_chunk_sec:
        return [total_len]

    max_len = int(max_chunk_sec * sr)
    expand = int(search_expand_sec * sr)
    win = max(4, int((100.0 / 1000.0) * sr))
    start = 0
    while (total_len - start) > max_len:
        cut = start + max_len
        left = max(start, cut - expand)
        right = min(total_len, cut + expand)
        if right - left <= win:
            boundary = cut
        else:
            seg = wav[left:right]
            seg_abs = np.abs(seg)
            window_sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")
            min_pos = int(np.argmin(window_sums))
            local = seg_abs[min_pos: min_pos + win]
            inner = int(np.argmin(local))
            boundary = left + min_pos + inner
        boundary = int(max(boundary, start + 1))
        boundary = int(min(boundary, total_len))
        out.append(boundary)
        start = boundary
    out.append(total_len)
    return out


def load_wav_mono_f32(path: Path) -> Tuple[np.ndarray, int]:
    from scipy.io import wavfile

    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        wav = data.astype(np.float32) / 32768.0
    elif data.dtype == np.float32:
        wav = data
    elif data.dtype == np.int32:
        wav = data.astype(np.float32) / float(2**31)
    else:
        wav = data.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1).astype(np.float32)
    return wav, sr


def run_cpp(binary: Path, wav: Path, max_chunk_sec: float,
            search_expand_sec: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "vad.json"
        cmd = [
            str(binary), str(wav),
            f"{max_chunk_sec:.6f}",
            f"{search_expand_sec:.6f}",
            str(out_path),
        ]
        print(f"[cpp] $ {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"C++ binary failed: {proc.returncode}")
        return json.loads(out_path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, type=Path)
    ap.add_argument("--binary", required=True, type=Path,
                    help="Path to test_audio_vad_split executable")
    ap.add_argument("--max-chunk-sec", type=float, default=13.0)
    ap.add_argument("--search-expand-sec", type=float, default=2.0)
    ap.add_argument("--tolerance-samples", type=int, default=1)
    ap.add_argument("--save", type=Path, default=None,
                    help="Optional: write merged parity report JSON")
    args = ap.parse_args()

    if not args.wav.exists():
        print(f"WAV not found: {args.wav}", file=sys.stderr)
        return 2
    if not args.binary.exists():
        print(f"Binary not found: {args.binary}", file=sys.stderr)
        return 2

    wav, sr = load_wav_mono_f32(args.wav)
    print(f"Loaded {args.wav}: {len(wav)} samples, "
          f"{len(wav)/sr:.3f}s, sr={sr}")

    py_bnds = reference_boundaries(
        wav, sr, args.max_chunk_sec, args.search_expand_sec)
    print(f"[py ] boundaries (samples): {py_bnds}")
    print(f"[py ] boundaries (sec):     {[round(b/sr, 4) for b in py_bnds]}")

    cpp = run_cpp(args.binary, args.wav, args.max_chunk_sec,
                  args.search_expand_sec)
    cpp_bnds = list(cpp["boundaries_samples"])
    print(f"[cpp] boundaries (samples): {cpp_bnds}")
    print(f"[cpp] boundaries (sec):     {[round(b/sr, 4) for b in cpp_bnds]}")

    if len(py_bnds) != len(cpp_bnds):
        print(f"FAIL: count mismatch py={len(py_bnds)} cpp={len(cpp_bnds)}",
              file=sys.stderr)
        return 1

    max_diff = 0
    for i, (p, c) in enumerate(zip(py_bnds, cpp_bnds)):
        d = abs(int(p) - int(c))
        if d > max_diff:
            max_diff = d
        marker = "OK" if d <= args.tolerance_samples else "FAIL"
        print(f"  [{i}] py={p:>8d}  cpp={c:>8d}  diff={d}  {marker}")

    print(f"max boundary diff: {max_diff} samples (tolerance={args.tolerance_samples})")

    if args.save:
        args.save.write_text(json.dumps({
            "wav": str(args.wav),
            "sample_rate": sr,
            "max_chunk_sec": args.max_chunk_sec,
            "search_expand_sec": args.search_expand_sec,
            "python_boundaries": py_bnds,
            "cpp_boundaries": cpp_bnds,
            "max_diff_samples": max_diff,
            "tolerance_samples": args.tolerance_samples,
            "pass": max_diff <= args.tolerance_samples,
        }, indent=2))
        print(f"Saved {args.save}")

    if max_diff > args.tolerance_samples:
        print("FAIL: boundary positions exceed tolerance", file=sys.stderr)
        return 1
    print("PASS: boundaries match within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
