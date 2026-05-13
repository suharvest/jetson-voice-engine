#!/usr/bin/env python3
"""
Generate golden mel-spectrogram tensors from the upstream WhisperFeatureExtractor.

The C++ MelExtractor test
(`native/edgellm_voice_worker/tests/test_mel_extractor.cpp`) compares its
output element-wise against these tensors. Acceptance threshold: max abs diff
≤ 1e-3 (FP32 rounding tolerance between np.fft and KissFFT).

Each test case ships as two files in tests/golden_mels/:

  <name>.pcm.f32   raw float32 16kHz mono PCM (input).
  <name>.mel.f32   raw float32 mel spectrogram, row-major [n_mels, n_frames].
  <name>.json      metadata: pcm_md5, mel_shape, settings_used.

Run on a host with the qwen-asr Python venv (e.g. wsl2-local):
    fleet exec wsl2-local -- 'source /home/harve/qwen3-asr-vllm-env/bin/activate \\
        && python /tmp/generate_golden_mels.py --out /tmp/golden_mels'

Pull /tmp/golden_mels back into tests/golden_mels/ on the dev host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import WhisperFeatureExtractor


SAMPLE_RATE = 16000


def load_wav_resample_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        sw = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sw == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(round(len(audio) * SAMPLE_RATE / sr))
        src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=n, endpoint=False)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)
    return audio


def feat_extract(fe: WhisperFeatureExtractor, pcm: np.ndarray) -> np.ndarray:
    """Run WhisperFeatureExtractor's no-padding path on a raw PCM buffer.

    We bypass padding/truncation (padding=False) because we're testing the
    streaming path where the worker hands the model the actual mel for a short
    PCM buffer, not 30 s of right-padded zeros.
    """
    out = fe(pcm.astype(np.float32),
             sampling_rate=SAMPLE_RATE,
             padding=False,
             truncation=False,
             return_tensors="np")
    # input_features shape: (batch=1, n_mels, n_frames)
    mel = np.asarray(out["input_features"])[0]
    return mel.astype(np.float32)


def save_case(out_dir: Path, name: str, pcm: np.ndarray, fe: WhisperFeatureExtractor,
              settings: dict) -> dict:
    pcm_f32 = pcm.astype(np.float32)
    mel = feat_extract(fe, pcm_f32)
    pcm_path = out_dir / f"{name}.pcm.f32"
    mel_path = out_dir / f"{name}.mel.f32"
    pcm_f32.tofile(pcm_path)
    mel.tofile(mel_path)
    meta = {
        "name": name,
        "pcm_path": pcm_path.name,
        "pcm_samples": int(pcm_f32.size),
        "pcm_sr": SAMPLE_RATE,
        "pcm_md5": hashlib.md5(pcm_f32.tobytes()).hexdigest(),
        "pcm_duration_sec": float(pcm_f32.size) / SAMPLE_RATE,
        "mel_path": mel_path.name,
        "mel_shape": list(mel.shape),  # [n_mels, n_frames]
        "mel_md5": hashlib.md5(mel.tobytes()).hexdigest(),
        "mel_min": float(mel.min()),
        "mel_max": float(mel.max()),
        "settings": settings,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"  {name:30s} pcm={pcm_f32.size:7d} ({float(pcm_f32.size)/SAMPLE_RATE:.2f}s)  "
          f"mel={tuple(mel.shape)} range=[{mel.min():.4f},{mel.max():.4f}]")
    return meta


def build_cases(out_dir: Path, wav_dir: Optional[Path]) -> list[dict]:
    fe = WhisperFeatureExtractor.from_pretrained("Qwen/Qwen3-ASR-0.6B")
    settings = {
        "n_fft": fe.n_fft, "hop_length": fe.hop_length,
        "n_mels": fe.feature_size, "sampling_rate": fe.sampling_rate,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    rng = np.random.default_rng(1234)

    def add(name: str, pcm: np.ndarray):
        cases.append(save_case(out_dir, name, pcm, fe, settings))

    # Synthetic test cases —
    # 1) silence at various durations
    for dur in [0.5, 1.0, 2.0, 5.0]:
        add(f"silence_{int(dur*1000)}ms", np.zeros(int(dur * SAMPLE_RATE), dtype=np.float32))

    # 2) sine waves
    for freq in [220.0, 440.0, 1000.0, 4000.0]:
        t = np.arange(int(1.0 * SAMPLE_RATE)) / SAMPLE_RATE
        pcm = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        add(f"sine_{int(freq)}hz_1s", pcm)

    # 3) white noise
    for dur in [0.5, 1.0, 5.0]:
        pcm = (rng.standard_normal(int(dur * SAMPLE_RATE)) * 0.1).astype(np.float32)
        add(f"white_noise_{int(dur*1000)}ms", pcm)

    # 4) chirp (linear sweep 100 → 4000 Hz, 2s)
    t = np.arange(int(2.0 * SAMPLE_RATE)) / SAMPLE_RATE
    f0, f1 = 100.0, 4000.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * 2.0) * t * t)
    add("chirp_100_4000hz_2s", (0.5 * np.sin(phase)).astype(np.float32))

    # 5) edge cases: very short / odd-length / longish
    add("short_0.1s", (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(0.1 * SAMPLE_RATE)) / SAMPLE_RATE)).astype(np.float32))
    add("short_0.5s", (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(0.5 * SAMPLE_RATE)) / SAMPLE_RATE)).astype(np.float32))
    add("odd_length_16001",
        (0.3 * np.sin(2 * np.pi * 440 * np.arange(16001) / SAMPLE_RATE)).astype(np.float32))
    add("long_10s",
        (0.3 * np.sin(2 * np.pi * 440 * np.arange(10 * SAMPLE_RATE) / SAMPLE_RATE)).astype(np.float32))

    # 6) impulse + dc
    impulse = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    impulse[800] = 1.0
    add("impulse_0.5s", impulse)
    add("dc_0.5s", np.full(int(0.5 * SAMPLE_RATE), 0.1, dtype=np.float32))

    # 7) speech-like: chirp + noise + silence
    sp = np.concatenate([
        np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32),
        (0.3 * rng.standard_normal(int(0.5 * SAMPLE_RATE))).astype(np.float32),
        np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32),
        (0.4 * np.sin(2 * np.pi * 700 * np.arange(int(0.6 * SAMPLE_RATE)) / SAMPLE_RATE)).astype(np.float32),
    ])
    add("synth_speechlike_1.4s", sp)

    # 8) Real-speech WAVs from docs/audio-evidence (resampled to 16k mono).
    if wav_dir is not None and wav_dir.exists():
        wav_files = sorted(wav_dir.glob("*.wav"))
        # Pick a small representative subset to keep test corpus < 30 cases.
        chosen = [
            "voice-clone-reference-S1-2026-05-11.wav",
            "nx-highperf-2026-05-11.wav",
            "nx-voice-clone-2026-05-11.wav",
            "nx-clone-pass-p2-2026-05-11.wav",
            "nano-official-2026-05-11.wav",
        ]
        for fname in chosen:
            wav_path = wav_dir / fname
            if not wav_path.exists():
                continue
            pcm = load_wav_resample_mono(wav_path)
            add(f"real_{wav_path.stem.replace('-', '_')}", pcm)

    # Index file
    idx = {"settings": settings, "cases": [c["name"] for c in cases]}
    (out_dir / "index.json").write_text(json.dumps(idx, indent=2))
    return cases


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True,
                   help="output directory for *.pcm.f32 / *.mel.f32 / *.json triplets")
    p.add_argument("--wav-dir", type=Path, default=None,
                   help="optional dir of source WAVs (e.g. docs/audio-evidence/)")
    args = p.parse_args()
    cases = build_cases(args.out, args.wav_dir)
    print(f"\nGenerated {len(cases)} cases in {args.out}")


if __name__ == "__main__":
    main()
