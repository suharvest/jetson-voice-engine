#!/usr/bin/env python3
"""Extract 1024-d speaker embedding from a reference WAV via speaker_encoder.onnx.

Matches the official Qwen3-TTS extract_speaker_embedding mel pipeline:
  - librosa.filters.mel (slaney norm)
  - torch-style STFT with center=False, pad_mode='reflect'
  - magnitude = sqrt(real^2 + imag^2 + 1e-9)
  - dynamic_range_compression: log(clip(x, min=1e-5))
  - hop=256, win=1024, n_fft=1024, n_mels=128, fmin=0, fmax=12000, sr=24000
"""
import sys, base64, wave, numpy as np
import librosa
import onnxruntime as ort

wav_path = sys.argv[1]
onnx_path = sys.argv[2]
out_b64_path = sys.argv[3]

with wave.open(wav_path, "rb") as w:
    n = w.getnframes(); sr_orig = w.getframerate(); nch = w.getnchannels(); sw = w.getsampwidth()
    assert sw == 2 and nch == 1, f"need mono 16-bit, got nch={nch} sw={sw}"
    raw = w.readframes(n)
audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
if sr_orig != 24000:
    audio = librosa.resample(audio.astype(np.float32), orig_sr=sr_orig, target_sr=24000)
sr = 24000
print(f"audio: sr={sr} samples={len(audio)} dur={len(audio)/sr:.2f}s min={audio.min():.3f} max={audio.max():.3f}")

# Match official mel_spectrogram() exactly
N_FFT, HOP, WIN, N_MEL = 1024, 256, 1024, 128
FMIN, FMAX = 0, 12000

# librosa slaney-norm mel filter (matches official librosa_mel_fn call)
mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MEL, fmin=FMIN, fmax=FMAX, htk=False, norm='slaney')  # [n_mels, n_fft/2+1]
hann = np.hanning(WIN).astype(np.float32)

# Reflective pad matching torch: (n_fft - hop_size) // 2
pad = (N_FFT - HOP) // 2
y = np.pad(audio.astype(np.float32), pad, mode='reflect')

# STFT center=False — frames are NOT padded, take raw windows
num_frames = 1 + (len(y) - WIN) // HOP
frames = np.lib.stride_tricks.sliding_window_view(y, WIN)[::HOP][:num_frames] * hann
S = np.fft.rfft(frames, n=N_FFT, axis=-1)
# magnitude (NOT power)
mag = np.sqrt(np.abs(S.real)**2 + np.abs(S.imag)**2 + 1e-9).astype(np.float32)  # [T, n_fft/2+1]
# mel projection
mel_spec = mag @ mel_basis.T  # [T, n_mels]
# dynamic_range_compression: log(clip(x, min=1e-5))
mel_spec = np.log(np.clip(mel_spec, 1e-5, None)).astype(np.float32)
print(f"mel: shape={mel_spec.shape} min={mel_spec.min():.3f} max={mel_spec.max():.3f} mean={mel_spec.mean():.3f}")

# Note: official does mel_spectrogram(...).transpose(1, 2) — so output is [B, T, n_mels]
# Our mel_spec is already [T, n_mels]. ONNX expects [1, T, 128].
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
print("onnx inputs:", [(i.name, i.shape) for i in inputs])
mel_input = mel_spec[None, ...]  # [1, T, 128]
out = sess.run(None, {inputs[0].name: mel_input})
emb = out[0].squeeze().astype(np.float32)
print(f"embedding: shape={emb.shape} L2={float(np.linalg.norm(emb)):.4f} min={emb.min():.4f} max={emb.max():.4f}")
assert emb.shape == (1024,), f"unexpected: {emb.shape}"
emb_b64 = base64.b64encode(emb.tobytes()).decode("ascii")
with open(out_b64_path, "w") as f: f.write(emb_b64)
print(f"wrote {len(emb_b64)} chars to {out_b64_path}")
