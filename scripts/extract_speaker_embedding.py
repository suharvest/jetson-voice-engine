#!/usr/bin/env python3
"""Extract 1024-d speaker embedding from a reference WAV via speaker_encoder.onnx.

Output: base64 of float32 little-endian bytes (1024 values × 4 = 4096 bytes).
Mel pipeline: 24 kHz mono → 128-bin log-mel (matches the encoder ONNX input).
"""
import sys, base64, wave, struct, numpy as np
import onnxruntime as ort

wav_path = sys.argv[1]
onnx_path = sys.argv[2]
out_b64_path = sys.argv[3]

# Load WAV (mono 24 kHz signed 16-bit PCM)
with wave.open(wav_path, "rb") as w:
    n = w.getnframes()
    sr = w.getframerate()
    nch = w.getnchannels()
    sw = w.getsampwidth()
    assert sw == 2 and nch == 1, f"need mono 16-bit, got nch={nch} sw={sw}"
    raw = w.readframes(n)
audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
if sr != 24000:
    # naive resample to 24 kHz
    import math
    new_len = int(round(len(audio) * 24000 / sr))
    audio = np.interp(np.linspace(0, len(audio)-1, new_len), np.arange(len(audio)), audio).astype(np.float32)
    sr = 24000

# Inspect ONNX input layout
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
outputs = sess.get_outputs()
print("inputs:", [(i.name, i.type, i.shape) for i in inputs])
print("outputs:", [(o.name, o.type, o.shape) for o in outputs])

# Compute log-mel (128 bins, hop 256, win 1024, 24 kHz)
N_FFT, HOP, WIN, N_MEL = 1024, 256, 1024, 128
def mel_filter_bank(sr, n_fft, n_mel):
    fmin, fmax = 0, sr/2
    def hz_to_mel(h): return 2595.0 * np.log10(1.0 + h/700.0)
    def mel_to_hz(m): return 700.0 * (10**(m/2595.0) - 1.0)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mel+2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft+1) * hz / sr).astype(int)
    fb = np.zeros((n_mel, n_fft//2 + 1), dtype=np.float32)
    for m in range(1, n_mel+1):
        lo, mi, hi = bins[m-1], bins[m], bins[m+1]
        for k in range(lo, mi):
            fb[m-1, k] = (k - lo) / max(mi - lo, 1)
        for k in range(mi, hi):
            fb[m-1, k] = (hi - k) / max(hi - mi, 1)
    return fb
fb = mel_filter_bank(sr, N_FFT, N_MEL)
win = np.hanning(WIN).astype(np.float32)
audio = np.pad(audio, N_FFT//2, mode="reflect")
frames = np.lib.stride_tricks.sliding_window_view(audio, WIN)[::HOP].copy() * win
S = np.fft.rfft(frames, n=N_FFT, axis=-1)
P = np.abs(S)**2
mel = P @ fb.T   # [T, 128]
logmel = np.log(np.maximum(mel, 1e-10)).astype(np.float32)
print("logmel shape:", logmel.shape)

mel_input = logmel[None, ...]  # [1, T, 128]
in_name = inputs[0].name
in_shape = inputs[0].shape
# Attempt straight pass-through; if encoder wants [B, F, T] (channels first), transpose.
out = sess.run(None, {in_name: mel_input})
emb = out[0].squeeze().astype(np.float32)
print(f"embedding shape={emb.shape} dtype={emb.dtype} L2={float(np.linalg.norm(emb)):.4f}")
assert emb.shape[-1] == 1024 or emb.shape == (1024,), f"unexpected emb shape: {emb.shape}"
emb_b64 = base64.b64encode(emb.tobytes()).decode("ascii")
with open(out_b64_path, "w") as f: f.write(emb_b64)
print(f"wrote {len(emb_b64)} chars of base64 to {out_b64_path} (raw={len(emb.tobytes())} bytes)")
