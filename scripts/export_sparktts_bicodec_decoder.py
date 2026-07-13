"""Export SparkTTS BiCodec decoder (detokenize path) to ONNX with DYNAMIC T.

Boundary (locked by spike, from sparktts/models/bicodec.py:detokenize):
    DecoderWrapper(semantic_tokens int64[B,T], d_vector f32[B,1024]) -> wav f32[B,1,L]
    chain = quantizer.detokenize -> prenet(feat_decoder) -> + d_vector -> wave_generator

Phase-1a upgrade over spike: validate dynamic T across multiple lengths (50/100/300)
with onnxruntime vs PyTorch, to ensure the engine is not frozen to a single T.

Run on WSL (x86 + CPU is fine for export determinism).
"""
import sys
import json
import hashlib

import numpy as np
import torch

MODEL_DIR = "/home/harve/project/v090-assets/spark-tts-0.5b"
REPO = "/home/harve/spike-sparktts/Spark-TTS"
DEV = "cpu"
ONNX_PATH = "bicodec_decoder_dynT.onnx"
OPSET = 17
EXPORT_T = 200  # opt-ish example length for tracing
VALIDATE_TS = [50, 100, 300]

sys.path.insert(0, REPO)
from sparktts.models.bicodec import BiCodec  # noqa: E402

torch.manual_seed(1234)
np.random.seed(1234)

print("Loading BiCodec...", flush=True)
model = BiCodec.load_from_checkpoint(f"{MODEL_DIR}/BiCodec")
model.eval().to(DEV)

codebook_size = model.quantizer.codebook.weight.shape[0]
try:
    n_codes_global = int(model.speaker_encoder.quantizer.codebook_size)
except Exception:
    n_codes_global = 4096
print("semantic codebook_size:", codebook_size, "global codebook_size:", n_codes_global, flush=True)


class DecoderWrapper(torch.nn.Module):
    """semantic_tokens (int) + d_vector (float) -> wav. Mirrors detokenize() minus global-FSQ."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, semantic_tokens, d_vector):
        z_q = self.m.quantizer.detokenize(semantic_tokens)
        x = self.m.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        return self.m.decoder(x)


wrapper = DecoderWrapper(model).eval()


def make_inputs(T):
    sem = torch.randint(0, codebook_size, (1, T), dtype=torch.long, device=DEV)
    dv = torch.randn(1, 1024, dtype=torch.float32, device=DEV)
    return sem, dv


# Trace example
sem_ex, dv_ex = make_inputs(EXPORT_T)
with torch.no_grad():
    wav_ex = wrapper(sem_ex, dv_ex)
print("trace example: T", EXPORT_T, "wav", tuple(wav_ex.shape), flush=True)

print("Exporting ONNX (dynamic T + dynamic L)...", flush=True)
torch.onnx.export(
    wrapper,
    (sem_ex, dv_ex),
    ONNX_PATH,
    input_names=["semantic_tokens", "d_vector"],
    output_names=["wav"],
    dynamic_axes={
        "semantic_tokens": {0: "B", 1: "T"},
        "d_vector": {0: "B"},
        "wav": {0: "B", 2: "L"},
    },
    opset_version=OPSET,
    do_constant_folding=True,
)
with open(ONNX_PATH, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
print("ONNX md5", md5, flush=True)

print("Validating dynamic T with onnxruntime vs PyTorch...", flush=True)
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
results = {}
for T in VALIDATE_TS:
    sem, dv = make_inputs(T)
    with torch.no_grad():
        wav_pt = wrapper(sem, dv).cpu().numpy().astype(np.float32)
    wav_ort = sess.run(
        None,
        {
            "semantic_tokens": sem.cpu().numpy().astype(np.int64),
            "d_vector": dv.cpu().numpy().astype(np.float32),
        },
    )[0].astype(np.float32)
    assert wav_ort.shape == wav_pt.shape, f"T={T} shape mismatch {wav_ort.shape} vs {wav_pt.shape}"
    max_abs = float(np.abs(wav_ort - wav_pt).max())
    rms = float(np.sqrt(np.mean(wav_ort**2)))
    results[T] = {"shape": list(wav_ort.shape), "max_abs": max_abs, "rms": rms}
    print(f"  T={T:4d}  out_shape={wav_ort.shape}  ORT_vs_PT_max_abs={max_abs:.3e}  rms={rms:.4f}", flush=True)

# sidecar config
cfg = {
    "model": "SparkTTS-0.5B BiCodec decoder (detokenize)",
    "inputs": {
        "semantic_tokens": "int64 [B,T]",
        "d_vector": "f32 [B,1024]",
    },
    "output": {"wav": "f32 [B,1,L] @16kHz, L = T*320"},
    "sample_rate": 16000,
    "upsample_per_token": 320,
    "semantic_codebook_size": int(codebook_size),
    "global_codebook_size": int(n_codes_global),
    "global_token_count": 32,
    "d_vector_dim": 1024,
    "opset": OPSET,
    "onnx_md5": md5,
    "dynamic_T_validation": results,
}
with open("bicodec_decoder_dynT.config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("Wrote sidecar bicodec_decoder_dynT.config.json", flush=True)
print("EXPORT_DYN_DONE", flush=True)
