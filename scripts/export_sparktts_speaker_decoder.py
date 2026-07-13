"""Export SparkTTS speaker_encoder.detokenize (global ids -> d_vector) to ONNX.

Boundary (from sparktts/modules/speaker/speaker_encoder.py:detokenize):
    detokenize(indices int64[B,1,32]) -> d_vector f32[B,1024]
    chain = quantizer.get_output_from_indices(indices.transpose(1,2)).transpose(1,2)
            -> reshape(B,-1) -> project (Linear 128*32=4096 -> 1024)

The FSQ gather (einx.get_at) inside ResidualFSQ.get_codes_from_indices is the part the
spike host_split flagged as "not ONNX-traceable". But the codebook is a FIXED implicit
buffer; we materialize it as a registered tensor and replace the einx gather with a plain
torch index_select / embedding gather (mathematically identical), which traces cleanly.

We then validate the exported ONNX vs the OFFICIAL detokenize (max_abs must be ~0) over
several random + the real mixed-engine global id sets.

Run on WSL (x86 CPU; export is deterministic).
"""
import sys
import json
import hashlib

import numpy as np
import torch

MODEL_DIR = "/home/harve/project/v090-assets/spark-tts-0.5b"
REPO = "/home/harve/spike-sparktts/Spark-TTS"
DEV = "cpu"
ONNX_PATH = "sparktts_speaker_decoder.onnx"
OPSET = 17

sys.path.insert(0, REPO)
from sparktts.models.bicodec import BiCodec  # noqa: E402

torch.manual_seed(1234)
np.random.seed(1234)

print("Loading BiCodec speaker_encoder...", flush=True)
model = BiCodec.load_from_checkpoint(f"{MODEL_DIR}/BiCodec")
model.eval().to(DEV)
spk = model.speaker_encoder
rfsq = spk.quantizer  # ResidualFSQ, num_quantizers=1, levels [4]*6 -> codebook_size 4096

# Materialize the fixed implicit codebook: [num_quant=1, codebook_size=4096, codebook_dim=6]
codebooks = rfsq.codebooks.detach().clone()  # [1, 4096, 6]
print("codebooks", tuple(codebooks.shape), codebooks.dtype, flush=True)
assert codebooks.shape[0] == 1, "expected single FSQ quantizer"
CB_SIZE = codebooks.shape[1]
CB_DIM = codebooks.shape[2]
codebook0 = codebooks[0]  # [4096, 6]


class SpeakerDecoderWrapper(torch.nn.Module):
    """global indices [B,1,32] -> d_vector [B,1024], traceable rewrite of detokenize()."""

    def __init__(self, m, codebook0):
        super().__init__()
        self.project_out = m.quantizer.project_out  # Linear(6 -> 128)
        self.project = m.project  # Linear(4096 -> 1024)
        self.register_buffer("codebook0", codebook0)  # [4096, 6]
        # scales for num_quantizers=1 -> (levels-1)**-0 == 1, so no scaling needed.

    def forward(self, indices):
        # indices: [B, 1, 32] long. Official: get_output_from_indices(indices.transpose(1,2))
        # transpose(1,2) -> [B, 32, 1] ; then internally packs to [B, 32, q=1] and gathers
        #   all_codes = get_at("q [c] d, b n q -> q b n d", codebooks, indices)
        # For q=1: gather per-token 6-dim code -> [B, 32, 6]; sum over q (only 1) ; project_out.
        B = indices.shape[0]
        idx = indices.reshape(B, -1)  # [B, 32]
        # gather codes: embedding lookup into codebook0 [4096,6] -> [B,32,6]
        codes = torch.nn.functional.embedding(idx, self.codebook0)  # [B,32,6]
        # scale = 1 (single quantizer), summed over q=1 is identity
        codes_summed = codes  # [B,32,6]
        zq = self.project_out(codes_summed)  # [B,32,128]
        # official get_output_from_indices returns [B, 32, 128]; detokenize does:
        #   .transpose(1,2) -> [B,128,32]  then reshape(B,-1) -> [B, 128*32]
        zq_cf = zq.transpose(1, 2)  # [B,128,32]
        x = zq_cf.reshape(B, -1)  # [B, 4096]
        d_vector = self.project(x)  # [B, 1024]
        return d_vector


wrapper = SpeakerDecoderWrapper(spk, codebook0).eval()


@torch.no_grad()
def official_detokenize(global_ids):
    g = torch.tensor(global_ids, dtype=torch.long, device=DEV).view(1, 1, -1)
    return spk.detokenize(g).cpu().numpy().astype(np.float32)


# Trace example
ex_ids = np.random.randint(0, CB_SIZE, size=32).tolist()
ex = torch.tensor(ex_ids, dtype=torch.long, device=DEV).view(1, 1, 32)
with torch.no_grad():
    dv_ex = wrapper(ex)
print("trace example d_vector", tuple(dv_ex.shape), flush=True)
assert tuple(dv_ex.shape) == (1, 1024)

print("Exporting ONNX...", flush=True)
torch.onnx.export(
    wrapper,
    (ex,),
    ONNX_PATH,
    input_names=["global_indices"],
    output_names=["d_vector"],
    dynamic_axes={"global_indices": {0: "B"}, "d_vector": {0: "B"}},
    opset_version=OPSET,
    do_constant_folding=True,
)
with open(ONNX_PATH, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
print("ONNX md5", md5, flush=True)

# ---- Validate ONNX vs official detokenize ----
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])


def onnx_dvec(global_ids):
    arr = np.array(global_ids, dtype=np.int64).reshape(1, 1, 32)
    return sess.run(None, {"global_indices": arr})[0].astype(np.float32)


results = []
# random sets
max_abs_all = 0.0
for i in range(5):
    ids = np.random.randint(0, CB_SIZE, size=32).tolist()
    ref = official_detokenize(ids)
    got = onnx_dvec(ids)
    ma = float(np.abs(got - ref).max())
    max_abs_all = max(max_abs_all, ma)
    results.append({"set": f"rand{i}", "max_abs": ma})
    print(f"  rand{i}  onnx_vs_official max_abs={ma:.3e}", flush=True)

# real mixed-engine global ids
try:
    gmap = json.load(open("mixed_globals.json"))
    for cid, ids in gmap.items():
        assert len(ids) == 32
        ref = official_detokenize(ids)
        got = onnx_dvec(ids)
        ma = float(np.abs(got - ref).max())
        max_abs_all = max(max_abs_all, ma)
        results.append({"set": cid, "max_abs": ma})
        print(f"  {cid:18s} onnx_vs_official max_abs={ma:.3e}", flush=True)
except FileNotFoundError:
    print("  (mixed_globals.json not found, skipping real-id check)", flush=True)

cfg = {
    "model": "SparkTTS-0.5B speaker_encoder.detokenize (global ids -> d_vector)",
    "inputs": {"global_indices": "int64 [B,1,32]"},
    "output": {"d_vector": "f32 [B,1024]"},
    "global_codebook_size": int(CB_SIZE),
    "codebook_dim": int(CB_DIM),
    "global_token_count": 32,
    "d_vector_dim": 1024,
    "opset": OPSET,
    "onnx_md5": md5,
    "onnx_vs_official_max_abs": max_abs_all,
    "validation": results,
}
with open("sparktts_speaker_decoder.config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("max_abs_all (onnx vs official):", max_abs_all, flush=True)
print("EXPORT_SPK_DECODER_DONE", flush=True)
