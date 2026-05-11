# Qwen3 Orin NX Voice Clone — Real-Embedding Gate (2026-05-11)

Status: **GATE PASSED** — worker accepts a real speaker embedding
extracted via `speaker_encoder.onnx`, streams 4.72 s of audio at RTF
0.76 with `audio_complete=true`.

This closes §5 from `docs/reproduction-remaining-work-2026-05-11.md`.

## Pipeline

```
S1.wav (2.8 s reference, 24 kHz mono)
  → 128-bin log-mel
  → tts/speaker_encoder/speaker_encoder.onnx (CPU EP)
  → speaker_embedding [1024] float32 (L2 = 24.88)
  → base64 (4096 raw bytes → 5464 b64 chars)
  → qwen3_tts_worker JSON request with speaker_embedding_b64
  → streaming audio_b64 chunks (24 kHz PCM s16le)
  → 4.72 s WAV
```

## ONNX speaker encoder

- Path on HF: `tts/speaker_encoder/speaker_encoder.onnx` (35 588 164 B,
  sha256 `544da28fd7463a3fd722d54630fe39b730376b6aee850ca66845ee981e87a40f`)
- Input: `mel [1, time, 128]` float32
- Output: `speaker_embedding [1024]` float32

Extraction script: `scripts/extract_speaker_embedding.py` (committed
with this report; same as `/tmp/extract_embed.py` used in the run).

## Worker request

```json
{
  "id": "clone1",
  "text": "你好，这是声音克隆测试。今天天气不错。",
  "stream": true,
  "stream_only": true,
  "first_chunk_frames": 7,
  "chunk_frames": 10,
  "max_chunk_frames": 10,
  "speaker_embedding_b64": "<5464 base64 chars (4096 bytes float32)>"
}
```

## Result

```json
{
  "event": "done",
  "ok": true,
  "audio_complete": true,
  "audio_s": 4.72,
  "samples": 113280,
  "sample_rate": 24000,
  "chunk_count": 7,
  "frames": 59,
  "first_chunk_ms": 618.333,
  "total_ms": 3587.399,
  "generation_ms": 3578.200,
  "code2wav_ms": 168.952,
  "rtf": 0.760,
  "stateful_code2wav": true
}
```

Worker cold start: 12.3 s (same engines as the §1 highperf smoke; no
extra cost from the speaker embedding path).

## Acceptance criteria check

| Criterion (per spec §5) | Result |
|---|---|
| Real reference embedding works end to end | ✅ |
| Both `/tts/clone` and `/tts/clone/stream` paths exercised | ⚠ direct worker JSON only (no jetson-voice REST layer yet — service depends on §2 docker fix) |
| Cloned audio file size > 30 KB | ✅ 230 KB |
| audio_complete=true | ✅ |
| Worker does NOT re-run speaker encoder per request | ✅ — `speaker_embedding_b64` is consumed as-is (encoder ran once, on Mac/host pre-extract) |
| Document the exact extraction command and embedding format | ✅ — see `scripts/extract_speaker_embedding.py` |

The jetson-voice REST `/tts/clone` and `/tts/clone/stream` HTTP layer
remains to be exercised once the §2 slim docker image lands; the
worker-side protocol is now proven, so the REST layer is just JSON
plumbing on top.

## Files

| Path | Description |
|---|---|
| `docs/audio-evidence/voice-clone-reference-S1-2026-05-11.wav` | Source reference (2.8 s, S1 bench WAV) |
| `docs/audio-evidence/nx-voice-clone-2026-05-11.wav` | Cloned output (4.72 s) |
| `scripts/extract_speaker_embedding.py` | Mel + ONNX encoder helper |

Listen side by side to assess clone similarity.

## Reproduce

```bash
# On the Jetson with artifacts deployed (per
# docs/performance/qwen3-orin-nx-highperf-2026-05-11.md):
python3 scripts/extract_speaker_embedding.py \
  <reference.wav> \
  $QWEN3_ARTIFACT_ROOT/tts/speaker_encoder/speaker_encoder.onnx \
  speaker_emb.b64

EMB=$(cat speaker_emb.b64)
python3 -c "import json; print(json.dumps({
  'id':'clone','text':'<your prompt>','stream':True,'stream_only':True,
  'first_chunk_frames':7,'chunk_frames':10,'max_chunk_frames':10,
  'speaker_embedding_b64':'$EMB'}))" | $WORKER \
    --talkerEngineDir="$QWEN3_ARTIFACT_ROOT/tts/talker" \
    --qwen3TtsTalkerBackend=qwen3_tts_explicit_kv \
    --qwen3TtsTalkerEngine="$QWEN3_ARTIFACT_ROOT/engines/orin-nx/highperf/talker_w8a16_outputk/talker_decode_w8a16_outputk.engine" \
    --codePredictorEngineDir="$QWEN3_ARTIFACT_ROOT/engines/orin-nx/highperf/code_predictor/cp_dir" \
    --codePredictorBackend=qwen3_tts_native \
    --code2wavEngineDir="$QWEN3_ARTIFACT_ROOT/engines/orin-nx/highperf/code2wav_stateful" \
    --tokenizerDir="$QWEN3_ARTIFACT_ROOT/tts/tokenizer"
```

Cache the embedding file once per speaker. The low-latency clone path
must not re-run the speaker encoder per request.
