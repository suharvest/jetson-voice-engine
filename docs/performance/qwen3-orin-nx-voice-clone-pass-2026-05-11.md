# Qwen3 highperf Voice Clone — exact-match 3/3 (2026-05-11)

Status: **RESOLVED**. Zero-shot voice clone via raw speaker embedding
now produces content-correct audio matching the input prompt.

## Result

Reference audio: `bench/wavs/S1.wav` (real human Chinese speech, 2.8 s,
24 kHz mono).

Speaker embedding extracted via `speaker_encoder.onnx` using the
official Qwen3-TTS mel pipeline → 1024-d float32, L2 = 10.57.

| Prompt | Clone WAV → ASR | Match |
|---|---|---|
| 今天天气真好。 | 今天天气真好。 | ✅ |
| 人工智能改变了世界。 | 人工智能改变了世界。 | ✅ |
| 一二三四五六七八九十。 | 一二三四五六七八九十。 | ✅ |

Audio evidence:
- `docs/audio-evidence/voice-clone-reference-S1-2026-05-11.wav`
  (the reference voice)
- `docs/audio-evidence/nx-clone-pass-p1-2026-05-11.wav`
- `docs/audio-evidence/nx-clone-pass-p2-2026-05-11.wav`
- `docs/audio-evidence/nx-clone-pass-p3-2026-05-11.wav`

Listen side-by-side to assess timbre similarity vs S1.wav; ASR confirms
the linguistic content is correct.

## What was actually wrong

The previous attempt produced gibberish (`啊！` / `嗯。`) and was
incorrectly attributed to a model-side limitation. The real bug was in
the Mel-spectrogram pre-processing inside the extractor script.

The 1024-d embedding is downstream of:
```
WAV → log-mel (128 bins, 24 kHz, hop 256, win 1024) → speaker_encoder.onnx → embedding
```

If any step of the mel computation diverges from what the speaker
encoder was trained against, the resulting embedding sits in a
different region of vector space than the talker expects, and the
talker produces low-entropy filler tokens.

| Step | Official (`qwen_tts.modeling_qwen3_tts.mel_spectrogram`) | Old script | New script |
|---|---|---|---|
| FFT magnitude | `sqrt(real² + imag² + 1e-9)` | **`abs(STFT)² (power)`** | `sqrt(real² + imag² + 1e-9)` |
| Mel filterbank | `librosa.filters.mel(slaney norm)` | hand-rolled triangles | `librosa.filters.mel(slaney norm)` |
| Reflect padding | `(n_fft - hop_size) // 2` = `(1024-256)/2 = 384` | `n_fft/2 = 512` | `(n_fft - hop_size) // 2 = 384` |
| Log compression | `log(clip(x, 1e-5, None))` (dynamic range compression) | `log(max(x, 1e-10))` | `log(clip(x, 1e-5, None))` |
| Window | `torch.hann_window(win_size)` (cosine starts at 0) | `np.hanning` (cosine starts at 0; identical) | matches |

The dominant errors were **power vs magnitude** and **missing
slaney-norm mel filter**. Both push the spectrogram values to a
completely different numerical range, so even though the ONNX runs
without complaint, the resulting embedding is out-of-distribution
relative to what the Talker prefill expects at the speaker conditioning
row.

The new `scripts/extract_speaker_embedding.py` matches the official
pipeline 1:1 and produces an embedding the Talker correctly conditions
on.

## How the runtime consumes the embedding

The official Qwen3-TTS reference (`modeling_qwen3_tts.py` line ~2103)
chooses the speaker conditioning vector by this branch:

```python
if voice_clone_prompt is None:
    # preset speaker:
    speaker_embed = talker.get_input_embeddings()(spk_id)  # embTable[spk_id]
else:
    # clone:
    if x_vector_only_mode or icl_mode:
        speaker_embed = ref_spk_embedding   # raw 1024-d from speaker_encoder
```

Both go through the same path downstream: `speaker_embed` is inserted
between `codec_prefill_list` and the trailing `[codec_pad, codec_bos]`
embeddings. So the Talker was trained to accept BOTH discrete
embTable rows AND raw speaker-encoder outputs at the same prefill
slot — they live in compatible (post-projection) vector spaces.

The EdgeLLM fork's `qwen3OmniTTSRuntime.cpp` + `assistantPreambleKernel`
(after `6239d5f` "support clone speaker slot") routes
`speaker_embedding_b64` to that same slot. The Talker accepts it as
long as the embedding's distribution matches what the encoder was
trained to produce — which requires the mel pipeline to be exactly the
official one.

## Reproduce

```bash
# On Mac (or any host with librosa + onnxruntime):
python3 scripts/extract_speaker_embedding.py \
  reference.wav \
  speaker_encoder.onnx \
  speaker_emb.b64

EMB=$(cat speaker_emb.b64)
curl -s -N -X POST http://orin-nx:18092/tts/clone/stream \
  -H 'content-type: application/json' \
  -d "{\"text\":\"任意中文文本。\",\"speaker_embedding_b64\":\"$EMB\",\"first_chunk_frames\":7,\"chunk_frames\":10,\"max_chunk_frames\":10}" \
  -o clone.pcm

# clone.pcm starts with 4-byte little-endian sample rate followed by
# PCM s16le mono samples. Wrap as WAV with sample_rate from the header.
```

## Closes

- Issue: `docs/issues/2026-05-11-tts-audio-quality-regression.md`
  (final unresolved sub-item — voice clone)
- Supersedes:
  `docs/performance/qwen3-orin-nx-voice-clone-frozen-artifact-2026-05-11.md`
  (which incorrectly attributed the failure to a model-side limitation).
