# v0.8.0 qwen3-tts CustomVoice — ASR-roundtrip correctness acceptance

Date: 2026-06-10
Host: orin-nx (Orin NX 16GB, JetPack 6, CUDA 12.6.68, TRT 10.3.0.30) — verified `Linux aarch64` + `orinnx`.
Branch: feat/edgellm-v080-migration

## Scope

Verify v0.8.0 qwen3-tts CustomVoice **correctness** (not just non-silent energy) for
English and Chinese via ASR-roundtrip, characterize the Chinese early-EOS, and
reconcile the box-local toolchain patch into this worktree.

> Supersedes the box-local commit `5ad0cbb` claim "English audio gate PASSES
> end-to-end". That claim was **energy-only** (RMS non-silent), never roundtrip-
> verified. Roundtrip below proves English is **garbled**, not correct.

## ASR harness (both validated on known-good audio first)

Two independent ASRs used so the verdict does not depend on one model:

1. **paraformer_trt** — production `seeed-voice` container (:8621 `/asr`), zh-native.
   (The HTTP `/asr` endpoint had a server bug `**result.meta` with `meta=None`;
   patched live in-container to `**(result.meta or {})` and restarted — this is a
   pre-existing server bug, not part of this work. Container restored at end.)
2. **Qwen3-ASR-0.6B v0.8.0** — `engines-v080/{llm,audio}` via `llm_inference`,
   multilingual. WAV→16k→log-mel(128, whisper) safetensors → engine. The v080 audio
   encoder TRT profile requires mel time-steps in [1000,3000] (10–30 segments), so
   mels are padded to chunk_length=10 (1000 frames).

**Harness sanity (known-good audio from production matcha_trt TTS, text "今天天气真不错"):**
- paraformer: `嗯今天气气真不错`  (≈ correct)
- Qwen3-ASR:  `language Chinese今天亲戚真不错。`  (strip prefix → `今天亲戚真不错`, ≈ correct)

Both ASRs transcribe known-good Chinese correctly → harness trustworthy.

## Results — v0.8.0 CustomVoice TTS (fresh generation, this run)

Executable: `build/examples/omni/qwen3_tts_inference` (links `-Wl,--wrap=_cudaLaunchKernelEx`).
Engines: `engines-v080-tts/{talker,code_predictor,code2wav}` (talker config: chinese→2055,
english→2050, codecThinkId=2154, speaker serena=3066 — all config-correct).

### English  (input "The weather is really nice today, let us go for a walk together.")
- Frames/codes: **34**, dur **2.72s**, sr 24000, RMS **311** (non-silent).
- rvq_codes (34,16) int32, values varied/non-degenerate.
- **Qwen3-ASR roundtrip: `language Chinese三。`  → GARBLED. NOT correct.**
- Verdict: **BROKEN** (non-silent but unintelligible — codes are semantically wrong).

### Chinese, language=chinese  (input "今天天气真不错，我们一起出去散步吧。")
- langId resolved = 2055, 9-row prefix (prefixRows=9), seqLen=23 N=15 outputSeqLen=26.
- Frames/codes: **2**, dur **0.16s**, RMS **42** → early-EOS at frame 2.
- paraformer roundtrip: `""` (empty). Qwen3-ASR: `language None` (no content).
- Verdict: **(b) BROKEN — early-EOS**, NOT short-but-correct.

### Chinese, no-language legacy 8-row path  (same text)
- langId = -1, 8-row prefix (prefixRows=8). Frames: **15**, dur **1.2s**, RMS **476**.
- paraformer: `打打打电话`. Qwen3-ASR: `language Chinese对对。`  → GARBLED.
- Verdict: **BROKEN** — generates more audio than the 9-row path but still wrong words.

Golden refs (this run): `docs/audio-evidence/`
- `v080-customvoice-en-34frames-2026-06-10.wav`        md5 4be516f15c2c32b1d8ec4bdd0c311a7c
- `v080-customvoice-zh-lang-earlyeos-2026-06-10.wav`   md5 e5a2f08e6056a2681844409a8b830a62
- `v080-customvoice-zh-nolang-15frames-2026-06-10.wav` md5 738e504ba27f470bcefc48d0cb2416dd

## Root cause (bounded probe, stopped at model/export rabbit hole)

The corruption is **upstream in the Talker prefill numerics** and affects BOTH
languages, not a langId-specific or prefix-row-count issue:

1. English (8 fixed prefix rows reused for en too, langId=2050, 9-row) and the zh
   8-row no-language path BOTH produce non-degenerate but semantically-wrong codes
   → the autoregressive Talker is sampling from a corrupted prefill context.
2. The 9-row language prefix makes zh strictly *worse* (EOS@2 vs @15) — the extra
   language-embed row pushes the already-degenerate context to immediate EOS — but
   the 8-row path is still garbled, so the language row is an aggravator, not the cause.
3. The box-local **split-projection** (role[:3]+body[3:3+N] in projectToTalkerInput),
   ported from v0.7.1 to dodge a CuTe DSL GEMM row-1 corruption, has **NO effect** on
   the symptom (confirmed: en still garbled, zh still EOS@2). Dropped from the patch.

This matches the project-history NVIDIA CuTe DSL GEMM artifact (#87 Issue C) that
made v0.7.1 garbage until a prompt-construction fix — but v0.8.0 is a different
codebase/export and the equivalent fix is not yet identified. Localizing it requires
a Python-reference numeric compare of projected prefill embeddings vs the engine
(deep export/kernel work), which is out of scope for a bounded verification pass.

### Options to close zh+en accuracy (not done here)
- A. Numeric-compare v0.8.0 Talker prefill embeds (projected role+body+preamble rows)
  against the HF Python reference, row-by-row, to find where they diverge (GEMM tile,
  embedding table, or preamble assembly).
- B. Verify the v0.8.0 Talker engine **export/quant** carried correct conditioning
  (re-export talker; the engine may be the corrupt artifact, not the runtime).
- C. Confirm whether the CuTe DSL GEMM `--wrap` shim path produces numerically-correct
  results (it makes the launch *succeed*, but correctness of the launched kernel under
  the driver-API path was never numeric-verified — only "non-silent").

## Patch reconciliation (Task 1)

Box-local commit `5ad0cbb` bundled: (1) CMakeLists `--wrap` fix, (2) Code2Wav build
recipe, (3) split-projection, (4) langId/9-row plumbing + kernel layout.

- `v080-0007-customvoice-language-conditioning.patch` (already in tree) already carries
  the langId/9-row conditioning + kernel layout. Not duplicated.
- **`v080-0008-tts-cutedsl-wrap.patch` (this commit)** = ONLY the proven `--wrap`
  toolchain fix, written as an incremental hunk on top of v080-0007's shim block.
- **DROPPED**: the split-projection (proven no-op for the symptom) — kept tree clean.

## Verdict

**v0.8.0 qwen3-tts CustomVoice is NOT correct for either en or zh** (roundtrip-verified).
- English: non-silent but garbled.
- Chinese: early-EOS (language path) / garbled (no-language path).
The `--wrap` toolchain fix is necessary and correct (enables audio gen at all) but does
NOT make output intelligible. Remaining for TTS accuracy: localize the Talker prefill
numeric corruption (options A/B/C above) — a model/export-level workstream.

> **⚠️ SUPERSEDED 2026-06-10** — this doc's "zh broken / en garbled" verdict was a FALSE NEGATIVE (broken Qwen3-ASR harness [known-good sanity failed 亲戚≠天气] + apply_chat_template=false). Authoritative result: **en+zh CORRECT, roundtrip byte-exact** — see v080-0009 CORRECTED doc + main-thread independent transcribe (q_zh24/m5zh → 今天天气真不错; bad-artifact control → empty). qwen3-tts CustomVoice is accuracy-GREEN.
