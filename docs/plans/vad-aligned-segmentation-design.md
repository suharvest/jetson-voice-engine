# VAD-aligned segmentation for streaming ASR worker

**Status:** Phase 1 (C++ port + parity test) — landed. Phase 2 (worker integration) — pending.
**Author:** streaming-asr task #25 (2026-05-14)
**Related:** task #24 (worker source unification), M3-D dedup attempt log
`docs/plans/m3-d-dedup-attempt-2026-05-13.log`.

## Problem

`runStreamingHop` auto-segment currently cuts the rolling buffer on a fixed
5.5 s wall-clock trigger. On long Chinese utterances (`zh-long-04`,
12.9 s) the cut lands mid-word, the next hop's hypothesis duplicates a
half-syllable, and the semantic-dedup boundary detector cannot recover a
clean splice → LCS lands at 0.78 instead of the M5-D target ≥ 0.95.

## Algorithm — `split_audio_into_chunks`

Mirror of `Qwen3-ASR/qwen_asr/inference/utils.py:246`:

```
total_sec ≤ max_chunk_sec → single chunk
else loop while (total - start) > max_len:
    cut   = start + max_len
    left  = max(start, cut - expand)
    right = min(total, cut + expand)
    if right - left ≤ win:           # not enough room to search
        boundary = cut
    else:
        seg_abs = |wav[left:right]|
        sums    = convolve(seg_abs, ones(win), 'valid')   # 100 ms window
        min_pos = argmin(sums)                            # first-min tie-break
        inner   = argmin(seg_abs[min_pos : min_pos+win])
        boundary = left + min_pos + inner
    boundary = clamp(boundary, start+1, total)
    emit chunk wav[start:boundary]; start = boundary
emit tail wav[start:total]
pad each chunk shorter than MIN_ASR_INPUT_SECONDS (0.5 s) with trailing zeros
```

Properties:
- Chunks concatenate to the input **sample-for-sample** (before tail padding).
- Internal chunk length ∈ `[1, max_len + expand]`.
- Tail chunk length ∈ `[1, max_len]` (pre-pad).
- Boundary is the lowest-energy 100 ms window within ±expand of the target cut.

## Parameter rationale

| Parameter            | Python default | Worker Phase 2 default | Rationale |
|----------------------|----------------|------------------------|-----------|
| `max_chunk_sec`      | n/a            | **5.5 s**              | Matches current `runStreamingHop` auto-segment trigger; can rise to 13 s once engine headroom verified. |
| `search_expand_sec`  | 5.0 s          | **2.0 s**              | At 5.5 s hops, ±5 s would allow a cut at 0.5–10.5 s, blowing past max. ±2 s keeps internal cap at 7.5 s — safely under 15 s engine cap. |
| `min_window_ms`      | 100.0 ms       | 100.0 ms               | Verbatim. 100 ms ≈ a syllable nucleus, robust to spectral noise. |
| `min_chunk_sec`      | 0.5 s (`MIN_ASR_INPUT_SECONDS`) | 0.5 s | Verbatim — tail-pad with zeros for very-short chunks. |

## Phase 1 — what landed

- `native/edgellm_voice_worker/audio_vad_split.{h,cpp}` — `AudioVadSplitter` class. Header-only API surface; impl uses an online running-sum window (`O(N)`, no FFT).
- `native/edgellm_voice_worker/tests/test_audio_vad_split.cpp` — WAV loader + concat-parity + max-len + boundary-energy sanity + JSON dump.
- `scripts/test_audio_vad_parity.py` — runs inlined numpy reference and C++ binary on the same WAV; compares within ±1 sample tolerance.
- `docs/plans/vad-cmake-integration-notes.md` — proposed CMake additions, deferred to Phase 2 to avoid conflicting with task #24 worker unification.

### Phase 1 evidence (zh-long-04, 12.9 s)

| max_chunk_sec | py boundaries (samples)       | cpp boundaries                 | max diff |
|---------------|-------------------------------|--------------------------------|----------|
| 13.0          | `[206400]`                    | `[206400]`                     | 0        |
| 5.5           | `[75531, 185000, 206400]`     | `[75531, 185000, 206400]`      | 0        |
| 4.0           | `[75531, 161385, 206400]`     | `[75531, 161385, 206400]`      | 0        |

Boundary energy ratio at the strong cut (sample 75531 = 4.72 s):
`e_at=0.000501  e_mean=0.037793  ratio=0.013` — i.e. 75× quieter than
surrounding region. Excellent silence point, matches the natural pause
between Chinese clauses heard in playback.

## Phase 2 — worker integration plan (NOT in this commit)

1. Add `audio_vad_split.{h,cpp}` to `qwen3_asr_worker` CMake target (see
   `vad-cmake-integration-notes.md`).
2. In `runStreamingHop`, replace the bare fixed-time cut with:
   ```cpp
   AudioVadSplitter splitter(16000, /*max=*/5.5, /*expand=*/2.0);
   auto bnds = splitter.boundaries(rolling_buffer);
   // first boundary is the next safe cut point
   ```
   The semantic-dedup boundary detector then operates on a cut that ends
   in a low-energy frame → adjacent hops share clean silence rather than
   a half-syllable.
3. Keep the fixed-time *trigger* (5.5 s elapsed), but use VAD to *choose
   the cut sample*. If no silence exists within ±2 s, fall back to the
   raw cut (same as today).
4. Optionally raise `max_chunk_sec` to 13.0 once engine headroom is
   re-measured under the unified worker (task #24 will repaint that math).

### Acceptance for Phase 2 (M5-D rerun)

Repeat the `zh-long-04` M5-D acceptance scenario:
- **Scenario A** (no auto-segment, single 12.9 s hop): LCS unchanged
  (this commit doesn't touch the single-hop path).
- **Scenario B** (auto-segment ON, VAD-aligned cuts at 4.72 s + 11.56 s):
  expect LCS ≥ 0.95 against ground truth — vs 0.78 today with fixed
  5.5 s cut. If LCS < 0.95, blame either (a) semantic-dedup boundary
  detector still mis-aligns (orthogonal fix), or (b) silence at 4.72 s
  is in the wrong clause for the dedup model.

## Out of scope (Phase 1)

- `qwen3_asr_worker.cpp` edits — task #24 owns the worker source surface.
- `CMakeLists.txt` edits — deferred to Phase 2.
- Any change to the M5-D semantic-dedup boundary detector.
- WebRTC-style aggressive VAD (we use energy-only, identical to Qwen3-ASR upstream).
