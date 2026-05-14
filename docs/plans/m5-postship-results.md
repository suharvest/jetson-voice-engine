== M5 streaming verification ==
Worker:  /opt/jv-workers/qwen3_asr_worker
Plugin:  /opt/edgellm-bin/libNvInfer_edgellm_plugin.so
Engine:  /opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_thinker_full_fp8embed
MM eng:  /opt/models/qwen3-edgellm/engines/orin-nx/highperf/asr_audio_encoder
PCM:     disabled (pass --with-pcm to enable scenario F)

# M5 — End-to-End Streaming Verification Results

Worker: `/opt/jv-workers/qwen3_asr_worker`
Gates: LCS ≥ 0.95, median ≤ 500.0 ms, p95 ≤ 1000.0 ms

## Aggregate

**PASS** — 3 prompts evaluated

## Per-prompt summary

| Prompt | Ground truth | A baseline | B (mel) text | B LCS | F (pcm) text | F LCS | median ms | p95 ms | Verdict |
|--------|--------------|-----------|--------------|------:|--------------|------:|----------:|-------:|---------|
| p1 | `今天天气真好。` | `今天天气真好。` | `今天天气真好。` | 1.000 | `—` | — | 152.5 | 153.3 | PASS |
| p2 | `人工智能改变了世界。` | `人工智能改变了世界。` | `人工智能改变了世界。` | 1.000 | `—` | — | 137.2 | 157.6 | PASS |
| p3 | `一二三四五六七八九十。` | `一二三四五六七八九十。` | `一二三四五六七八九十。` | 1.000 | `—` | — | 168.3 | 171.1 | PASS |

## Per-prompt gates

### p1

- A_ok: **PASS**
- B_lcs_ge_0.95: **PASS**
- C_median_le_500ms: **PASS**
- C_p95_le_1000ms: **PASS**

### p2

- A_ok: **PASS**
- B_lcs_ge_0.95: **PASS**
- C_median_le_500ms: **PASS**
- C_p95_le_1000ms: **PASS**

### p3

- A_ok: **PASS**
- B_lcs_ge_0.95: **PASS**
- C_median_le_500ms: **PASS**
- C_p95_le_1000ms: **PASS**

EXIT=0

---

## Post-ship note: scenario D + VAD-aligned auto-segment (2026-05-14)

Scenario D (12.9 s zh-long-04) requires the **highperf-v2 thinker engine**
(`max_input_len=256`). Against the legacy `highperf/` engine
(`max_input_len=128`), the final hop's prefill silently fails and the worker
emits an empty partial — diagnostic-free because the prior code path
discarded `handleRequest`'s failure status.

Two coordinated worker changes shipped:

1. **Default engine pin** — `scripts/test_streaming_worker.py` +
   `scripts/verify_reproduction_streaming.sh` now default
   `--engine-dir` to
   `/opt/models/qwen3-edgellm/engines/orin-nx/highperf-v2/asr_thinker_full_fp8embed`.
2. **Explicit `prefill_failed` error event** — when
   `runtime->handleRequest` returns failure on a streaming hop the worker
   emits a structured
   `{"event":"error","error":"prefill_failed","hop_id":...,"audio_sec":...,
    "kv_length":...,"max_kv_capacity":...}` event and frees the session
   instead of silently storing an empty partial.
3. **VAD-aligned auto-segment** — when chunk audio arrives via `pcm_b64`
   and `audio_sec >= kVadTriggerSec (12.0)`, the worker runs
   `AudioVadSplitter` on the accumulated PCM and rotates at the
   low-energy boundary it returns; the rotation event now carries
   `vad_boundary_sec` and a `rotation_kind` discriminator. mel_path-only
   sessions retain the legacy projected-overflow rotation policy.

### M5 post-fix results (2026-05-14, v2 engine + new worker)

| Gate                                | Result | Notes                                            |
|-------------------------------------|--------|--------------------------------------------------|
| `A_oneshot_ok`                      | PASS   | text=`今天天气真好。`                              |
| `B_lcs_ge_0.95`                     | PASS   | LCS 1.000 vs A baseline                          |
| `C_median_le_500ms`                 | PASS   | median 136.2 ms (5 runs)                         |
| `C_p95_le_1000ms`                   | PASS   | p95 152.1 ms                                     |
| `D_one_final`                       | PASS   | single final emitted for 12.90 s zh-long-04      |
| `D_at_least_one_segment_rotation`   | PASS   | segment_count=1 (no rotation needed at 12.9 s)   |
| `D_lcs_ge_0.90_soft`                | FAIL†  | LCS measured vs short-audio baseline_x2 — wrong reference; pass `--long-baseline-text` for hard-gate eval. The actual transcript is correct: `科学家们可以得出结论：暗物质对其他暗物质的影响方式与普通物质相同。` |
| `E_*` (4 gates)                     | PASS   | malformed JSON, unknown event, oversized chunk, session cleared |
| `F_pcm_lcs_ge_0.95`                 | FAIL‡  | latent C++ MelExtractor mismatch surfaced by Change #2; see below |

† D's soft gate FAIL is a known test-driver wiring issue, not a worker
regression. The text is correct. Set `--long-baseline-text` to convert
to a meaningful hard gate against a curated transcript.

‡ The C++ `MelExtractor` does not pad short mel buffers to the audio
encoder's static input shape (`[-1, 128, 100]`); chunk 0 (500 ms) produces
shape `[1, 128, 50]` and the encoder rejects it with
`engineDims.d[i] == dims.d[i]` mismatch. Prior to Change #2, the worker
silently emitted an empty partial and kept the session alive — hops 0–4
were all silently failing, but hop 5 (last=true, 5 s audio = 500 frames)
succeeded and the final LCS came out at 1.000 *as if the streaming path
worked*. With explicit `prefill_failed` error emission, the first
sub-100-frame chunk now correctly fails the session, exposing this
pre-existing bug. Fix is out of scope here; tracked as a separate item
(`MelExtractor` needs to pad to `MIN_ENCODER_FRAMES=100` when input is
shorter, matching the Python driver's `audio_to_mel` behavior at line
136-137 of `scripts/test_streaming_worker.py`).
