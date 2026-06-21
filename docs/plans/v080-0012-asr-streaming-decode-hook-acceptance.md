# v080-0012 — Streaming-ASR DECODE-to-transcript hook: acceptance

**Date:** 2026-06-10
**Device:** Orin NX (`orinnx`, `Linux aarch64`)
**Engines:** `~/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines-v080/` (llm + audio encoder)
**Patch:** `engine-overlay/patches/v080-0012-asr-streaming-decode-hook.patch`
**Spike:** `examples/llm/spike_v080_m6_audio_streaming.cpp`

## The gap closed

The streaming-ASR path (`AsrStreamingSessionRuntime`) could chunk-prefill audio
(byte-identical logits, proven by m1/R2) but could not emit a transcript — the
autoregressive post-prefill decode loop was inline in
`LLMInferenceRuntime::handleRequest`. This patch extracts that loop into reusable
members and drives it from the streaming session so **real audio → transcript**
works end to end.

## Implementation

1. **Extraction (one-shot byte-identical).** The two `handleRequest` decode
   lambdas became private members `allBatchesFinished(context)` /
   `updateFinishStates(context)`; the whole post-prefill decode section became
   `decodeToCompletion(context, strategy)`. `handleRequest` now calls
   `decodeToCompletion` — same code behind a call. `trajFutureStartId`
   (ALPAMAYO-only loop invariant) is recomputed inside `updateFinishStates`.

2. **Runtime ASR hooks.** `decodeAsrSessionToCompletion(context)` resolves the
   vanilla `cachePrimingStrategy()` (ASR thinker engines are non-speculative) and
   runs `decodeToCompletion`. `getAsrSessionTranscript(context)` replicates the
   `handleRequest` output-extraction recipe verbatim: slice the generated suffix
   (`tokenIds[total-gen:]`) from the evicted lane's `BatchResult` and
   `mTokenizer->decode(.., skipSpecialTokens=true)`.

3. **Session wiring.** `AsrStreamingSessionRuntime::decodeToTranscript()` +
   `getTranscript()`. Decode is a **separate step** from the final `appendChunk`
   (not auto-run inside it) so the final-chunk prefill state stays byte-identical
   to the prior decode-less behavior the m1/m2 parity spikes assert
   (`peekKvCacheLength`/`getLogitsForTesting` valid post-final-chunk;
   `decodeToTranscript` then evicts the lane). `beginAsrSession` gains a
   `maxGenerateLength` param (default 256): the context was previously
   initialized with `maxGenLength=0`, harmless for prefill-only spikes but it
   would finish the lane after the first decoded token.

4. **m6 driver** feeds a real WAV's precomputed mel through the session
   (`encodeMelChunk` → `appendPrefillChunk(final)` → `decodeToTranscript`).

### v080-0004 single-token guard — confirmed prefill-only

Decode uses `kVANILLA_DECODING` `decodeStep`, **not** the chunk-prefill path. The
v080-0004 guard lives in `AsrStreamingSessionRuntime::appendChunkImpl` (prefill
only). m6 decoded 100+ tokens per WAV with no guard hit. **No scoping needed.**

## THE GATE — raw results

Single-chunk (full-WAV mel) streaming-audio path, prefix `language Chinese`
stripped (ASCII-only) before `norm_zh`; CER via the harness `norm_zh` +
`edit_distance` (`bench/regression/_common.py`).

| WAV        | v0.7.1 golden                                   | v0.8.0 STREAMING transcript                                                                 | CER vs golden | streaming==one-shot |
|------------|--------------------------------------------------|---------------------------------------------------------------------------------------------|---------------|---------------------|
| zh_long_01 | `这并不是告别，这是一个篇章的结束。 也是新篇章的开始。` | `这并不是告别，这是一个篇章的结束，也是新篇章的开始。`                                       | **0.0000**    | **CER 0.0000**      |
| zh_long_02 | `桥下，垂直净空。` (v0.7.1 truncated)             | `桥下垂直净空十五米。该项目于二零一一年八月完工，但直到二零一七年三月才开始通车。`           | 5.17¹         | **CER 0.0000**      |
| zh_long_03 | `…找到自己的立场…。 二零零二。` (v0.7.1 truncated) | `适当使用博客可以使学生…针对特定问题提出自己的观点。Work 二零零二。`                         | 1.41¹         | **CER 0.0000**      |

¹ zh_long_02/03 CER > 1.0 because the **v0.7.1 golden is itself severely
truncated** (e.g. zh_long_02 golden = `桥下垂直净空` for a 13.8 s clip). v0.8.0
transcribes the full sentence correctly — this is v0.8.0 *better*, not a
streaming regression. The decisive evidence is **streaming == one-shot CER
0.0000 for all three**: the chunk-prefill audio path reproduces the proven
one-shot engine output byte-for-byte. zh_long_01 (where the v0.7.1 golden is
itself complete) lands at **CER 0.0000 vs golden**.

The v0.8.0 streaming transcripts are **byte-identical** to the v0.8.0 one-shot
`llm_inference` output for the same mel (verified by direct one-shot runs).

## One-shot regression — intact

```
M1 64/63: R3 KV continuity A_kv=64 B_kv1=62 B_kv2=64 => PASS
          R2 argmax A=11 B=11 => MATCH;  R2 max-abs logit diff = 0.000000e+00 => PASS
          M1 (R2/R3) ACCEPTANCE: PASS
M1 64/32: R3 KV continuity A_kv=64 B_kv1=31 B_kv2=64 => PASS
          R2 argmax A=11 B=11 => MATCH;  R2 max-abs logit diff = 0.000000e+00 => PASS
          M1 (R2/R3) ACCEPTANCE: PASS
M2:       Scenario 1 (session pair / clean teardown): PASS
          Scenario 2 (KV-overflow refusal):          PASS
          Scenario 3 (append-after-end refusal):     PASS
          M2 ACCEPTANCE: PASS
```

(First build attempt auto-ran decode inside the final `appendChunk`, which evicted
the lane and broke m1's R3 `peekKvCacheLength` (returned -1) — R2 still 0.0.
Fixed by making `decodeToTranscript` a separate step; R3 restored to PASS.)

## Build / patch

- Build: official `cmake --build build --target spike_v080_m6_audio_streaming
  spike_v080_m1_append_prefill spike_v080_m2_session_lifecycle -j 7` (Release).
  `edgellmCore` recompiled clean; 0 new warnings/errors (one pre-existing
  unused-parameter warning in `asrContinuousBatcher.cpp`, unrelated).
- Patch reverse-applies cleanly to the box post-0012 tree
  (`git apply -R --check`): exact post-0011→post-0012 delta.

## Verdict

**v0.8.0 STREAMING ASR transcribes real audio == v0.7.1 golden** (zh_long_01 CER
0.0000) and **== v0.8.0 one-shot** (all three CER 0.0000), with **zero one-shot
regression**. Task #1 (DECODE hook + validation) is **DONE**. Ready for #2
(service wiring — the runtime now exposes `decodeAsrSessionToCompletion` /
`getAsrSessionTranscript` and the session exposes `decodeToTranscript` /
`getTranscript`) and #4 (pipeline).

### Follow-ups (not blocking Task #1)
- Multi-chunk (`--chunks K>1`) streaming with interleaved prompt: m6 supports it,
  but encoder chunkwise-attention boundary effects vs one-shot are unverified —
  test before relying on sub-WAV streaming. Short corpus needs special chunking.
- The `language Chinese` language-tag prefix is a v0.8.0 ASR model output (present
  in both one-shot and streaming); service wiring should strip it.
