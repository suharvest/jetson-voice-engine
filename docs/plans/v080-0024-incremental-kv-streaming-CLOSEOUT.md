# v080-0024 — ASR Incremental-KV Streaming: CLOSEOUT (done-offline, production-integration DEFERRED)

Status: **OFFLINE-PROVEN, NOT integrated to production** (user decision 2026-06-11, option A). Code kept dormant behind default-OFF env flags; production flag NOT flipped. This supersedes the optimistic framing in `v080-incremental-kv-streaming-plan.md` (the codex design) — read this for what was actually built and measured.

## What this is
Net-new perf work (NOT regression recovery — v0.7.1 also shipped streaming partials as cumulative re-decode; the incremental path was never wired in either version: `asr-worker-build-verify/qwen3_asr_worker.cpp:287-289` "each hop rebuilds, no KV survives between handleRequest calls"). Goal was O(N²)→O(N) per-hop streaming partials by appending only delta audio to a live KV lane + a non-evicting transcript peek, instead of re-decoding the full cumulative audio every hop.

## What was built (all additive, behind 2 default-OFF env flags)
- **Phase 1 — non-evicting peek primitive** (runtime): `LLMInferenceRuntime::peekAsrSessionTranscript` + `restoreKvCacheLength` (1-elem H2D mirror of the D2H read at llmInferenceRuntime.cpp:1851) + `AsrStreamingSessionRuntime::peekTranscript`. Decode-to-EOS then total-restore of the whole `DecodingInferenceContext` + device KV length + cache-mgr activeBatchSize, under the engine mutex. Works because `performBatchEvict` on a single finished lane (`batchMapping=[-1]`) never overwrites that lane's physical KV row (`compactKVCacheBatched` relocates only *retained* lanes; Mamba compaction is a no-op for pure-attention Qwen3-ASR).
- **Phase 2a — wire session-runtime into worker** (`OVS_ASR_SESSION_PATH=1`): `asrSessionPath` namespace; single-hop session path = begin→encodeMel(full)→appendChunk(full)→decode→getTranscript. `runOneShotCore` stays the default.
- **Phase 2b — multi-hop delta + peek-with-suffix** (`OVS_ASR_STREAM_DELTA=1`): `asrStreamDelta` worker namespace + `AsrStreamingSessionRuntime::peekTranscriptWithSuffix` (session layer; speculatively appends `audio_end`+assistant suffix INSIDE the Phase-1 snapshot/restore envelope so the partial can be transcribed from a mid-audio KV, then rolls everything back). Single-token guard (v080-0004) honored.

Token layout (confirmed vs engine `config.json`): prefix `[151644,872,198]` + `audio_start 151669` + N×`audio_pad 151676` + `audio_end 151670` + suffix `[151645,198,151644,77091,198]`.

## Footprint / maintenance
~664 functional lines, all additive, default-OFF (flag-OFF path proven byte-identical):
- worker `qwen3_asr_worker.cpp` **+418** — OUR addon (`native/edgellm_voice_worker/`), never upstream → zero upstream-rebase cost.
- runtime **+246** (`llmInferenceRuntime.cpp +96 / .h +43`, `asrStreamingSessionRuntime.cpp +67 / .h +40`) — patches on the NVIDIA fork, but extend the ALREADY-self-maintained ASR-streaming-session patch (0003), same category, incremental re-port at upstream bumps.
- spikes `spike_v080_m7_peek.cpp` / `spike_v080_m8_delta_peek.cpp` — test-only.
No engine re-export, no plugin/ABI change. Cleanly layered (2b keeps the Phase-1 primitive byte-stable).

## Gates (offline, worker-direct on orin-nx)
- **G1 single-hop** (2a): session-path final == one-shot final, **CER 0.0000** on zh_long_01/02/03 (byte-exact).
- **G1 multi-hop** (synthetic): delta-final == one-shot-final **byte-exact** across ≥5 real window-boundary appends (re-proves delta-encode correctness; `[1,30]` encoder, 3000-frame synthetic).
- **G3 peek zero-residue** (THE crux): 8 mid-audio peeks + 5-hop interleaved sequence — durable KV-length/tokenIds/activeBatch byte-identical pre/post every peek; post-peek real decode byte-identical to golden.
- **G4 regression**: default `runOneShotCore` + 2a single-hop both still CER 0.0000.
- **G2 — per-hop (encode+append) FLAT**: 82–85 ms across hops 1–5 while KV grows 138→814 (ratio 1.03×). The append side is genuinely O(1)/hop.

## ⚠️ HONEST PERF CONCLUSION — corrected by real measurement 2026-06-11 (#12 optimizes the WRONG cost)
**Measured the DEPLOYED v0.8.0 #13 streaming path directly** (runOneShotCore cumulative re-decode, flags OFF, [1,30] minchunk1 encoder ede676fb, worker bin 3adcdf35, real zh_long_03 cumulative slices, decode isolated via `max_generate_length_override`; 30 runs, rep variance <1 ms — task #14 / `~/asr_perf_measure/`):

```
cum_audio  out_chars  prefill_ms  decode_ms  total_ms
  1.0 s        2         28.6        69.2       97.8
  3.0 s       14         31.7       151.0      182.8
  6.0 s       32         36.1       289.8      325.9
 10.0 s       56         43.8       486.7      530.5
 15.4 s       87         56.9       726.8      783.7
```

Ground truth:
- **DECODE is the dominant, growing cost** — `t_decode` 69→727 ms (10.5×), in lockstep with output chars (2→87). The deployed #13 path **re-decodes the whole transcript-so-far every hop** (O(audio_so_far) per hop, explicitly noted at `qwen3_asr_worker.cpp:433-439`), so decode = **87–93% of per-hop time** at ≥3 s. Per-hop `t_total` grows **8× (98→784 ms)**, crossing 200 ms at ~3–3.5 s. Fine for short commands (≤3 s), a real problem ≥6 s.
- **PREFILL (encode+prefill) is a minor slow floor** — only 29→57 ms (2.0×), never the bottleneck.
- **Therefore #12 (incremental-KV) optimizes PREFILL = the minor floor, NOT decode = the bottleneck.** Best case #12 shaves the ~30 ms prefill growth out of 784 ms total at 15 s ⇒ **<5% win** (and ~0 for short commands). My earlier "~10% / prefill is the right bottleneck" framings were BOTH wrong — corrected here by measurement. The synthetic G2 peek_ms curve (565→2388 ms) was inflated by 10 s/hop fixtures + a peek that decodes to EOS; it nonetheless directionally agreed that decode dominates, now confirmed on the real path.
- **#12's peek ALSO re-decodes to EOS every hop**, so #12 does NOT eliminate the decode cost either. (The G2 agent's closing claim that #12 "eliminates re-decode-every-hop" is wrong.)

## The optimization that ACTUALLY matters (not #12): port the v0.7.x decode trick forward
The dominant cost is decode, and the **v0.7.x streaming worker already solved it** (`asr-worker-build-verify/qwen3_asr_worker.cpp:150,832,878`): `maxDecodeTokensPerHop=64` + **assistant-message-as-prefix continuation** = decode only the NEW words each hop (fixed per-hop decode), not the whole transcript. The v0.8.0 #10/#13 migration took a "first-cut" that **DROPPED** this → regressed per-hop decode to O(audio_so_far). **Porting prefix-continuation + per-hop cap into the v0.8.0 worker flattens the dominant cost, is simpler than #12, and is orthogonal to it.** The ideal long-form design = incremental-KV append (#12's half) **+** incremental/capped decode (v0.7.x's half) combined; #12 alone is the less-valuable half.

## Decision (2026-06-11, user option A — reinforced by measurement)
Stop at offline. Keep the dormant flag-OFF code (the incremental-KV-append *half* of a future combined optimization). Do NOT integrate to production: measurement shows #12 optimizes the minor prefill floor (**<5%**), not the decode bottleneck, so it doesn't change felt latency. P2c DEFERRED.
**Follow-on worth more than #12 (NOT yet scoped):** port the v0.7.x `maxDecodeTokensPerHop=64` + assistant-prefix-continuation decode trick into the v0.8.0 worker to flatten the dominant per-hop decode cost (latent regression vs v0.7.x for ≥6 s utterances; short-command use unaffected).

## Durable artifacts
- All changed+new files: `_v080_incremental_kv_12_files.tar.gz` (overlay worktree, md5 `9704f3f30dd9bfc125ca77be42fd061f`) + box `~/asr_incremental_kv_12/`.
- Combined diff: `engine-overlay/patches/v080-0024-asr-incremental-kv-streaming.patch` (744 lines).
- Phase-1 standalone backup: `_phase1_peek_20260610.tar.gz` (md5 `750b8ad6…`).
- Box worker binary with the delta path: md5 `3adcdf35e604c8976729934b264f84fe` (NOT deployed).
- Build: runtime `cmake --build build -j$(nproc)` in `~/project/edgellm-v080`; worker `cmake --build build_v080 -j$(nproc) --target qwen3_asr_worker` in `~/project/v080-worker-build`.
