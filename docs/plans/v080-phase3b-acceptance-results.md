# Phase 3b — Empirical acceptance of the v0.8.0 ASR streaming hooks

Status: RUN 2026-06-09 on orin-nx against a freshly-built v0.8.0 Qwen3-ASR-0.6B engine.
Gates the Phase 2-3 hooks (patches v080-0001 / v080-0002) that until now were only
compile-proven + code-inspection-proven byte-identical. This is the de-risk gate before Phase 4.
**UPDATE 2026-06-10: the lone 1-token-final-chunk regression is FIXED (patch `v080-0004`)
and re-verified empirically — R2 parity is now FULLY green for all final-chunk sizes. See §6.**

Verdict: **R2 chunk-prefill parity HOLDS (byte-identical, diff 0.0) for EVERY split — both
final chunk ≥ 2 tokens AND, after patch `v080-0004`, final chunk == 1 token.** The single
isolated regression (a 1-token final/intermediate prefill chunk mis-sampling, argmax flip,
max-abs-diff ~2e+01) was root-caused to the AttentionPlugin's shape-only prefill-vs-decode
deduction and fixed by a runtime single-token append-prefill guard (§6). Lifecycle /
KV-overflow / R3 / R4 all PASS. **Phase 4 is UNBLOCKED.**

---

## 1. v0.8.0 Qwen3-ASR engine build

The pre-existing `engines-0508` and a rebuild from the May-1 ONNX were **rejected by the
v0.8.0 runtime** — that ONNX config predates the v0.8.0 schema:

```
[ERROR] llmEngineConfig.cpp:98: parseEngineConfig: config.json missing required field
        'kv_cache_dtype'. Re-export the model with the latest llm_export.py to record it.
```

So a fresh v0.8.0 export + build was required.

- **Export** (`.venv-x86export`, HF mirror; `hf-xet` had to be uninstalled — the xet
  cas-bridge stalled the download):
  ```
  tensorrt-edgellm-export Qwen/Qwen3-ASR-0.6B \
      $WS/Qwen3-ASR-0.6B/onnx-v080 --max-kv-cache-capacity 4096   # EXPORT_EXIT=0
  ```
  New LLM config: `"edgellm_version": "0.8.0"`, `"kv_cache_dtype": "fp16"`.

- **Build** (official `llm_build` / `audio_build`):
  ```
  llm_build  --onnxDir onnx-v080/llm   --engineDir engines-v080/llm \
             --maxBatchSize 1 --maxInputLen 4096 --maxKVCacheCapacity 4096   # LLM_EXIT=0
  audio_build --onnxDir onnx-v080/audio --engineDir engines-v080/audio \
             --minTimeSteps 1000 --maxTimeSteps 3000                          # AUDIO_EXIT=0
  ```

- **Engine paths + md5** (orin-nx):
  ```
  b133dff24c8aa96ac1679b95e2f97153  engines-v080/llm/llm.engine            (1.21 GB)
  5c877cfe58b8fcb7914679c6fe274f90  engines-v080/audio/audio/audio_encoder.engine (377 MB)
  ```

- **v0.8.0 compatibility proof** — the official `llm_inference` loads the engine and runs a
  full prefill→decode→sample→EOS on a text request:
  ```
  INFER_EXIT=0
  "finish_reason": "end-of-sequence"
  "output_text": "language None"
  ```
  (Text output is semantically meaningless — Qwen3-ASR is audio-conditioned — but the
  thinker runtime path is fully exercised, which is what the spikes drive.) The one-shot
  *audio* path additionally needs a correctly-shaped mel; the pre-made mel on the box
  (`[6,128,100]`) falls outside this encoder profile's `[10..30,128,100]` and is an input-data
  mismatch, not an engine fault.

## 2. Spike drivers (patch v080-0003-asr-streaming-spikes.patch)

Both drive the new `rt::AsrStreamingSessionRuntime` over a real `LLMInferenceRuntime`
constructed from the engine above. They use ordinary **text token IDs** (not `<audio_pad>`)
so the embedding lookup is a pure text gather — this isolates the refactored prefill-chunk
packing/exec/commit math (the v080-0002 R2 seam at `runBaseModelPrefillChunk`,
`llmInferenceRuntime.cpp:1105-1108`) with no mel / audio encoder.

- `spike_v080_m1_append_prefill.cpp` — R2 + R3.
  Path A: `begin → appendChunk(all N, final)`. Path B: `begin → appendChunk(N1,!final) →
  appendChunk(N2,final)`. Asserts: argmax(A)==argmax(B); `max_i |logitA−logitB| < 1e-2`;
  KV continuity `B_kv1==N1`, `B_kv2==N==A_kv`.
- `spike_v080_m2_session_lifecycle.cpp` — lifecycle + capacity refusal.
  Sc1 session-pair (two identical sessions → byte-identical logits + KV→0 after end).
  Sc2 KV-overflow (`appended+chunk>maxPositions` ⇒ `appendChunk`==false, KV NOT advanced,
  recovers after end+begin). Sc3 append-after-end refused.

Both compile clean into the example binaries and link against the patched `edgellmCore`.
(Needed `loadEdgellmPluginLib()` in `main()` to register `AttentionPlugin` before deserialize,
mirroring `llm_inference.cpp:486`.)

## 3. Empirical gate results (raw)

### R2 / R3 — single-chunk vs split-chunk (m1)

```
N=40 split=17  R3 KV: A=40 B1=17 B2=40 PASS   R2 argmax A=13 B=13 MATCH  diff=0.000000e+00  PASS
N=64 split=1   R3 KV: A=64 B1=1  B2=64 PASS   R2 argmax A=11 B=11 MATCH  diff=0.000000e+00  PASS
N=64 split=32  R3 KV: A=64 B1=32 B2=64 PASS   R2 argmax A=11 B=11 MATCH  diff=0.000000e+00  PASS
N=100 split=50 R3 KV: A=100 B1=50 B2=100 PASS R2 argmax A=13 B=13 MATCH  diff=0.000000e+00  PASS
N=64 split=62  (final 2 tok)  argmax MATCH  diff=0.000000e+00  PASS
N=64 split=61  (final 3 tok)  argmax MATCH  diff=0.000000e+00  PASS
N=40 split=38  (final 2 tok)  argmax MATCH  diff=0.000000e+00  PASS
```
=> **R2 byte-identical (diff exactly 0.0) and R3 KV continuous for every final-chunk ≥ 2 tokens.**

### ⚠️ Regression — 1-token FINAL prefill chunk diverges  →  ✅ FIXED in §6 (patch `v080-0004`, 2026-06-10)

```
N=64 split=63 (final 1 tok)  argmax A=11  B=11   MATCH      diff=2.002344e+01  FAIL
N=40 split=39 (final 1 tok)  argmax A=13  B=11   MISMATCH   diff=2.251562e+01  FAIL
N=30 split=29 (final 1 tok)  argmax A=487 B=2142 MISMATCH   diff=1.889355e+01  FAIL
```
- Reproducible, deterministic, ONLY when the final `appendChunk` carries exactly 1 token.
- KV length is still correct (`B_kv2==N`) → prefill/commit is fine; only the **logits sampled
  from a length-1 final prefill chunk** are wrong, and the error is large enough to flip argmax.
- Likely root cause (for Phase 4): the last-token gather / sampling-from-prefill after
  `runBaseModelPrefillChunk` uses the per-chunk `hostContextLengths`/`selectTokenIndices`
  (`mStepPreparer->prepare(kPrefill,...)`, `pipelineIO.cpp` `selectTokenIndices`) which for a
  chunk now holds the CHUNK length (=1), not the accumulated sequence length — the
  `inputIdsLength==1` prefill profile / index selection does not align with how the one-shot
  path gathers the final-position logits. The one-shot path never hits `inputIdsLength==1`,
  so this surface was invisible to the code-inspection proof.

### Lifecycle / KV-overflow (m2)

```
Scenario 1 (session pair / clean teardown):  kv_after_appends=38/38, kv_after_end=0/0,
                                             max-abs diff between sessions = 0.000000e+00  PASS
Scenario 2 (KV-overflow refusal):  call0 ok=1 kv=100, call1 ok=1 kv=200,
                                    call2 ok=0 kv=200 (refused, no advance), recover kv=100  PASS
Scenario 3 (append-after-end refusal):  append after end ok=0 (status kIdle)  PASS
=== M2 ACCEPTANCE: PASS ===
```

### R4 — sys-prompt cache mismatch fallback

Structurally verified in the compiled box source that produced the passing runs:
`restoreKVCache` (`llmInferenceRuntime.cpp:1408`) is now INSIDE `if (matchIds)` (`:1405`),
and any mismatch falls through to a fresh prefill via `if (!useCachedKV)` (`:1443`) — the
validation (`shapeOk`/`lengthOk`/`matchIds`) runs BEFORE restore. The public
`AsrStreamingSessionRuntime` API cannot inject a *mismatched* primed sys-prompt cache (it
always passes the same empty prompt), so this is code-verified-in-binary, not independently
engine-driven. The session-pair byte-identity (Sc1, diff 0.0) confirms re-priming does not
pick up stale KV.

## 4. Verdict (gates Phase 4)

| Gate | Result |
|---|---|
| v0.8.0 engine build + loads in `llm_inference` | ✅ |
| R2 single-vs-split argmax MATCH + logit diff < 1e-2 (final chunk ≥ 2 tok) | ✅ **diff = 0.0 exactly** |
| R3 MRope/KV position continuity across chunks | ✅ |
| KV-overflow refusal (no silent advance) | ✅ |
| Lifecycle session-pair byte-identity + clean teardown | ✅ |
| R4 sys-prompt fallback (validation before restore) | ✅ code-verified-in-binary |
| **R2 with 1-token FINAL chunk** | ✅ **FIXED (patch `v080-0004`) — argmax MATCH + diff 0.0; see §6** |

**The Phase 2-3 chunk-prefill refactor is empirically non-regressing for multi-token chunks
(byte-identical to one-shot). The lone failure was a length-1 final prefill chunk — now FIXED
(patch `v080-0004`, §6).**

## 5. Reproduce

```
# engines: $WS/Qwen3-ASR-0.6B/engines-v080/{llm, audio/audio}  (orin-nx)
cd ~/project/edgellm-v080
cmake --build build --target spike_v080_m1_append_prefill spike_v080_m2_session_lifecycle
ENG=$HOME/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines-v080/llm
./build/examples/llm/spike_v080_m1_append_prefill  $ENG 40 17   # PASS (diff 0.0)
./build/examples/llm/spike_v080_m1_append_prefill  $ENG 40 39   # PASS post-v080-0004 (was FAIL: 1-tok final)
./build/examples/llm/spike_v080_m2_session_lifecycle $ENG       # PASS x3
```

## 6. Fix — single-token append-prefill guard (patch `v080-0004`, 2026-06-10)

### Root cause (file:line)
Not in `runBaseModelPrefillChunk`. `deduceModeVanilla` in
`cpp/plugins/attentionPlugin/attentionPlugin.cpp:93-110` distinguishes a chunked-prefill step
from a vanilla-decode step **SOLELY** by `qInputTensor.getShape()[1]` (runtime Q seq-len):

```
non-empty kvcache_start_index && qSeqLen >  1  -> kCHUNKED_PREFILL
non-empty kvcache_start_index && qSeqLen == 1  -> kVANILLA_DECODING
```

A FINAL (or intermediate) prefill chunk of exactly **1 token** landing on a **non-empty KV
cache** therefore has `qSeqLen==1` + non-empty `kvcache_start_index` → misrouted to the DECODE
path. In decode, RoPE position is derived from `contextLength`:
`launchApplyRopeWriteKV(..., contextLengthTensor, ...)` → in
`cpp/kernels/posEncoding/applyRopeWriteKV.cu:150` `posStartId = kvCacheEndLens[b] - qSeqLen`.
The chunk path sets `contextLength = chunkLen = 1`, so `posStartId = 1 - 1 = 0`
(**chunk-relative**) instead of the absolute `priorKV + 0`; and the XQA decoder attends over
`sequence_lengths = contextLength = 1` instead of the full committed KV. That is the ~2e+01
logit error (argmax flip). The chunked-prefill branch instead computes absolute
`kvCacheEndIdxs = kvCacheStartIdx + contextLength` and FMHA attends the full KV via
`cuKVSeqLens` — which is why multi-token chunks were byte-identical. The gather index
(`selectTokenIndices = hostContextLengths-1 = 0`) is correct for chunkLen==1 and is NOT the
bug; the RoPE/attention **position binding** is. The plugin boundary has no shape-distinct
signal: a genuine decode and a 1-token prefill-append are both `qSeqLen==1` with non-empty KV.

### The fix (why correct; one-shot byte-identical)
Enforce the invariant in the runtime, not the plugin/engine ABI:
`AsrStreamingSessionRuntime::appendChunkImpl` carries the trailing token of a non-final
**text** chunk forward into the next chunk, so every step after the first (and the final step)
packs ≥ 2 tokens and stays on the chunked-prefill path. The first chunk runs on an empty KV
cache (`kNORMAL_PREFILL`, valid at length 1) and is never deferred. Carry-over is text-only
(an audio chunk binds its own per-chunk audio embeddings and is never length 1 in practice);
an audio 1-token step on non-empty KV is hard-refused. Carry state commits only after the
prefill succeeds (refused appends leave it intact). This covers BOTH a 1-token final chunk
(wrong sampled token) and a 1-token intermediate chunk (wrong RoPE position baked into KV).
**The one-shot path (`runBaseModelPrefill → runBaseModelPrefillChunk(fullSpans)`) is NOT
touched** — `appendChunkImpl` is the only caller affected; Path A (one-shot-equivalent) logits
are unchanged (argmax 487/13/11/13 identical to §3). The m1 R3 assertion is relaxed to accept
the coalesced intermediate boundary (`kvB1 ∈ {splitAt-1, splitAt}`; final `kvB2 == totalTokens`
and strict monotonicity unchanged).

### Empirical gate (raw, orin-nx, engines-v080)

1-token FINAL splits (was FAIL diff ~1.9–2.3e+01, argmax could flip):
```
N=30 split=29  R3 B_kv1=28 B_kv2=30 PASS  R2 argmax A=487 B=487 MATCH  diff=0.000000e+00  PASS
N=40 split=39  R3 B_kv1=38 B_kv2=40 PASS  R2 argmax A=13  B=13  MATCH  diff=0.000000e+00  PASS
N=64 split=63  R3 B_kv1=62 B_kv2=64 PASS  R2 argmax A=11  B=11  MATCH  diff=0.000000e+00  PASS
N=100 split=99 R3 B_kv1=98 B_kv2=100 PASS R2 argmax A=13  B=13  MATCH  diff=0.000000e+00  PASS
```

Multi-token splits (no regression):
```
N=64 split=32  diff=0.000000e+00 PASS   N=40 split=17  diff=0.000000e+00 PASS
N=100 split=50 diff=0.000000e+00 PASS   N=64 split=62  diff=0.000000e+00 PASS
N=40 split=38  diff=0.000000e+00 PASS
```

m2 lifecycle: Scenario 1 (session-pair, max-abs diff 0.0) PASS; Scenario 2 (KV-overflow
refusal, kv 99→199, call2 refused no-advance, recover) PASS; Scenario 3 (append-after-end
refused) PASS → **M2 ACCEPTANCE: PASS**.

**Verdict: R2 chunk-prefill parity is FULLY green for ALL final-chunk sizes. Phase 4 unblocked.**
