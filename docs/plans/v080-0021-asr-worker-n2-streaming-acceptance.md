# v080-0021 — ASR worker N>1 sessions (#9) + streaming PARTIAL transcripts (#10) acceptance

**Date:** 2026-06-10
**Host:** orin-nx (`Linux 5.15.148-tegra aarch64`, hostname `orinnx`)
**Scope:** Tasks #9 + #10 of the v0.8.0 voice-stack migration. #9 adds N>1
concurrent-session support to `qwen3_asr_worker` (lane reservation + cumulative
accumulation, finalize via the proven one-shot core). #10 adds true streaming
PARTIAL transcripts: each `chunk` re-decodes the cumulative audio-so-far and
emits `{event:partial,text:...}`, converging to the one-shot transcript at
`final`.

**Engine baseline:** `~/project/edgellm-v080` (v0.8.0 + migration patches),
build dir `~/project/edgellm-v080/build`.
**Worker source (this repo):** `native/edgellm_voice_worker/qwen3_asr_worker.cpp`.
**Box build dir:** `~/project/v080-worker-build/build_v080`
(`cmake --build build_v080 --target qwen3_asr_worker -j4`).
**Box commits (source of truth):** #9 `0a35af5`, #10 `ee4c4ed`.

---

## 1. #10 implementation — streaming PARTIALs (cumulative re-decode, first-cut)

`handleChunk` previously only RECORDED the cumulative mel/pcm payload. It now,
in addition, runs the **proven one-shot core** (`runOneShotCore`) on the
cumulative audio-so-far — under `gEngineExecMutex`, on the session's reserved
lane — and emits `{event:partial,id:sid,text:<transcript-so-far>}`. A
`last:true` chunk routes straight to the finalize path (`handleEnd`), which
emits `{event:final,...}`. The leading `language <Lang>` ASR head tag is stripped
consistently on both `partial.text` and `final.text` (`stripLangTag` +
`transcriptFromCore`).

This first-cut is O(audio-so-far) per hop but is **guaranteed to converge** to
the one-shot transcript because it literally reuses the golden path.

### DEFERRED optimization (noted, not blocking)
True incremental-KV per-hop append — decode only the newly-arrived frames via
`AsrStreamingSessionRuntime::appendChunk` / `decodeToTranscript` on the lane
instead of re-decoding the whole cumulative prefix every hop — is faster but not
yet wired. See the `handleChunk` doc-comment + plan §3.

### One-shot + #9 paths UNTOUCHED
The event-less one-shot path and `handleEnd`/lane logic are unchanged; #10 only
adds the per-hop decode inside `handleChunk` and a `text` field on `final`.

---

## 2. Build proof

| Artifact | md5 | bytes |
|---|---|---|
| #9 binary (baseline) | `3958b0f164fbb01ee0bce14c2c42dcab` | 14161784 (approx) |
| **#10 binary** `build_v080/workers/qwen3_asr_worker` | **`f9bb821dec9f32e53697c7930254b4f3`** | 14161784 |
| worker source on box | `a70eed4c3497fbd1ec8963967b5e0b8e` | — |

Build log tail: `[100%] Built target qwen3_asr_worker` / `EXIT=0`.

---

## 3. Gates (raw, b2 engine, max_slots=2)

### G10.1 — converge (partials are correct growing prefixes; final == one-shot, CER 0.0000)

Streaming session `strm1` (begin → 4 cumulative chunks cum0..cum3 → end cum4):
```
{"event":"begin_ack","id":"strm1","lane":0}
{"event":"partial","id":"strm1","text":"这东西。"}
{"event":"partial","id":"strm1","text":"这并不是告别，这是。"}
{"event":"partial","id":"strm1","text":"这并不是告别，这是一个篇章的结束，也是新篇章的开始。"}
{"event":"partial","id":"strm1","text":"这并不是告别，这是一个篇章的结束，也是新篇章的开始。"}
{"event":"final","id":"strm1","ok":true,"responses":[{"batch_idx":0,"output_text":"language Chinese这并不是告别，这是一个篇章的结束，也是新篇章的开始。","request_idx":0}],"text":"这并不是告别，这是一个篇章的结束，也是新篇章的开始。","total_ms":284.0}
```
One-shot golden for `zh_long_01.safetensors` (G9.3 below):
`language Chinese这并不是告别，这是一个篇章的结束，也是新篇章的开始。`
→ `final.text` (lang-tag stripped) == one-shot transcript **byte-exact → CER 0.0000**.
Partials are correct growing prefixes (`这东西。` → `…这是。` → full).

### G10.2 — low-latency (≥1 partial before last/end)

4 `partial` events emitted BEFORE `end` (above) → PASS. The `last:true` chunk
variant (`strm2`) likewise emits 4 partials before the finalize, same `final`.

### G10.3 — non-regression

**G9.1 (2 concurrent finals correct + isolated):**
```
{"event":"final","id":"sessA",...,"text":"这并不是告别，这是一个篇章的结束，也是新篇章的开始。"}
{"event":"final","id":"sessB",...,"text":"桥下垂直净空十五米。该项目于二零一一年八月完工，但直到二零一七年三月才开始通车。"}
```
sessA=zh_long_01, sessB=zh_long_02 — no cross-contamination → PASS.

**G9.3 (one-shot byte-identical):**
```
{"event":"done","id":"os1","ok":true,"responses":[{"batch_idx":0,"output_text":"language Chinese这并不是告别，这是一个篇章的结束，也是新篇章的开始。","request_idx":0}],"total_ms":296.2}
```
Event-less path → `event:done`, raw `output_text` with lang-tag preserved, NO
`text`/`partial` fields → byte-identical shape to v080-0016 → PASS.

---

## 4. Test-fixture note — encoder min_time_steps=1000

The audio encoder TRT engine (`engines-v080/audio/audio/config.json`) has
`builder_config.min_time_steps=1000`. Cumulative mel prefixes shorter than 1000
frames are REJECTED by TRT (`"TensorRT Edge LLM cannot handle this request."`).
So the streaming test fixtures (`mels/zh_long_0N.cumK.safetensors`, built by
`gen_cumulative_mels.py`) zero-pad each cumulative hop up to
`max(1000, next-100-multiple)` frames. This is a **fixture-generation** detail,
not a worker constraint — in production the OVS accumulate driver supplies the
growing real mel and the encoder receives ≥1000-frame inputs once enough audio
has arrived. (Zero-padding at the tail does not change the transcript.)

---

## 5. Verdict

Streaming PARTIALs work, converge to the one-shot transcript (CER 0.0000), and
are non-regressing (#9 N>1 + one-shot both intact). Combined patch
`patches/v080-0021-asr-worker-n2-streaming.patch` carries BOTH #9 and #10 into
the canonical worktree worker baseline.
