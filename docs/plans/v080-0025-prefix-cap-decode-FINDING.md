# v080-0025 — Prefix-continuation flat-decode streaming

> ⚠️ **2026-06-11 — THIS DOC'S CONCLUSION IS WRONG. RESOLVED — see `v080-0026-prefix-rollback-streaming-WORKS.md`.** The CER 0.08–0.975 was a **PORT BUG** (dropped the v0.7.x **chunk-and-confirm rollback**), NOT a model limitation. With rollback (`unfixedTokenNum=5` + `unfixedChunkNum=2`): G-correct **CER 0.0000 / 0.025 / 0.0115** (catastrophic poisoning GONE), G-perf **decode FLAT** (3–6× faster per-hop at 10–15 s). The "correctness vs flat-decode is fundamentally incompatible / no free lunch" claim below is **FALSE** — the rollback window resolves it (revision is local). Everything below is the analysis of the BUGGY no-rollback version; kept only for the record.

Status (original, now under correction): ~~REJECTED on correctness.~~ Code reverted (box worker back to #12 state, md5 `3960ccb5`). The exploratory diff is preserved at `engine-overlay/patches/v080-0025-prefix-cap-BROKEN.patch` for the record only — DO NOT ship `OVS_ASR_STREAM_PREFIX`. This is the most important conclusion of the whole streaming-latency investigation (#14→#15): **there is no free algorithmic win on per-partial decode.**

## Goal (from #14's finding)
Deployed v0.8.0 #13 streaming re-decodes the whole transcript every hop → decode dominates and grows O(audio_so_far) (measured: total 98→784 ms over 1–15.4 s). Port the v0.7.x "assistant-message-as-prefix continuation + per-hop cap" trick so each hop decodes only NEW words (fixed decode).

## Feasibility gate: GREEN (no runtime change needed)
The v0.8.0 vanilla `rt::LLMInferenceRuntime` correctly does assistant-prefix continuation (the "Method B fix" the v0.7.x worker required is already present/equivalent). Probe (full zh_long_03 audio + assistant message = `language Chinese` + first-half-transcript, `apply_chat_template=true, add_generation_prompt=false, cap=64`): the runtime prefilled the prefix and generated a genuine continuation, byte-identical to the one-shot tail. The request schema carries it (`requests[].messages[]` generic role; `requestFileParser` honors the JSON keys at `llmInferenceRuntime.cpp:450`). So the mechanism WORKS.

## Correctness gate: FAIL — and it's fundamental, not tunable
Per-hop prefix-continuation final transcripts vs one-shot:
- zh_long_01 CER 0.115, zh_long_03 CER 0.081, **zh_long_02 CER 0.975** (`桥下垂直净空十五米…` → `《中国共产党历史》。`).

Root cause (diagnosed, cap-swept to rule out tuning): the trick requires each hop's prefix to be a **strict prefix** of the final transcript, but on a **partial-audio slice** the ASR model emits a natural sentence-final `。`+EOS — or outright **hallucinates** (zh_long_02 @2.8 s confidently emits `《中国共产党历史》。`+EOS). These errors are **permanently committed** into the prefix (append-only — later hops can only append, never revise). A wrong EOS-terminated prefix then poisons every subsequent hop (continuations see a "finished" sentence contradicting the audio → emit empty). Cap sweep 64/96/128/160 → identical output ⇒ it's intrinsic model behavior on truncated audio, NOT cap truncation.

## The fundamental tradeoff (the real takeaway — no free lunch)
| Streaming approach | per-hop decode | partial correctness |
|---|---|---|
| Deployed #13 cumulative re-decode | O(audio_so_far), grows | **correct + revising** (re-derives whole transcript each hop, self-corrects) |
| Prefix-continuation (this, v080-0025) | **flat** (new words only) | **wrong + non-revising** (commits premature-EOS/hallucination, can't fix) |
| #12 peek (`OVS_ASR_STREAM_DELTA`) | O(audio_so_far) (peek decodes to EOS) | correct + revising |

**Producing a CORRECT, REVISING full-transcript partial is inherently O(transcript-so-far) decode per hop** — you must re-derive to revise, and revising is required because the model is unreliable on partial audio (premature boundaries, hallucination). To make decode flat you must commit/reuse prior decode, which on partial audio commits unrevisable errors. The two cannot be had together. The deployed cumulative path pays the O(audio) decode as the **inherent price of correct revising partials**; it is a sensible correctness/latency point, not a defect.

## Correction to the earlier "v0.8.0 regressed decode vs v0.7.x" hypothesis (#14 doc)
Not a clean regression. v0.7.x's prefix-continuation was **cheaper but produced non-revising, error-prone partials** (the broken behavior proven here). v0.8.0 #13's cumulative re-decode is **costlier but produces correct, revising partials** — a quality-for-latency trade. For the short-command workload (≤3 s) both are <200 ms, so there is **no perceived difference**; the cost gap only appears on ≥6 s utterances, where #13 buys better partial quality.

## What remains (no pure win)
- The only clean algorithmic optimization is **#12's prefill floor (<5%, deferred)** — does not touch the dominant decode.
- Reducing long-utterance per-partial latency is a **product/UX tradeoff**, not a free win: coarser partial cadence (emit less often), bounded/truncated partial display (show only last N words), or cheap non-revising partials for display + one correct final re-decode (accepts jarring un-revised intermediate text).
- Recommendation: **keep the deployed cumulative re-decode path.** It's correct, revising, and fine for the short-command workload. Close this line.

## Guardrails / artifacts
- Box worker reverted to #12 state (`3960ccb5`); `OVS_ASR_STREAM_PREFIX` symbols removed; #12 paths + `runOneShotCore` default intact; default path byte-identical to deployed; no container/image/runtime touched.
- Exploratory diff (record only): `engine-overlay/patches/v080-0025-prefix-cap-BROKEN.patch` (239 lines).
- Measurement baseline this was tested against: `v080-0024-...CLOSEOUT.md` table + `~/asr_perf_measure/` on box.
