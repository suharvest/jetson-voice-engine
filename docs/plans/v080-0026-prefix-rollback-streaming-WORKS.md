# v080-0026 — Prefix-continuation + chunk-and-confirm rollback streaming: WORKS (flat decode, 3–6× faster, restores v0.7.x behavior)

Status: **VALIDATED OFFLINE on orin-nx (G-correct + G-perf both PASS).** This is the CORRECT outcome of the streaming-latency investigation and SUPERSEDES the wrong "fundamentally broken / no free lunch" conclusion in `v080-0025-prefix-cap-decode-FINDING.md` (that was a PORT BUG — the first attempt dropped the rollback). User instinct ("v0.7.x fixed-decode had no accuracy problem → we must have a bug") was correct.

## What it is
The deployed v0.8.0 #13 streaming (`runOneShotCore` cumulative re-decode) re-decodes the WHOLE transcript every hop → decode dominates and grows O(audio_so_far) (#14: total 98→784 ms, decode 69→727 ms over 1–15.4 s). The v0.7.x worker avoided this via **assistant-prefix continuation + chunk-and-confirm rollback**, which the v0.8.0 "first-cut" DROPPED → a real regression for medium/long utterances. This ports it back.

## The mechanism (faithful port of v0.7.x `asr-worker-build-verify`)
- `unfixedChunkNum=2`: first 2 hops use NO prefix (full re-decode; early audio unstable).
- `unfixedTokenNum=5`: each hop rolls back the last 5 tokens of the accumulated transcript (treats them UNCONFIRMED) — the confirmed prefix = `tok.decode(tok.encode(rawDecoded)[:-5])`, with a U+FFFD guard (don't split a multibyte char; roll back one more token if it appears).
- Per hop: `prefix = computePrefix(...)` → request `{user: full cumulative audio, assistant: kAssistantGenPrompt + "language X<asr_text>" + prefix}`, `apply_chat_template=true, add_generation_prompt=false, max_generate_length=64` → generates only the CONTINUATION → `rawDecoded = prefix + generated`.
- The premature `。`+EOS / hallucination the model emits on PARTIAL audio lands in the last-5-token unconfirmed window → rolled back → re-decoded next hop with more audio → **corrected, never permanently committed**. That is why it is both flat AND correct: revision is LOCAL (only the recent tail is unstable; old words are stable).
- Worker-side `gPrefixTokenizer` loaded for the encode/decode rollback; runtime decode/eviction untouched; default `runOneShotCore` + #12 paths untouched; gated by `OVS_ASR_STREAM_PREFIX=1` (default OFF).
- Feasibility was GREEN: v0.8.0 vanilla `rt::LLMInferenceRuntime` does assistant-prefix continuation correctly (no "Method B" runtime change needed).

## G-correct (PASS) — main-thread-run against the rollback binary (md5 5fc720e9)
```
[zh_long_01] CER 0.0000  (byte-exact)
[zh_long_02] CER 0.0250  (catastrophic hallucination collapse GONE; partial[0] '《中国共产党历史》。' → rolled back → partial[1] '桥下垂直净空十五米…' → correct final)
[zh_long_03] CER 0.0115
```
Residual CER is punctuation only (，vs。 at hop boundaries, one missing space) — the confirmed-prefix-can't-revise-beyond-window residual that v0.7.x accepted. The broken (no-rollback) version was CER 0.08–0.975.

## G-perf (PASS) — decode is now FLAT (zh_long_03, median of reps 2–3)
```
            DEPLOYED cumulative (#14)        ROLLBACK prefix (this)
cum_sec | t_prefill t_decode t_total |  t_prefill t_decode t_total  gen_chars/hop
  1.0   |    29       69       98    |    28.5     68.7     97.3        2
  3.0   |    32      151      183    |    31.3     96.3    127.5        8
  6.0   |    36      290      326    |    36.2    110.3    146.5        7
 10.0   |    44      487      531    |    47.0    125.5    172.5        7
 15.4   |    57      727      784    |    58.2     71.1    129.3        0(EOS)
```
- **t_decode FLAT/bounded 68–125 ms** vs deployed 69→727 ms (10.5× growth). gen_chars/hop bounded 2–8 (rollback window + new words, never the whole transcript).
- Per-hop total latency improvement: **3.1× at 10 s, 6.1× at 15.4 s** (widens with length). Decode: 3.9× at 10 s, 10.2× at 15.4 s.
- t_prefill grows mildly ~2× (prefills cumulative audio + confirmed-prefix text) — the only growing component; rollback does NOT optimize prefill (that's #12's separate <5% concern). **Internal consistency check: rollback t_prefill (28.5→58.2) ≈ deployed t_prefill (29→57) — identical as expected (same cumulative-audio prefill), validating the measurement.**

## Net
- Short commands (≤3 s, main seeed workload): ~same latency as deployed (both <200 ms), CER ≈0 — no perceived change.
- Medium/long (≥6 s): **3–6× faster per-hop partials** at negligible accuracy cost. Restores the v0.7.x streaming UX that v0.8.0 #13 regressed.
- **Refinement for byte-exact final**: keep cheap rollback partials, but finalize with ONE clean `runOneShotCore` re-decode → byte-exact final (eliminates the CER 0.01–0.025 residual) while partials stay flat. Strictly ≥ deployed (same final accuracy, 3–6× cheaper partials). Recommended if shipped.

## Cost / footprint
Worker-side only (~393-line additive patch `v080-0026-prefix-rollback-WORKS.patch`), default OFF, runtime/engines/#12 untouched. Like #12's worker part, it's OUR addon → zero upstream-rebase cost. Validated binary md5 `5fc720e9`.

## Durable artifacts
- `engine-overlay/patches/v080-0026-prefix-rollback-WORKS.patch` (393 lines, vs clean #12 baseline).
- `_v080_prefix_rollback_WORKS.tar.gz` (overlay, md5 `128402370524c302fa6baa44a122627f`) = working source + g_correct.py + patch.
- Box: `~/asr_prefix_cap/` (gperf.py, g_correct.py, gperf_rows.jsonl 170 rows, mels/) + worker bin `5fc720e9`.

## Productionized (2026-06-11, user chose: ship WITH byte-exact-final)

**Stage 1 — byte-exact final (offline, worker `5ddbcdf7`):** in `OVS_ASR_STREAM_PREFIX` mode the FINAL event now routes to a clean `runOneShotCore` full re-decode (golden path, gated `OVS_ASR_STREAM_PREFIX_FINAL_ONESHOT=1`), so final == deployed one-shot byte-exact (CER **0.0000 ×3**, was 0.0000/0.025/0.0115); partials stay the cheap flat rollback; default + #12 untouched. Patch `v080-0027-prefix-rollback-byteexact-final.patch` (419 lines) + `_v080_prefix_rollback_byteexact_DEPLOY.tar.gz`.

**Stage 2 — deploy + through-service verify (orin-nx, NOT real prod seeed-orin-nx):**
- Image **`:v0.8.0-edgellm-20260611-prefix`** built by overlay onto prod base; baked ASR worker md5 `5ddbcdf7` confirmed inside image; env `OVS_ASR_STREAM_PREFIX=1 + OVS_ASR_STREAM_PREFIX_FINAL_ONESHOT=1` + rollback params (`stream_unfixed_chunks=2/tokens=5`) wired.
- **Through-service gates GREEN** (main-thread-run, throwaway container :8649): G1 short-clip prefix-final == offline one-shot byte-exact; G3 N>1 = 2 concurrent isolated + 3rd→4429 (`session_limiter: WS 4429 active=2 limit=2`); G4 logs clean, no SIGABRT, worker survives. (Long-clip byte-exact already proven worker-direct Stage 1; through-service long "mismatch" was a driver/segmentation artifact. Partial flatness proven worker-direct G-perf; partials observed streaming through service.)
- **Registry: PUSHED** — `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.8.0-edgellm-20260611-prefix`, digest `sha256:354df1eb1642751de37542ad94f1a733d094dd471de2ca5f951ab83f849ebf25`. **Deployable.**
- **HF worker artifact: DONE** — uploaded `…/v0.8.0/workers/qwen3_asr_worker` (via wsl2-local relay, proxy recovered), download-back md5-verified == `5ddbcdf7` (size 14623704), MANIFEST (v080-0017 line 83) updated `be7bee91…` → `5ddbcdf7…`.

**Real production seeed-orin-nx NOT flipped** — deliberate separate runbook step, per migration discipline. Default behavior (flags unset) is byte-identical to deployed; this image is opt-in via the env flags.
