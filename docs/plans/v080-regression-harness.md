# v0.8.0 Migration Regression Harness

Status: DESIGN (codex), main-thread verified 2026-06-09 — confirmed `bench/perf/{gate,perf,stats,asr_stream_ws_bench,stability_tts_n2_common,load_2client_tts,paraformer_direct_stream_bench}.py` + `bench/perf/corpus/manifest.json` all exist in seeed-local-voice. This harness pins v0.7.1 voice-engine behavior as GOLDEN so the v0.8.0 migration (see `asr-streaming-v080-migration.md`) can be proven NON-REGRESSING. Every migration phase must pass its mapped gates (§8).

> **Dependency note:** the harness reuses `bench/perf/asr_stream_ws_bench.py` and `paraformer_direct_stream_bench.py`, which are currently UNTRACKED WIP on `feat/voice-rebot-arm` (not on main). Before harness code lands they must be committed/merged onto the harness branch base, or vendored into `bench/regression/`.

## 0. Reuse existing infra (don't reinvent)
`bench/perf/` is the outer runner: fixed corpus + SHA256 manifest (`corpus/manifest.json`), unified result JSON/MD (`stats.py:179-196`), per-device attribution `meta.client_host` (`perf.py:31-43`), regression gate keyed by `client_host` (`gate.py:20-21,78-87`), hard tolerances +25% latency/RTF, +0.10 abs error rate, -10% similarity (`gate.py:39-43`). ASR: `asr_stream_ws_bench.py` (zh/en normalize, error rate, `eos_to_final_ms`), `paraformer_direct_stream_bench.py` (`finalize_ms`). TTS/N=2: `load_2client_tts.py` (N=2 TTFA p50 ≤1.5× spec), `stability_tts_n2_common.py` (N=1 baseline, 30+ bursts, pre/post MD5, CUDA log scan), MOSS `smoke_moss_tts_backend.py` / `stress_moss_tts_n2.py`.

**New files:**
```
bench/regression/
  test_asr_streaming_correctness.py
  test_tts_correctness.py
  test_tts_n2_slotpool.py
  test_build_abi_sanity.py
  run_v080_regression.py            # capture + check runner
  goldens/v071/{asr,tts/customvoice,tts/moss,build}/...
```

## 1. ASR streaming correctness
GOLDEN: 20 WAVs from `corpus/manifest.json` (short/long × zh/en), SHA256-pinned, expected transcripts from manifest, stored `goldens/v071/asr/*.json`. Engine invariants (named test cases from `asr-streaming-v080-migration.md` §4 Phase 5):
- **R2** single-chunk vs split-chunk: argmax match, max abs diff `<1e-2` (scaffold `spike_m1_append_prefill_embeds.cpp`)
- **R3** MRope position continuity: byte-stable untouched lanes/positions, no tolerance
- **KV-overflow**: worker rejects with error, no silent advance (scaffold `spike_m2_session_lifecycle.cpp`)
- **R4** sys-prompt cache mismatch fallback: cached-KV path byte-equal to uncached full prefill
COMPARE: transcripts semantic (reuse `asr_stream_ws_bench.py` normalize); error rate passes at `golden+0.10`. R2 fp tol 1e-2. R3/KV-overflow no tol.
GATE: hard fail on R2 argmax mismatch / R3 overwrite / KV silent advance / R4 divergence. Hardware ASR roundtrip required (`DIVERGENCE.md:44-52`).

## 2. TTS correctness
GOLDEN: 12 prompts (zh/en short/long); CustomVoice with `language=chinese|english` (9-row prefix, patch 0005); MOSS 3 zh prompts (`DIVERGENCE.md:131-138`). Fields: PCM MD5, sr/channels, duration, RMS+peak energy, ASR-roundtrip transcript → `goldens/v071/tts/`.
COMPARE: **byte-non-empty is NEVER sufficient** (project history: empty-valid audio = silence). Energy + ASR-roundtrip mandatory. MD5 exact for deterministic engines; energy ±25%; roundtrip semantic `golden+0.10`.
GATE: hard fail on MD5 drift (deterministic prompts), silence (RMS <10% golden), roundtrip >golden+0.10, or CustomVoice wrong-language transcript.

## 3. Concurrency N=2 slot-pool (the moat — must survive Section 6 v2 batch-lane redesign)
GOLDEN (`DIVERGENCE.md:54-62`): N=1 PCM MD5/prompt, N=2 dual-client MD5s, TTFA p50 N=1 & N=2, CUDA error count, burst count.
COMPARE: `test_tts_n2_slotpool.py` wraps `stability_tts_n2_common.py` (service N=2) + `stress_moss_tts_n2.py --mode parity --mode burst --rounds 30` (MOSS). Observable moat unchanged whether TTS uses native batch-lane or slot replication.
GATE: hard fail if any of 30 bursts fails / any CUDA error / any MD5 differs N=2 vs N=1 / TTFA ratio >1.5× (`load_2client_tts.py`).

## 4. Perf gates
GOLDEN: `bench/perf/results/*.json` via `perf.py`. Gated: ASR `finalize_rtf`/`error_rate`/`eos_to_final_ms`; TTS `rtf`/`tfd_ms`/`total_ms`; v2v `asr_finalize_ms`; concurrent `rtf`/`tfd_ms` (`gate.py:52-60`).
COMPARE: `gate.py check bench/perf/results --baseline bench/perf/baselines/baseline.json --strict`, keyed by `client_host`.
GATE: ×1.25 latency/RTF, `base+0.10` error rate (`gate.py:118-127`); exit 1 on FAIL or strict NO-BASE. Memory peak / energy tracked-only.

## 5. Build / ABI sanity
GOLDEN: engine MD5s (ASR thinker/audio enc, TTS Talker/CodePredictor/Code2Wav, MOSS worker bin) → `goldens/v071/build/engine_md5s.json`.
GATE: MD5 exact (deterministic build); `docker logs|grep -iE 'error|crash|fail'` + CUDA regex (`stability_tts_n2_common.py:50-60`) = 0. Hard fail on MD5 drift w/o refresh note or missing worker binary.

## 6. Golden capture procedure
1. `python bench/perf/corpus/fetch.py --verify`
2. On known-good v0.7.1 on-device: `bench/perf/run_on_device.sh <node> -- matrix` (local mode, no latency mixing)
3. `python bench/regression/run_v080_regression.py --capture --out bench/regression/goldens/v071/`
4. Record engine MD5s + `meta.json` (device, git SHA, date)
5. Refresh only via `--accept-goldens --note "<rationale>"` (records MD5/SHA/device/date). No silent updates.

## 7. CI runner
```
python bench/regression/run_v080_regression.py --base-url http://localhost:8621 \
  --container <name> --device <client_host> --golden bench/regression/goldens/v071/
```
Order: (1) build/ABI fail-fast → (2) ASR R2/R3/R4/KV-overflow → (3) TTS CustomVoice zh/en + MOSS → (4) N=2 30-burst+CUDA+TTFA → (5) `perf.py` + `gate.py --strict`. Exit 0 only if all 5 pass.

## 8. Phase-gate mapping
| Phase | Gates that must pass |
|---|---|
| 1 Scaffolding | build/ABI compile; M1/M2 spike compile; no perf gate |
| 2 Audio chunk API | R3 MRope continuity; M3.5/M3.6 audio split LCS ≥0.95 |
| 3 Runtime hooks | R2 single-vs-split argmax/logit; lifecycle reset; KV-overflow refusal; R4 fallback |
| 4 Cache/scheduler | N=2 lane isolation; ASR/TTS shared-lane reset; 30-burst CUDA gate |
| 5 Full validation | ALL: ASR corpus, TTS MD5+energy+roundtrip, N=2 TTFA ≤1.5×, gate.py --strict, engine MD5, docker grep |
