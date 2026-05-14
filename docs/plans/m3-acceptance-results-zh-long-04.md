# M3 Step 5 — Streaming Worker Acceptance Test Results

Worker: `/opt/jv-workers/qwen3_asr_worker`
Short WAV: `docs/audio-evidence/nx-loopback-pass-p1-2026-05-11.wav` (2.64s clipped)
Long WAV: `docs/audio-evidence/zh-long-04-2026-05-13.wav` — 12.90s

## Gate summary

| Gate | Result | Kind |
|------|--------|------|
| A_oneshot_ok | PASS | hard |
| B_lcs_ge_0.95 | PASS | hard |
| C_median_le_500ms | PASS | hard |
| C_p95_le_1000ms | PASS | hard |
| D_one_final | PASS | hard |
| D_at_least_one_segment_rotation | PASS | hard |
| D_lcs_ge_0.90_soft | FAIL | soft |
| E_malformed_json_handled | PASS | hard |
| E_unknown_event_handled | PASS | hard |
| E_chunk_too_long_handled | PASS | hard |
| E_session_cleared_after_error | PASS | hard |
| F_pcm_lcs_ge_0.95 | PASS | hard |

**Hard gates: PASS** | All gates (incl. soft): FAIL

## A — one-shot baseline

- ok: True
- text: `今天天气真好。`
- response keys: ['event', 'id', 'ok', 'responses', 'total_ms']

## B — streaming happy path

- text: `今天天气真好。`
- LCS vs baseline: 1.000
- end-of-speech latency (1 run): 157.0 ms
- rotations during B (expected 0): 0

## C — end-of-speech latency (5 runs)

- runs (ms): [157.7]
- median: 157.7 ms
- p95:    157.7 ms
- min/max: 157.7 / 157.7 ms

## D — auto-segmentation

- duration: 12.90 s
- rotations: 1
- segment_count: 2
- text: `科学家们可以得出结论：暗物质对其他暗物质的影响。其他物质的影响方式与普通物质相同。`
- curated ground truth: `科学家们可以得出结论，暗物质对其他暗物质的影响方式与普通物质相同。`
- LCS vs curated GT: 0.780
- LCS vs baseline_x2: 0.049
- LCS vs baseline_x1: 0.024
- LCS best: 0.780

## E — error paths

- **malformed_json**: PASS — got `{'error': "json_parse_failed: [json.exception.parse_error.101] parse error at line 1, column 3: syntax error while parsing object key - invalid literal; last read: '{th'; expected string literal", 'event': 'error', 'ok': False}`
- **unknown_event**: PASS — got `{'error': 'unknown_event', 'event': 'error', 'id': 'e2', 'ok': False, 'received': 'unknown_thing'}`
- **chunk_too_long**: PASS — got `{'chunk_sec': 8.0, 'error': 'chunk_too_long', 'event': 'error', 'id': 'e3', 'limit_sec': 7.0, 'ok': False}`
- **session_cleared_after_error**: PASS — got `{'error': 'no_active_session', 'event': 'error', 'id': 'e3', 'ok': False}`
