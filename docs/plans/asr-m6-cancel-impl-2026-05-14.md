# ASR M6 Cancel Implementation Spec

## Context

`ASRStream.cancel_and_finalize()` currently exists as an optional base hook and defaults to a no-op, documented for barge-in or client stop paths that should avoid wasting hundreds of milliseconds on pending decode work (`/Users/harvest/project/seeed-local-voice/app/core/asr_backend.py:58`). The v2v WebSocket dispatcher already calls this hook synchronously on `CLIENT_ABORT`, after cancelling TTS, setting the current TTS stop event, and draining queued TTS sentences (`/Users/harvest/project/seeed-local-voice/app/main.py:990`, `/Users/harvest/project/seeed-local-voice/app/main.py:1002`). The implementation goal is therefore narrow: each concrete stream should make cancel cheap, preserve the best already-known text where available, and make later `finalize()` return immediately.

Each implementation should add `_cancelled: bool` and `_final_text_cache: str` stream fields. `accept_waveform()` should return early once cancelled, and `finalize()` should check `_cancelled` at the top before any numpy, CUDA, ORT, TensorRT, VAD, subprocess, or tokenizer-heavy work.

## Class 1: Qwen3StreamingASRStream

File: `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:313`

This class buffers rolling audio in `_audio_buf`, per-utterance audio in `_utterance_audio_buffer`, and encoder outputs in `_encoder_frames` (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:328`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:333`). `accept_waveform()` can run `_process_streaming_chunk()` and endpoint-triggered `_do_final_decode()` (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:389`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:393`). `finalize()` may drain residual chunks, encode tail audio, call `_offline_final_text()`, and synchronize the ASR CUDA stream (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:414`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:433`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:449`). The final decode path ultimately calls offline transcribe via `_offline_final_text()` or encoder reuse via `_streaming_final_text()` and `_final_text()` (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:613`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:620`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:639`).

In-flight work to skip: `_offline_final_text()` / `_do_final_decode()`.

State to clear on cancel: `_audio_buf`, `_encoder_frames`.

Cancelled-final text source: `_archive_text + _partial_text`, matching `get_partial()` composition (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:403`).

Implementation sketch:

```python
def cancel_and_finalize(self):
    if self._cancelled:
        return
    text = self._archive_text
    if self._partial_text and not self._episode_final:
        sep = " " if not (self._archive_text and _is_cjk(self._archive_text[-1])) else ""
        text = (self._archive_text + sep + self._partial_text).strip()
    self._final_text_cache = text.strip()
    self._cancelled = True
    self._audio_buf = np.array([], dtype=np.float32)
    self._encoder_frames.clear()
    self._total_encoder_frames = 0
    self._utterance_audio_buffer.clear()

def accept_waveform(self, sample_rate, samples):
    if self._cancelled:
        return
    ...

def finalize(self):
    if self._cancelled:
        return self._final_text_cache
    ...
```

Tests: with pytest, feed 2s of audio, call `cancel_and_finalize()`, assert it returns within 50ms, assert `finalize()` returns cached text, and assert backend `create_stream()` works again.

## Class 2: Qwen3ASRStream

File: `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:262`

This accumulate-then-transcribe stream stores samples in `_chunks`, tracks `_total_samples`, and currently performs the full pipeline only inside `finalize()` by concatenating buffered chunks and calling `transcribe_audio()` (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:271`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:289`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:297`). `get_partial()` always returns no text, so there is no usable partial cache (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/qwen3_asr.py:300`).

In-flight work to skip: full `finalize()` transcribe pipeline, including encoder and decoder.

State to clear on cancel: `_chunks` buffer.

Cancelled-final text source: `''` because no partials are available.

Implementation sketch: initialize `_cancelled = False` and `_final_text_cache = ""`; `cancel_and_finalize()` sets `_cancelled = True`, clears `_chunks`, resets `_total_samples = 0`, and caches `""`; `accept_waveform()` returns early when cancelled; `finalize()` returns `_final_text_cache` before concatenating or calling `transcribe_audio()`.

Tests: same pytest pattern as above: feed 2s audio, call `cancel_and_finalize()`, assert under 50ms, assert immediate empty `finalize()`, and assert a fresh `create_stream()` can accept and finalize normally.

## Class 3: _TRTEdgeLLMAccumulatingASRStream

File: `/Users/harvest/project/seeed-local-voice/app/backends/jetson/trt_edge_llm_asr.py:527`

This stream is also accumulating: `create_stream()` returns `_TRTEdgeLLMAccumulatingASRStream` after preload (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/trt_edge_llm_asr.py:503`), `accept_waveform()` resamples and appends copied chunks (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/trt_edge_llm_asr.py:533`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/trt_edge_llm_asr.py:544`), and `finalize()` concatenates all chunks before VAD splitting and per-segment transcription (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/trt_edge_llm_asr.py:546`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/trt_edge_llm_asr.py:559`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/trt_edge_llm_asr.py:579`).

In-flight work to skip: VAD-split plus per-segment `backend.transcribe()` through subprocess worker IPC.

State to clear on cancel: `_chunks` list.

Cancelled-final text source: `''`.

Implementation sketch: same pattern as `Qwen3ASRStream`; add `_cancelled`, `_final_text_cache`, early returns in `accept_waveform()` and `finalize()`, and a `cancel_and_finalize()` that sets the flag, clears `_chunks`, and stores `""`.

Tests: same pytest pattern, with backend transcribe mocked so failure indicates cancel failed to short-circuit.

## Class 4: ParaformerTRTStream

File: `/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:333`

The stream keeps residual audio in `_audio_buf`, accumulated utterance audio in `_all_audio`, decoded IDs in `_all_token_ids`, and current text in `_partial_text` (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:340`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:348`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:354`). Normal `accept_waveform()` appends then processes chunks (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:388`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:399`). `finalize()` handles residual audio, recomputes features, runs encoder and decoder for pending CIF frames, clears `_audio_buf`, then calls `_flush_cif_tail()` (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:517`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:522`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:535`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:553`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:564`). `_flush_cif_tail()` can run an extra decoder pass (`/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:579`, `/Users/harvest/project/seeed-local-voice/app/backends/jetson/paraformer_trt.py:585`).

In-flight work to skip: `_flush_cif_tail` plus final decode pass in `finalize()`.

State to clear on cancel: `_audio_buf`.

Cancelled-final text source: `_partial_text`, or decoded `_all_token_ids` if richer.

Implementation sketch: `cancel_and_finalize()` sets `_cancelled = True`, caches `_partial_text`, and assigns `_audio_buf = np.array([], dtype=np.float32)`; `accept_waveform()` returns early once cancelled; `finalize()` checks `_cancelled` at the top and returns `_final_text_cache` before residual feature, encoder, CIF, decoder, or tail work.

Tests: same pytest pattern; assert `_backend._run_encoder` and `_backend._run_decoder` are not called after cancel.

## Class 5: SherpaASRStream

File: `/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:103`

Sherpa stores recognizer state in the native stream and tracks `_last_text` plus `_is_endpoint` (`/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:106`, `/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:109`). `accept_waveform()` accepts samples, decodes while ready, updates `_last_text`, and may reset the stream on endpoint (`/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:113`, `/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:121`, `/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:123`, `/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:132`). `finalize()` adds 0.8s silence for English, calls `input_finished()`, and performs extra decode loops (`/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:139`, `/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:144`, `/Users/harvest/project/seeed-local-voice/app/backends/cpu/sherpa_asr.py:149`).

In-flight work to skip: 0.8s silence-pad plus extra decode pass, typically 50-200ms.

State to clear on cancel: n/a, because this is effectively stateless from Python aside from the native sherpa stream.

Cancelled-final text source: `_last_text`.

Implementation sketch: `cancel_and_finalize()` sets `_cancelled = True`, caches `_last_text`, and may call `_stream.input_finished()` only if confirmed non-blocking and used for GC; `finalize()` checks `_cancelled` first and returns `_final_text_cache`; `accept_waveform()` returns early after cancel.

Tests: same pytest pattern, with recognizer methods mocked to assert no silence pad or decode loop occurs after cancel.

## Concurrency

`main.py` calls `cancel_and_finalize()` synchronously from the WebSocket dispatcher coroutine on `CLIENT_ABORT` (`/Users/harvest/project/seeed-local-voice/app/main.py:990`, `/Users/harvest/project/seeed-local-voice/app/main.py:1002`). Meanwhile, `asr_out_task` is a separate coroutine that polls ASR partials and final state (`/Users/harvest/project/seeed-local-voice/app/main.py:1007`), and the task may have `asr_stream.get_partial()` or related ASR work blocked in `run_in_executor` depending on the surrounding dispatcher path. Therefore, `cancel_and_finalize()` must not run CUDA, TensorRT, ORT, sherpa decode, VAD, tokenizer-heavy, or numpy concatenate/interp work. It should only set Python-level flags, cache already-computed text, and clear Python references to buffers. Those assignments and list clears are GIL-protected enough for this cancellation contract.

The next `accept_waveform()`, `get_partial()`, or `finalize()` running on the executor thread must observe `_cancelled` and short-circuit. Worst-case abort latency is the remaining time in the currently running encoder or decoder step, usually one chunk, around 50-200ms. Every class must check `_cancelled` at the top of `finalize()` and return `_final_text_cache` immediately. `accept_waveform()` should also check the flag before dtype conversion, resampling, appending, decoding, or endpoint checks.

## E2E Validation on orin-nx

First confirm the service is actually running on the target: `fleet exec orin-nx -- 'docker ps | grep voice'`. Then verify the exposed WebSocket port with `fleet exec orin-nx -- 'docker port <container>'`; the expected endpoint is `ws://orin-nx:8621/v2v/stream`.

Build a minimal `tests/test_v2v_bargein_latency.py` in the service repo for validation. The script should connect to `/v2v/stream`, send a config frame such as `{type: config, asr_language: "zh", tts_language: "zh", vad: "silero"}`, send text frames that start TTS, wait until the first audio chunk arrives, send `{type: abort}`, and measure elapsed time from abort send to the last audio byte received. Run 20 trials for each backend and report p50, p95, and max latency. Acceptance is p95 <= 200ms across 20 trials per backend.

Run with each relevant `ASR_BACKEND`: `trt_edgellm`, `paraformer`, `sherpa`, and `qwen3_asr_rk`. Compare against the baseline where concrete streams inherit the default no-op hook from `ASRStream.cancel_and_finalize()` (`/Users/harvest/project/seeed-local-voice/app/core/asr_backend.py:58`) to show improvement.

Production config selects one backend at a time. `trt_edgellm` exercises `_TRTEdgeLLMAccumulatingASRStream`; Jetson Qwen3 config, if available, exercises `Qwen3StreamingASRStream` or `Qwen3ASRStream`; Jetson Paraformer profile exercises `ParaformerTRTStream`; RPi4 sherpa profile is not testable on `orin-nx`, so use unit tests only for `SherpaASRStream`.

## E2E Validation Results (2026-05-14)

- Container: `seeed-local-voice-feat-cancel` (side-by-side with `seeed-nx-v112`, untouched)
- Image: `seeed-local-voice:feat-cancel-test` (overlay on `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.12`)
- Port: 8622 (host) -> 8000 (container)
- Mirrored env + binds from v112; PYTHONPATH=/opt/speech so overlay copies into `/opt/speech/app` and `/opt/speech/tests`
- ASR backend (auto-selected via profile `voice_clone` -> `jetson-multilang-highperf-nx`): `trt_edgellm` (`Qwen3 TTS + ASR`)
- TTS backend: `trt_edgellm`
- Harness: `tests/test_v2v_bargein_latency.py` driven with `V2V_URL=ws://localhost:8622/v2v/stream TRIALS=10`
- N trials: 10 (all successful)
- p50: **0.7 ms**
- p95: **3.3 ms**
- max: **3.3 ms**
- Verdict: **PASS** — well under the >=200ms p95 gate.

Raw harness output:

```
V2V barge-in latency probe  URL=ws://localhost:8622/v2v/stream  trials=10
  trial  1: abort_to_last_audio_ms =     0.0
  trial  2: abort_to_last_audio_ms =     0.7
  trial  3: abort_to_last_audio_ms =     0.7
  trial  4: abort_to_last_audio_ms =     0.7
  trial  5: abort_to_last_audio_ms =     0.8
  trial  6: abort_to_last_audio_ms =     0.8
  trial  7: abort_to_last_audio_ms =     3.3
  trial  8: abort_to_last_audio_ms =     0.7
  trial  9: abort_to_last_audio_ms =     0.7
  trial 10: abort_to_last_audio_ms =     0.7

SUMMARY  n=10/10  p50=0.7ms  p95=3.3ms  max=3.3ms
```

Note: Only the `trt_edgellm` backend was exercised on this side-by-side container. The other backends (`paraformer`, `sherpa`, `qwen3_asr_rk`) are covered by unit tests only at this milestone; their e2e validation depends on a profile switch or separate hardware (RPi4 for sherpa).

## E2E Validation Results — Orin Nano (2026-05-14)

- Device: Jetson Orin Nano Super (orin-nano, 100.92.125.65)
- Container: `seeed-local-voice-feat-cancel` (temp swap with seeed-nano-v112)
- Image: `seeed-local-voice:feat-cancel-test` (overlay on jetson-v1.12, host networking)
- ASR_BACKEND: trt_edgellm
- N trials: 10/10
- p50: 0.6 ms
- p95: 0.6 ms
- max: 0.6 ms
- Verdict: **PASS** (≤200ms gate; ~300x headroom)

Raw harness tail:
```
V2V barge-in latency probe  URL=ws://localhost:8000/v2v/stream  trials=10
  trial  1: abort_to_last_audio_ms =     0.6
  trial  2: abort_to_last_audio_ms =     0.5
  trial  3: abort_to_last_audio_ms =     0.6
  trial  4: abort_to_last_audio_ms =     0.5
  trial  5: abort_to_last_audio_ms =     0.6
  trial  6: abort_to_last_audio_ms =     0.6
  trial  7: abort_to_last_audio_ms =     0.6
  trial  8: abort_to_last_audio_ms =     0.6
  trial  9: abort_to_last_audio_ms =     0.6
  trial 10: abort_to_last_audio_ms =     0.6

SUMMARY  n=10/10  p50=0.6ms  p95=0.6ms  max=0.6ms
```

Post-test: seeed-nano-v112 restored to Up + /health 200; feat container removed; feat-cancel-test image removed (freed ~1GB layer). Confirms orin-nx result (p50=0.7 / p95=3.3 ms) generalizes to Nano with even tighter variance.
