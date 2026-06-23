#!/usr/bin/env python3
"""Measure streaming TTS TTFT and RTF via HTTP /tts/stream endpoint.

The /tts/stream endpoint streams raw PCM:
  - First 4 bytes: sample_rate as uint32 LE
  - Remaining bytes: int16 PCM chunks as they are generated

TTFT = time from request sent to when the first PCM audio byte (after the
       4-byte header) arrives.  This is what a user hears: first audio chunk.
RTF  = total_synthesis_time / audio_duration.

Usage
-----
python bench/bench_http_tts_stream.py --url http://device:8621 --repeat 3
python bench/bench_http_tts_stream.py --url http://192.168.3.x:8621 \
    --repeat 5 --output-jsonl results.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import struct
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("bench_http_tts_stream")

DEFAULT_TEXTS = [
    "你好，世界！",
    "今天天气怎么样？",
    "请告诉我最近的新闻。",
    "Hello, how are you today?",
    "欢迎使用语音合成服务，希望您有一个愉快的体验。",
    "TensorRT accelerates deep learning inference on NVIDIA Jetson Orin.",
]


def _stream_tts_once(
    url: str,
    text: str,
    api_key: Optional[str] = None,
    language: str = "auto",
    timeout_s: float = 60.0,
) -> tuple[float, float, float, int]:
    """POST /tts/stream; return (ttft_ms, total_ms, rtf, pcm_bytes_count).

    Uses requests with stream=True so chunked-transfer is decoded automatically
    and we can timestamp the exact moment first PCM data arrives.
    """
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {"text": text, "language": language}

    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/tts/stream",
        json=payload,
        headers=headers,
        stream=True,
        timeout=timeout_s,
    )
    resp.raise_for_status()

    # Accumulate body; detect first PCM byte (after 4-byte sample-rate header)
    buf = b""
    t_first_audio: Optional[float] = None

    for chunk in resp.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buf += chunk
        if t_first_audio is None and len(buf) > 4:
            # bytes 0-3 = sample_rate header; byte 4+ = first PCM sample
            t_first_audio = time.perf_counter()

    t_done = time.perf_counter()

    if len(buf) < 4:
        raise RuntimeError(f"Response too short ({len(buf)} bytes) — not raw PCM?")

    sample_rate = struct.unpack_from("<I", buf, 0)[0]
    if sample_rate == 0 or sample_rate > 96000:
        raise RuntimeError(f"Unexpected sample_rate={sample_rate} — endpoint may not be /tts/stream")

    pcm_bytes = buf[4:]
    n_samples = len(pcm_bytes) // 2  # int16
    audio_duration_s = n_samples / sample_rate

    ttft_ms = (t_first_audio - t0) * 1000 if t_first_audio else (t_done - t0) * 1000
    total_ms = (t_done - t0) * 1000
    rtf = (t_done - t0) / audio_duration_s if audio_duration_s > 0 else float("nan")

    return ttft_ms, total_ms, rtf, len(pcm_bytes)


def _bench_one(
    url: str,
    text: str,
    repeat: int,
    warmup: int,
    api_key: Optional[str],
) -> dict:
    for i in range(warmup):
        try:
            _stream_tts_once(url, text, api_key)
        except Exception as exc:
            logger.warning("Warmup %d failed: %s", i + 1, exc)

    ttft_list, total_list, rtf_list = [], [], []
    for trial in range(repeat):
        try:
            ttft, total, rtf, n_bytes = _stream_tts_once(url, text, api_key)
            ttft_list.append(ttft)
            total_list.append(total)
            rtf_list.append(rtf)
            logger.info(
                "trial %d/%d  %-24s  ttft=%5.0fms  total=%5.0fms  rtf=%.3f  pcm=%dB",
                trial + 1, repeat, repr(text[:20]), ttft, total, rtf, n_bytes,
            )
        except Exception as exc:
            logger.error("trial %d failed: %s", trial + 1, exc)
            ttft_list.append(float("nan"))
            total_list.append(float("nan"))
            rtf_list.append(float("nan"))

    vt = [x for x in ttft_list if x == x]
    vto = [x for x in total_list if x == x]
    vr = [x for x in rtf_list if x == x]

    return {
        "text": text,
        "repeat": repeat,
        "ttft_ms": ttft_list,
        "total_ms": total_list,
        "rtf": rtf_list,
        "summary": {
            "ttft_mean_ms": round(statistics.mean(vt), 1) if vt else None,
            "ttft_min_ms": round(min(vt), 1) if vt else None,
            "total_mean_ms": round(statistics.mean(vto), 1) if vto else None,
            "rtf_mean": round(statistics.mean(vr), 3) if vr else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HTTP /tts/stream TTFT + RTF benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", default="http://localhost:8621",
                        help="Service base URL (default: http://localhost:8621)")
    parser.add_argument("--text", action="append", dest="texts",
                        help="Text to synthesize (repeatable). Default: 6 built-in sentences.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()

    texts = args.texts or DEFAULT_TEXTS

    # Health check
    try:
        import requests
        r = requests.get(f"{args.url}/health", timeout=5)
        logger.info("Health %s → %s", args.url, r.json())
    except Exception as exc:
        logger.warning("Health check failed (%s) — proceeding anyway", exc)

    all_results = []
    for text in texts:
        result = _bench_one(args.url, text, args.repeat, args.warmup, args.api_key)
        all_results.append(result)
        if args.output_jsonl:
            with open(args.output_jsonl, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print()
    print("=" * 78)
    print(f"  bench_http_tts_stream  url={args.url}  repeat={args.repeat}")
    print("=" * 78)
    print(f"  {'text':<32}  {'ttft_mean':>10}  {'ttft_min':>9}  {'total_mean':>11}  {'rtf':>6}")
    print("-" * 78)
    for r in all_results:
        s = r["summary"]
        tm = f"{round(s['ttft_mean_ms'])} ms" if s.get("ttft_mean_ms") else "N/A"
        ti = f"{round(s['ttft_min_ms'])} ms" if s.get("ttft_min_ms") else "N/A"
        to = f"{round(s['total_mean_ms'])} ms" if s.get("total_mean_ms") else "N/A"
        rv = f"{s['rtf_mean']:.3f}" if s.get("rtf_mean") else "N/A"
        print(f"  {r['text'][:32]:<32}  {tm:>10}  {ti:>9}  {to:>11}  {rv:>6}")
    print("=" * 78)
    print()
    print(json.dumps(
        {"url": args.url, "repeat": args.repeat, "results": all_results},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
