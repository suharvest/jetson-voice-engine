#!/usr/bin/env python3
"""v080-0013 — Direct ASR -> (LLM) -> MOSS-TTS pipeline driver + per-stage perf/memory.

Runs on Orin NX (aarch64). Drives the three v0.8.0 engines as separate processes:
  1. ASR  : spike_v080_m6_audio_streaming (WAV mel -> transcript)
  2. LLM  : llm_inference on the GDN Qwen3.5-4B engine (optional; --llm-engine DIR)
  3. MOSS : moss_tts_nano_worker (stdin/stdout JSON-line streaming worker)

Captures per-stage wall-clock, MOSS TTFA (worker-reported ttfa_ms), and tegrastats
peak unified RAM. Without --llm-engine it runs the ASR->MOSS two-stage chain.

Usage:
  v080_pipeline_driver.py --mel MEL.safetensors [--llm-engine DIR] [--tag NAME]
"""
import argparse, base64, json, os, subprocess, sys, time, wave, struct, signal

EDGELLM = "/home/harvest/project/edgellm-v080"
BUILD = f"{EDGELLM}/build"
PLUGIN = f"{BUILD}/libNvInfer_edgellm_plugin.so"
ASR_LLM = "/home/harvest/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines-v080/llm"
ASR_AUDIO = "/home/harvest/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines-v080/audio"
MOSS_WORKER = "/opt/jv-workers/moss_tts_nano_worker"
MOSS_ENGINES = "/opt/models/moss-tts-nano/engines"
OUT = "/home/harvest/asr_v080_e2e/v080_0013"

ENV = dict(os.environ, EDGELLM_PLUGIN_PATH=PLUGIN,
           LD_LIBRARY_PATH=f"{BUILD}:{os.environ.get('LD_LIBRARY_PATH','')}")


def start_tegra(tag):
    f = open(f"{OUT}/tegrastats_{tag}.log", "w")
    p = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=f, stderr=subprocess.STDOUT)
    return p, f


def peak_ram(tag):
    best = 0
    try:
        for ln in open(f"{OUT}/tegrastats_{tag}.log"):
            toks = ln.split()
            for i, t in enumerate(toks):
                if t == "RAM" and i + 1 < len(toks):
                    v = int(toks[i + 1].split("/")[0])
                    best = max(best, v)
    except FileNotFoundError:
        pass
    return best


def run_asr(mel, tag):
    cmd = [f"{BUILD}/examples/llm/spike_v080_m6_audio_streaming",
           "--mel", mel, "--chunks", "1", "--llm", ASR_LLM, "--audio", ASR_AUDIO]
    t0 = time.time()
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    dt = time.time() - t0
    log = r.stdout + r.stderr
    open(f"{OUT}/asr_{tag}.log", "w").write(log)
    # m6 prints the transcript between markers:
    #   === TRANSCRIPT (chunks=N) ===
    #   <text>
    #   === END TRANSCRIPT ===
    text = ""
    lines = log.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("=== TRANSCRIPT"):
            body = []
            for ln2 in lines[i + 1:]:
                if ln2.startswith("=== END TRANSCRIPT"):
                    break
                body.append(ln2)
            text = "\n".join(body).strip()
            break
    text = text.replace("language Chinese", "").strip()
    return dt, text, log


def run_llm(engine_dir, prompt, tag):
    req = {"batch_size": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 1,
           "max_generate_length": 128,
           "requests": [{"messages": [{"role": "user",
                        "content": [{"type": "text", "text": prompt}]}]}]}
    reqf, outf = f"{OUT}/llm_req_{tag}.json", f"{OUT}/llm_out_{tag}.json"
    json.dump(req, open(reqf, "w"), ensure_ascii=False)
    cmd = [f"{BUILD}/examples/llm/llm_inference", "--engineDir", engine_dir,
           "--inputFile", reqf, "--outputFile", outf]
    t0 = time.time()
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    dt = time.time() - t0
    open(f"{OUT}/llm_{tag}.log", "w").write(r.stdout + r.stderr)
    resp = ""
    try:
        resp = json.load(open(outf))["responses"][0].get("output_text", "")
    except Exception:
        pass
    return dt, resp


def run_moss(text, tag):
    p = subprocess.Popen([MOSS_WORKER, f"--engine-dir={MOSS_ENGINES}",
                          f"--tokenizer-model={MOSS_ENGINES}/tokenizer.model",
                          f"--codec-onnx-dir={MOSS_ENGINES}/../codec_onnx"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(f"{OUT}/moss_{tag}.stderr", "w"),
                         text=True, env=ENV, bufsize=1)
    sr, ch = 48000, 2
    # wait for worker_ready
    while True:
        ln = p.stdout.readline()
        if not ln:
            break
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        if ev.get("event") == "worker_ready":
            sr, ch = ev.get("sample_rate", sr), ev.get("channels", ch)
            break
    req = {"id": "r0", "text": text, "stream": True}
    t0 = time.time()
    p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    p.stdin.flush()
    pcm = bytearray()
    ttfa_wall = None
    ttfa_ms = None
    done = False
    while not done:
        ln = p.stdout.readline()
        if not ln:
            break
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        e = ev.get("event")
        if e == "chunk":
            if ttfa_wall is None:
                ttfa_wall = (time.time() - t0) * 1000
            pcm += base64.b64decode(ev["audio_b64"])
        elif e == "done":
            ttfa_ms = ev.get("ttfa_ms")
            done = True
        elif e == "error":
            open(f"{OUT}/moss_{tag}.err", "w").write(json.dumps(ev))
            done = True
    total = time.time() - t0
    p.stdin.close()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    # write WAV
    wav = f"{OUT}/moss_{tag}.wav"
    samples = struct.unpack(f"<{len(pcm)//2}h", bytes(pcm)) if pcm else []
    with wave.open(wav, "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(bytes(pcm))
    # RMS energy
    rms = 0.0
    if samples:
        rms = (sum((s / 32768.0) ** 2 for s in samples) / len(samples)) ** 0.5
    dur = (len(samples) / ch / sr) if samples else 0.0
    return dict(total=total, ttfa_wall=ttfa_wall, ttfa_ms=ttfa_ms, wav=wav,
                rms=rms, dur=dur, sr=sr, ch=ch, nsamp=len(samples))


def main():
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel", required=True)
    ap.add_argument("--llm-engine", default="")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    tegra, tf = start_tegra(a.tag)
    time.sleep(1)
    print(f"==== v080-0013 pipeline tag={a.tag} mel={os.path.basename(a.mel)} ====")

    asr_dt, asr_text, _ = run_asr(a.mel, a.tag)
    print(f"[ASR ] wall={asr_dt:.3f}s  text={asr_text!r}")
    if not asr_text:
        asr_text = "这并不是告别，这是一个篇章的结束，也是新篇章的开始。"
        print(f"[ASR ] (no transcript parsed; using fallback for downstream stages)")

    llm_dt, llm_resp = None, asr_text
    if a.llm_engine and os.path.isfile(f"{a.llm_engine}/llm.engine"):
        llm_dt, llm_resp = run_llm(a.llm_engine, asr_text, a.tag)
        print(f"[LLM ] wall={llm_dt:.3f}s  resp={llm_resp!r}")
    else:
        print("[LLM ] SKIPPED (GDN engine absent) -> feeding ASR transcript to MOSS")

    moss_text = llm_resp or asr_text
    m = run_moss(moss_text, a.tag)
    print(f"[MOSS] wall={m['total']:.3f}s ttfa_wall={m['ttfa_wall']}ms "
          f"ttfa_worker={m['ttfa_ms']}ms dur={m['dur']:.2f}s rms={m['rms']:.5f} "
          f"sr={m['sr']} ch={m['ch']} samples={m['nsamp']}")

    time.sleep(1)
    tegra.kill()
    try:
        tegra.wait(timeout=5)
    except Exception:
        pass
    tf.close()
    peak = peak_ram(a.tag)

    e2e = asr_dt + (llm_dt or 0) + m["total"]
    print("\n==== SUMMARY ====")
    print(json.dumps(dict(tag=a.tag, asr_s=round(asr_dt, 3),
        llm_s=(round(llm_dt, 3) if llm_dt else None),
        moss_s=round(m["total"], 3), moss_ttfa_ms=m["ttfa_ms"],
        moss_dur_s=round(m["dur"], 2), moss_rms=round(m["rms"], 5),
        e2e_s=round(e2e, 3), peak_ram_mb=peak,
        asr_text=asr_text, moss_in=moss_text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
