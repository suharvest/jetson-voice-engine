#!/usr/bin/env python3
"""v080-0013 co-residency memory probe.

Holds the MOSS-TTS-Nano worker RESIDENT (engines loaded, idle on stdin) while the
ASR engine is loaded and run in a separate process, so tegrastats captures the TRUE
peak with BOTH ASR + MOSS engines co-resident in unified memory simultaneously.

(The GDN Qwen3.5-4B LLM engine is added to the projection from its documented
5.76GB real-hardware measurement when its engine is absent.)
"""
import base64, json, os, subprocess, sys, time, signal

EDGELLM = "/home/harvest/project/edgellm-v080"
BUILD = f"{EDGELLM}/build"
PLUGIN = f"{BUILD}/libNvInfer_edgellm_plugin.so"
ASR_LLM = "/home/harvest/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines-v080/llm"
ASR_AUDIO = "/home/harvest/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/engines-v080/audio"
MOSS_WORKER = "/opt/jv-workers/moss_tts_nano_worker"
MOSS_ENGINES = "/opt/models/moss-tts-nano/engines"
OUT = "/home/harvest/asr_v080_e2e/v080_0013"
MEL = "/home/harvest/asr_v080_e2e/mels/zh_long_01.safetensors"

ENV = dict(os.environ, EDGELLM_PLUGIN_PATH=PLUGIN,
           LD_LIBRARY_PATH=f"{BUILD}:{os.environ.get('LD_LIBRARY_PATH','')}")


def peak(logf):
    best = 0
    for ln in open(logf):
        toks = ln.split()
        for i, t in enumerate(toks):
            if t == "RAM" and i + 1 < len(toks):
                best = max(best, int(toks[i + 1].split("/")[0]))
    return best


def snap_ram():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            avail = int(ln.split()[1]) // 1024
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemTotal"):
            tot = int(ln.split()[1]) // 1024
    return tot - avail, tot


def main():
    os.makedirs(OUT, exist_ok=True)
    tlog = f"{OUT}/tegrastats_coresident.log"
    tf = open(tlog, "w")
    tegra = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=tf, stderr=subprocess.STDOUT)
    time.sleep(1)

    used0, tot = snap_ram()
    print(f"[baseline]  used={used0} MB / {tot} MB")

    # 1. Start MOSS worker, wait until engines loaded (worker_ready), keep RESIDENT
    moss = subprocess.Popen([MOSS_WORKER, f"--engine-dir={MOSS_ENGINES}",
                             f"--tokenizer-model={MOSS_ENGINES}/tokenizer.model",
                             f"--codec-onnx-dir={MOSS_ENGINES}/../codec_onnx"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=open(f"{OUT}/moss_coresident.stderr", "w"),
                            text=True, env=ENV, bufsize=1)
    while True:
        ln = moss.stdout.readline()
        if not ln:
            break
        try:
            if json.loads(ln).get("event") == "worker_ready":
                break
        except Exception:
            pass
    time.sleep(1)
    used_moss, _ = snap_ram()
    print(f"[+MOSS res] used={used_moss} MB  (MOSS engines = +{used_moss-used0} MB)")

    # 2. Generate one MOSS utterance so its decode/code2wav buffers are also live,
    #    THEN (while MOSS still resident) load + run ASR.
    moss.stdin.write(json.dumps({"id": "warm", "text": "测试语音合成的内存占用。",
                                 "stream": True}, ensure_ascii=False) + "\n")
    moss.stdin.flush()
    while True:
        ln = moss.stdout.readline()
        if not ln or json.loads(ln).get("event") == "done":
            break
    used_moss2, _ = snap_ram()
    print(f"[MOSS gen ] used={used_moss2} MB  (after one synth, buffers live)")

    # 3. Run ASR while MOSS still resident -> BOTH co-resident
    t0 = time.time()
    r = subprocess.run([f"{BUILD}/examples/llm/spike_v080_m6_audio_streaming",
                        "--mel", MEL, "--chunks", "1", "--llm", ASR_LLM, "--audio", ASR_AUDIO],
                       env=ENV, capture_output=True, text=True)
    asr_dt = time.time() - t0
    open(f"{OUT}/asr_coresident.log", "w").write(r.stdout + r.stderr)
    used_both, _ = snap_ram()
    txt = ""
    for i, ln in enumerate(r.stdout.splitlines()):
        if ln.startswith("=== TRANSCRIPT"):
            txt = r.stdout.splitlines()[i + 1].strip()
    print(f"[ASR+MOSS ] used={used_both} MB  (ASR engines while MOSS resident) "
          f"ASR_wall={asr_dt:.3f}s text={txt!r}")

    # shut down MOSS
    moss.stdin.close()
    try:
        moss.wait(timeout=10)
    except Exception:
        moss.kill()
    time.sleep(1)
    tegra.kill()
    try:
        tegra.wait(timeout=5)
    except Exception:
        pass
    tf.close()
    pk = peak(tlog)

    print("\n==== CO-RESIDENCY SUMMARY ====")
    print(json.dumps(dict(
        baseline_used_mb=used0,
        moss_resident_mb=used_moss,
        moss_after_synth_mb=used_moss2,
        asr_plus_moss_used_mb=used_both,
        asr_moss_engines_delta_mb=used_both - used0,
        tegrastats_peak_mb=pk,
        total_mb=tot,
        documented_gdn_llm_mb=5760,
        projected_3engine_used_mb=used_both + (5760 - 0),  # add LLM on top of ASR+MOSS+baseline
        headroom_vs_16gb_mb=tot - (used_both + 5760),
    ), indent=2))


if __name__ == "__main__":
    main()
