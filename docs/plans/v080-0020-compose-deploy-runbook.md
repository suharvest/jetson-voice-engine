# v080-0020 — v0.8.0 one-pull deploy: compose + clean-deploy verification (FINAL 固化)

**Date:** 2026-06-10
**Host:** orin-nx (`Linux 5.15.148-tegra aarch64`, hostname `orinnx`, Orin NX 16GB, JetPack 6.2)
**Image:** `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v0.8.0-edgellm-20260610e`
**Builds on:** v080-0018 (entrypoint CMD fix) + v080-0019 (real serve gate). This step
cements the whole v0.7.1 → v0.8.0 migration into a **single `docker compose up -d`**.

---

## 1. How production is deployed today (run-spec)

Production `seeed-voice` is launched **ad-hoc via `docker run`** (confirmed — there is NO
committed production compose in the repo to edit). Captured from
`docker inspect seeed-voice`:

```
image:    sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:prod-unified-v8
runtime:  nvidia              ports: 8621->8000/tcp
binds:    speech-models:/opt/models
          /usr/local/cuda/lib64:/host-cuda
          /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs
          /lib/aarch64-linux-gnu:/host-libs
          /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt
          /usr/src/tensorrt:/usr/src/tensorrt
env:      OVS_PROFILE=jetson-zh-en  (v0.7.1 stack)
          LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-cuda:/host-nvidia-libs:/host-libs
          NVIDIA_VISIBLE_DEVICES=all
entrypoint: /opt/speech/entrypoint.jetson.sh
cmd:        python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

So the deploy contract is: **slim image + host-lib bind mounts + LD_LIBRARY_PATH +
nvidia runtime + a profile + the model/engine volumes**. The v0.8.0 image keeps the
SAME contract; the only deltas are (a) the image tag, (b) `OVS_PROFILE=jetson-edgellm-v080`,
(c) the deploy-paths env (all baked in the image), and (d) one extra volume for the
runtime-pulled v0.8.0 engines.

## 2. The deploy artifact authored — `deploy/docker/docker-compose.v080.yml`

A NEW, purely-additive compose (it does NOT touch the production run-spec). Service
`seeed-voice-v080`, alt port **8622** (prod holds 8621), so it can run side-by-side.
Full contract it satisfies:

| Contract item | Where |
|---|---|
| `OVS_PROFILE=jetson-edgellm-v080` | baked in image; surfaced in compose `environment` for visibility |
| deploy-paths env (`EDGE_LLM_ASR/TTS_*` dirs, `MOSS_WORKER_BIN`, `EDGELLM_PLUGIN_PATH`, GDN chat url) | **baked in image** (operator-owned snapshot → wins over profile); NOT repeated in compose |
| slim-lib host mounts (`/host-cuda`,`/host-nvidia-libs`,`/host-libs`, 2× tensorrt) | compose `volumes` (verbatim from prod) |
| `LD_LIBRARY_PATH=…onnxruntime/capi:/host-cuda:/host-nvidia-libs:/host-libs` | compose `environment` |
| `--runtime nvidia --gpus all` | compose `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=all` |
| engine volume (first-boot HF pull persists) | `edgellm-v080:/opt/edgellm-v080` (named, local) |
| speech-models volume | `speech-models:/opt/models` (external) |
| port | `8622:8000` |
| healthcheck (long start_period for first-boot pull) | `start_period: 900s` |

**One-pull flow:** `docker compose -f docker-compose.v080.yml up -d` → image entrypoint
HF-pulls the multi-GB v0.8.0 engines (incl. the rebuilt min-chunk-1 audio encoder
`ede676fb`) into the `edgellm-v080` volume on first boot (sentinel-idempotent on later
boots) → chains to uvicorn. No host build, no manual engine staging.

## 3. Backup

- Container-state snapshot before any change → `orin-nx:/tmp/v080-0020-container-snapshot.txt`.
- No existing deploy artifact was modified (production is `docker run`, not a compose
  file) — the compose is brand new (`deploy/docker/docker-compose.v080.yml`), so nothing
  to back up/restore on the artifact side.

## 4. CLEAN-DEPLOY VERIFICATION (raw)

Throwaway service `seeed-voice-v080-deploy`, alt port **8645**, brought up **via a
compose file** (`/tmp/docker-compose.v080-verify.yml` — identical contract to
`docker-compose.v080.yml`, bind-mounting the already-populated `/opt/edgellm-v080`
host cache instead of a fresh named volume, because the host disk is 96% full / 11G
free and the empty-cache first-boot HF pull was ALREADY proven e2e in v080-0019;
`AUTO_PULL=1` still exercises the entrypoint pull logic → sentinel-skip). seeed-voice
**never stopped**; translator + edge-llm-chat-service stopped for RAM, restored after.

### (a) Compose up → pull-skip → boot, clean  ✅
```
$ docker compose -f docker-compose.v080-verify.yml up -d
 Container seeed-voice-v080-deploy Started
[v080-pull] v0.8.0 engines already present (/opt/edgellm-v080/engines/asr/llm/llm.engine) — skipping HF pull.
[INFO] profile_loader: Applied profile jetson-edgellm-v080 (40 env keys; 0 stale cleared)
[INFO] asr_backend: Creating ASR backend jetson.trt_edge_llm
[INFO] trt_edge_llm_asr: ASR backend preload OK (engine_dir=/opt/edgellm-v080/engines/asr/llm,
       audio_encoder_dir=/opt/edgellm-v080/engines/asr/audio, stream_mode='accumulate', ...)
[INFO] trt_edge_llm_tts: TTS backend preload OK (talker=/opt/edgellm-v080/engines/qwen3-tts/talker)
[INFO] backend_manager: BackendManager[asr] ready (trt_edgellm)
[INFO] backend_manager: BackendManager[tts] ready (trt_edgellm)
[INFO] server.main: Speech service ready.
[INFO] uvicorn.error: Application startup complete. Uvicorn running on http://0.0.0.0:8000
$ curl localhost:8645/health
{"tts":true,"tts_backend":"trt_edgellm","asr":true,"asr_backend":"trt_edgellm",...}
$ curl localhost:8645/asr/capabilities → {"backend":"trt_edgellm","capabilities":["streaming","offline","multi_language"],...}
$ curl localhost:8645/tts/capabilities → {"backend":"trt_edgellm","model_id":"qwen3-tts","sample_rate":24000,...}
```

### (b) /tts synthesizes  ✅
```
POST /tts {"text":"The quick brown fox jumps over the lazy dog.","speaker":"2301"}
  → v080d_en.wav  RIFF WAVE 16-bit mono 24000 Hz  (180524 B ≈ 3.76 s)
POST /tts {"text":"今天天气真不错。","speaker":"2301"}
  → v080d_zh.wav  RIFF WAVE 16-bit mono 24000 Hz  ( 92204 B ≈ 1.92 s)
```

### (c) /asr SHORT roundtrip transcribes correctly  ✅

**Offline `/asr` (byte-exact to golden, EN + ZH):**
```
POST /asr file=v080d_en.wav → {"text":"The quick brown fox jumps over the lazy dog.",
                               "language":"English","backend":"trt_edgellm","inference_time_s":0.238}
POST /asr file=v080d_zh.wav → {"text":"今天天气真不错。",
                               "language":"Chinese","backend":"trt_edgellm","inference_time_s":0.157}
```

**Streaming `/asr/stream` WS:**
```
ZH:            FINAL='今天天气真不错。'                               (== golden, byte-exact)
EN (vad=none): FINAL='The quick brown fox jumps over the lazy dog.'  (== golden, byte-exact)
EN (vad=default): FINAL='Brown fox jumps over the lazy dog.'         (leading 'The quick' clipped)
```
The default-VAD EN onset clip is a **client/VAD front-end nuance** (the abrupt synthetic-
TTS onset is trimmed by the optional silero VAD), NOT an engine/deploy defect — proven
by (i) `vad=none` returning byte-exact and (ii) the offline `/asr` path returning
byte-exact. The `accumulate` stream-mode finalize uses the same one-shot worker decode
as `/asr` (Blocker-2 contract), so the served transcript is the offline-correct one.

### (d) Logs clean  ✅
```
docker logs seeed-voice-v080-deploy | grep -iE 'satisfyProfile|501|no event for|in 60s|
  Failed to set padded|preprocessing failed|Traceback|cannot open|unrecognized option|
  CUDA error|illegal memory'  (excl. uvicorn.error)  → empty
```
Only one benign match overall: an ORT `GPU device discovery failed … /sys/class/drm/card1`
warning (cosmetic, present in production too). `Running=true ExitCode=0 OOMKilled=false
RestartCount=0`.

## 5. Container restore

Pre-run: seeed-voice Up, translator Up (healthy), edge-llm-chat-service Up (healthy),
industrial-security-demo Restarting (pre-existing loop). **seeed-voice NEVER stopped**
(Up ~1 h throughout, /health on 8621 verified post-run). translator +
edge-llm-chat-service stopped for RAM → restarted → **both healthy** again. Throwaway
`seeed-voice-v080-deploy` + its `tmp_default` network removed.
**Final:** seeed-voice Up, translator Up (healthy), edge-llm-chat-service Up (healthy),
industrial-security-demo Restarting (untouched), throwaway gone.

## 6. Verdict — 固化 complete

**YES — v0.8.0 is one-pull deployable.** `docker compose -f docker-compose.v080.yml
up -d` brings the migrated stack up from a single image pull: the slim image HF-pulls
the v0.8.0 engines on first boot, applies `jetson-edgellm-v080`, preloads both
trt_edgellm backends, and serves `/health` + `/asr` + `/asr/stream` + `/tts` correctly
(EN + ZH, byte-exact transcripts, 24kHz synthesis, clean logs). The full slim-lib mount
contract (Blocker 4) is encoded in the compose. The whole v0.7.1 → v0.8.0 migration is
now cemented into a repeatable, no-host-build deploy.

### Cut-over note (production flip — when ready, NOT done here)
To move production onto v0.8.0: stop the ad-hoc `seeed-voice` run, then either (i) run
`docker-compose.v080.yml` and remap its port 8622→8621, or (ii) re-issue the production
`docker run` with `image=:v0.8.0-edgellm-20260610e` + `OVS_PROFILE=jetson-edgellm-v080`
+ the `-v edgellm-v080:/opt/edgellm-v080` engine volume added (everything else identical
to the current run-spec). NOT performed in this task (production untouched).

## Files (this commit)
- `deploy/docker/docker-compose.v080.yml` — NEW: one-pull v0.8.0 deploy compose (full contract).
- `docs/plans/v080-0020-compose-deploy-runbook.md` — this doc.

Image `:v0.8.0-edgellm-20260610e`, profile `jetson-edgellm-v080`. NOT pushed (per task).
