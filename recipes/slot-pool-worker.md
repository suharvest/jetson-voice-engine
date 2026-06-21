# Slot-pool worker binaries (N>1) — build

Build the N>1 voice worker binaries from the engine overlay: the Base TTS
streaming worker (`qwen3_tts_streaming_worker`, slot-pool + shared-engine ctor)
and the ASR voice worker (`qwen3_asr_worker`, N>1 lane-pool + streaming
PARTIALs). This is the reproducible formalization of the previously hand-built
`~/project/v080-worker-build`.

- **Artifacts:**
  - `build/examples/omni/qwen3_tts_streaming_worker` — Base TTS N>1, slot-pool +
    shared-engine ctor (shares read-only engine weights across lanes).
  - `build/voice-workers/workers/qwen3_asr_worker` — N>1 ASR lane-pool +
    streaming PARTIALs.
  - (also `qwen3_tts_worker`, `moss_tts_nano_worker`; plugin `.so`.)
- **Inputs:** NVIDIA TRT-Edge-LLM upstream @ overlay pin + overlay `addon/` +
  `patches/`. ASR worker source of truth = `native/edgellm_voice_worker/`
  (vendored at feat HEAD; a SEPARATE CMake project the build references).
- **Pin:** `engine-overlay/UPSTREAM_PIN` = **`a361221`** = fork branch
  `suharvest/port/qwen3-tts-base-v080-n1n2` (NVIDIA `release/0.8.0` `f9cc746` +
  6 fork-port commits + the D2-1 shared-engine ctor). The Base N>1 slot-pool
  worker is `10b338d`; shared-engine ctor `a361221`. ASR worker recorded build
  `5ebd436b`.
- **md5:** worker binaries get md5 sidecars written at build time (step 5 of
  `build.sh`); they are not byte-reproducible across toolchains — sidecars are
  for the published binaries' integrity.
- **HF:** worker binaries ship inside the per-device engine sets /
  `harvestsu/seeed-local-voice-artifacts`.
- **Build device:** **SM 87 Jetson Orin** (NX/Nano). CUDA 12.6 / TRT 10.3 /
  JetPack 6. (Note: ENABLE_CUTE_DSL=OFF on CUDA 12.6 → cuBLAS fallback; see
  memory `edgellm_v080_jetson_cutedsl_cuda126`.)

## Command (canonical build entrypoint — do not bypass)

The single legal build entrypoint is `engine-overlay/build.sh`. It clones
upstream@pin → copies `addon/` → applies `patches/*` → builds the engine core +
plugin + workers. The relevant steps:

```bash
cd engine-overlay
./build.sh          # step 4: engine core + plugin + qwen3_tts_streaming_worker
                    # step 4b: ASR voice worker (qwen3_asr_worker) from
                    #          native/edgellm_voice_worker (VOICE_WORKER_SRC)
                    # step 4c: MOSS worker (cpp/workers/build_moss_worker.sh)
                    # step 5: collect binaries + plugin .so + .engine, write md5 sidecars
```

`build.sh` internals for the workers (already wired — for reference):

```bash
# Base TTS N>1 streaming worker (examples/omni)
cmake --build "${WORKDIR}/build" -j"$(nproc)" --target qwen3_tts_streaming_worker

# ASR voice worker (separate CMake project)
VOICE_WORKER_SRC="${HERE}/../native/edgellm_voice_worker"
cmake -S "${VOICE_WORKER_SRC}" -B "${WORKDIR}/build/voice-workers" ...
cmake --build "${WORKDIR}/build/voice-workers" -j"$(nproc)" --target qwen3_asr_worker
```

> Do NOT bypass `build.sh` with bare cmake/make: without the overlay's
> `ORT_ROOT`/`CUDA_HOME`/plugin wiring you get ABI-incompatible artifacts, and
> the build-time plugin `.so` must match the runtime plugin.
> N>1 ASR is delivered by the vendored worker `qwen3_asr_worker.cpp` (lane-pool)
> on the vanilla one-shot ASR engine; the b2 *engine* path
> ([`asr-b2-engine.md`](asr-b2-engine.md)) is the native-batch alternative.

## Runtime selection

N=2 runtime: select profile `jetson-edgellm-v080-n2` (`stream_mode=worker`).
See `engine-overlay/build.sh` step 5/6 banner for the collected binary paths.
