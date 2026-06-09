# v0.8.0 TTS Phase A — CustomVoice 9-row language conditioning port + coherent engines

Status: **PORT LANDED + COMPILES + 9-ROW LANGUAGE PATH PROVEN; full audio gate BLOCKED on two
precisely-diagnosed device-toolchain issues** (CuTe DSL GEMM launch on the box's CUDA-12.6 patch
level + the Code2Wav engine build under GPU-memory pressure). 2026-06-10.

Patch `engine-overlay/patches/v080-0007-customvoice-language-conditioning.patch`
(md5 `74b31e98e8f90c758bc5a222f55c187e`, 7 files).
Spec: `docs/plans/asr-streaming-v080-migration.md` §6 v2 ("TTS — Talker batches …"). This is the
TTS track; Phase A = **N=1 single-session correctness** of the CustomVoice 9-row language
conditioning (patch 0005 ported onto v0.8.0). Native-batch N=2 is Phase 5b (separate).

Host: orin-nx (`Linux aarch64`, hostname `orinnx`), repo worktree `~/project/edgellm-v080`
(worktree of `~/project/TensorRT-Edge-LLM`; migration tracked as PATCHES, not device commits).

## 0. Host-identity proof
```
$ fleet exec orin-nx -- 'uname -srm; hostname; ls -d ~/project/edgellm-v080'
Linux 5.15.148-tegra aarch64
orinnx
/home/harvest/project/edgellm-v080
```

## 1. Port of patch 0005 (9-row CustomVoice language conditioning) onto v0.8.0

v0.8.0's `talkerMLPKernels.{cu,h}` is the pristine 8-row baseline (`kFixedPrefixLen = 8`,
`TalkerGenerationRequest` has no `language`, config struct has no `codecThinkId`/`codecLanguageId`).
Patch 0005's kernel/header/example hunks **applied cleanly** (3-way) to v0.8.0; the runtime wiring
and the export-side hunk were hand-ported (v0.8.0 moved the TTS export from the deleted
`experimental/llm_loader/export_all_cli.py` to `tensorrt_edgellm/scripts/export.py`).

`v080-0007` (7 files):
- `cpp/kernels/talkerMLPKernels/talkerMLPKernels.cu` — `assistantPreambleKernel` gains `codecThinkId`
  + `langId`; `kFixedPrefixLen = (langId>=0) ? 9 : 8`; 9-row branch injects `langId` at row 5,
  shifts `codecThinkEosId→row6`, `speaker→row7`, `codecPad/ttsBos→row8`. (= patch-0005 hunk.)
- `cpp/kernels/talkerMLPKernels/talkerMLPKernels.h` — `invokeAssistantPreamble` signature +
  langId/codecThinkId; doc block. (= patch-0005 hunk.)
- `examples/omni/qwen3_tts_inference.cpp` — per-request `language` + user→assistant role coerce.
  (= patch-0005 hunk.)
- `cpp/runtime/qwen3OmniTTSRuntime.h` — `TalkerGenerationRequest.language`; config `codecThinkId`
  + `std::unordered_map<std::string,int32_t> codecLanguageId`; `projectToTalkerInput(...,langId,...)`.
  (hand-port from v0.7.1 fork `~/project/v071-customvoice-port`.)
- `cpp/runtime/qwen3OmniTTSRuntime.cpp` — load `codec_think_id` + `codec_language_id` map;
  resolve `request.language`→lower-case→map→`langId`; thread `langId`+`codecThinkId` into both
  `invokeAssistantPreamble` call sites (single-request path: real langId; omni-segment path:
  langId=-1 = 8-row, matching the v0.7.1 fork decision). (hand-port.)
- `examples/omni/CMakeLists.txt` — link `trt_edgellm_cutedsl_cudart_shim` whole-archive
  (the LLM examples already do this; the omni example was missing it — see §3 build note).
- `tensorrt_edgellm/scripts/export.py` — `_patch_tts_config` copies `codec_language_id` into the
  talker config (v0.8.0's port of patch-0005's `export_all_cli.py` hunk; v0.8.0 already copied
  `codec_think_id` but not the language map).

Round-trip validated on the Mac: `git apply --check` OK against the reconstructed pristine tree, and
the applied result is byte-identical to the modified files pulled off the box (`diff -r` empty).

## 2. Coherent v0.8.0 qwen3-tts worker (worker↔runtime↔engine version match)

The v0.8.0 "TTS worker" IS `examples/omni/qwen3_tts_inference` (no separate worker binary; the
v0.7.1 worker incoherence was an artifact-version problem). Coherence is by construction here:
the same v0.8.0 tree provides the runtime (`cpp/runtime/qwen3OmniTTSRuntime.cpp`), the example, and
the engines (re-exported below from v0.8.0 `tensorrt-edgellm-export`). The binary linked + ran
loading the v0.8.0-schema engines (§4), and the runtime read `kv_cache_dtype` from the engine
config (the field the v0.7.1-engine incoherence lacked).

Build (official entrypoint only, detached + tail-poll):
```
cmake -S . -B build  →  Configuring/Generating done
cmake --build build --target edgellmKernels edgellmCore exampleUtils qwen3_tts_inference -j4
→ BUILD_EXIT=0,  build/examples/omni/qwen3_tts_inference (18 MB)
```

### Build note: two pre-existing v0.8.0 link hazards fixed as part of this port
1. **Missing cudart shim on the omni example.** The CuTe DSL artifact references
   `cudaKernelSetAttributeForDevice`; the LLM examples link `trt_edgellm_cutedsl_cudart_shim`
   whole-archive but the omni example's `CMakeLists.txt` did not → `undefined reference`. Added the
   same block (now in v080-0007).
2. **CUDA separable-compilation stale device-link.** After the `talkerMLPKernels.cu` change, the
   fatbin hash flipped (`dc27a880…`→`1378ce47…`), but `examples/utils/libexampleUtils.a`'s
   pre-device-linked `cmake_device_link.o` still demanded the OLD hash → `undefined reference to
   __fatbinwrap_…_talkerMLPKernels_cu_dc27a880_…`. Fixed by rebuilding `exampleUtils` +
   `edgellmKernels` (their device-link objects regenerated against the new hash). This is a
   build-procedure note, not a source change.

## 3. Re-export + build of v0.8.0 qwen3-tts engines (CustomVoice)

ONNX re-export was MANDATORY (the on-box TTS engines were `edgellm_version 0.7.0/0.7.1`, no
`kv_cache_dtype`; the v0.8.0 export stamps `edgellm_version 0.8.0` + `kv_cache_dtype fp16` into the
ONNX config — same as the ASR `onnx-v080`). The `.venv-x86export` runs on aarch64 (torch
2.12.0+cu130), so the export ran on the Orin itself.

```
HF_ENDPOINT=https://hf-mirror.com \
  .venv-x86export/bin/tensorrt-edgellm-export Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice  $WS/.../onnx
```
- Talker + CodePredictor ONNX exported clean.
- **Code2Wav ONNX needed a CPU override**: the default code2wav exporter forces `decoder.to("cuda")`;
  the torch+cu130 wheel rejects the box's driver ("too old, found 12060"). Re-ran ONLY code2wav with
  `export_qwen3_tts_code2wav(..., device="cpu")` (a 1-line driver, CUDA_VISIBLE_DEVICES="") → OK.

ONNX (v0.8.0 schema) — talker `config.json`:
```
"edgellm_version": "0.8.0"   "kv_cache_dtype": "fp16"   "codec_think_id": 2154
"codec_language_id": {"chinese":2055,"english":2050,"german":2053,"italian":2070,
  "portuguese":2071,"spanish":2054,"japanese":2058,"korean":2064,"french":2061,
  "russian":2069,"beijing_dialect":2074,"sichuan_dialect":2062}
```
⇒ the export.py port worked: the **CustomVoice language map is now in the v0.8.0 talker config**.

Engine builds (`build/examples/llm/llm_build` + `build/examples/multimodal/audio_build`,
`--maxBatchSize 1`, official entrypoints):
```
talker/llm.engine          md5 471d36d8f730f560a6949c741f49e6ae   (864 MiB loaded, v0.8.0+langmap)
code_predictor/llm.engine  md5 baff21ea48a9f7e8de30e3d698544e4d   (182 MiB loaded)
code2wav/code2wav.engine   NOT BUILT — see §5 blocker B
```
Talker engine `config.json` carries `edgellm_version 0.8.0` + `kv_cache_dtype fp16` +
`codec_language_id` + `codec_think_id` — the coherence + language-conditioning proof at the
engine-config level.

## 4. THE CORRECTNESS GATE — 9-row CustomVoice language path PROVEN; full-audio gate BLOCKED

Ran `qwen3_tts_inference` with the v0.8.0 talker + code_predictor engines (code2wav engine absent →
runtime gracefully degrades to "RVQ codes only"). The run loads everything coherently and **executes
the 9-row CustomVoice language-conditioning path with the correct per-language codec ids**:

```
ZH (language="chinese"):
  validateAndFillConfig: CustomVoice language config: codecThinkId=2154, codecLanguageId entries=12
  prepareTalkerInput: CustomVoice language conditioning enabled: language="chinese" -> codec_id=2055
  projectToTalkerInput: seqLen=18 N=10 outputSeqLen=21 speakerId=3065 langId=2055 prefixRows=9

EN (language="english"):
  prepareTalkerInput: CustomVoice language conditioning enabled: language="english" -> codec_id=2050
  projectToTalkerInput: seqLen=17 N=9  outputSeqLen=20 speakerId=3061 langId=2050 prefixRows=9
```
⇒ **`prefixRows=9` (vs the legacy 8) with distinct, correct codec ids (chinese→2055, english→2050)
proves the ported 9-row language path resolves langId from the config map and dispatches the 9-row
kernel** — the exact behaviour patch 0005 adds, now live on v0.8.0. zh vs en differ in langId,
speakerId, and sequence length, as expected.

**The full energy + ASR-roundtrip gate could NOT be completed** because actual audio requires
the CuTe DSL Talker MLP GEMM (`invokeTalkerMLP`, called immediately after the log above) and the
Code2Wav vocoder engine — both blocked (§5). The run aborts right after the `projectToTalkerInput`
log with:
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  CUDA runtime error in cudaPeekAtLastError(): invalid device function
```

## 5. Two precisely-diagnosed blockers (why no WAV / no energy+roundtrip)

### Blocker A — CuTe DSL GEMM "invalid device function" on the box's CUDA-12.6 patch level
`invokeTalkerMLP` (`qwen3OmniTTSRuntime.cpp:789`, the Talker text-projection GEMM) is the first
kernel after langId resolution. It dispatches the prebuilt CuTe DSL artifact
(`cpp/kernels/cuteDSLArtifact/aarch64/sm_87/libcutedsl_aarch64.a`, `gemm_ampere_*_fp16` variants).
The module "loads for SM87" but the **launch** throws `invalid device function`.
- The project's own `talkerMLPKernels.cu.o` IS compiled for sm_87 (`talkerMLPKernels.sm_87.cubin`),
  so this is NOT a missing-arch issue.
- The CuTe artifact `metadata.json`: `cuda_version 12.6.68`, `cutlass_dsl_version 4.5.1`, sm_87,
  built 2026-06-02. It link-references `cudaKernelSetAttributeForDevice`.
- **The box's installed libcudart (`/usr/local/cuda 12.6.x`) does NOT export
  `cudaKernelSetAttributeForDevice`** (nm count 0). That symbol exists in newer 12.6 patch releases
  (the artifact's 12.6.68). The link-time `trt_edgellm_cutedsl_cudart_shim` stubs the symbol so the
  binary LINKS, but at runtime the GEMM kernel's device-side registration is a no-op stub → the
  kernel is never validly registered for the device → `invalid device function` at launch.
- **Root cause: the box's CUDA 12.6 patch level is older than the CuTe DSL artifact's 12.6.68 build
  target.** The LLM/ASR path does not use the Talker CuTe GEMM, which is why the ASR track built and
  ran cleanly and only the TTS talker hits this.
- Fix options (all out-of-scope here, none safe on a 99%-full disk): (a) rebuild the CuTe DSL
  artifact against the box's exact CUDA 12.6.x via `kernelSrcs/build_cutedsl.py`; or (b) update the
  box's CUDA 12.6 to ≥12.6.68 so `cudaKernelSetAttributeForDevice` is real (then drop the stub).

### Blocker B — Code2Wav TensorRT engine build starved by GPU memory
`audio_build` on the code2wav ONNX ran >17 min without finishing; TensorRT repeatedly logged
`Tactic Device request: 11250–16875MB Available: 8140MB … insufficient memory … Skipping tactic`,
stalling for ~13 min on a single heavy vocoder layer (process healthy, CPU-pinned, log frozen).
With only ~8 GB GPU free on the 16 GB Orin (other services co-resident), TensorRT exhaustively
eliminates the large tactics. It was killed to restore containers (the two LLM engines + all ONNX
are preserved). Retry needs a quieter GPU (all co-resident services stopped) and/or a longer wall.

## 6. Container restore (snapshot → restore)
For RAM during build/export/validation, `seeed-voice` + `edge-llm-chat-service` were stopped
(`--sudo docker stop`), then restarted at the end:
```
$ docker ps --format '{{.Names}}\t{{.Status}}'
seeed-voice               Up 2 minutes
translator                Up 10 hours (healthy)        # untouched
industrial-security-demo  Restarting (1) ...           # pre-existing crash-loop, untouched
edge-llm-chat-service     Up 2 minutes (healthy)
```
Disk left at ~5–6 GB free; only my own throwaway export scratch
(`~/edgellm-workspace/qwen3-tts-onnx-minimal-fix`, the HF checkpoint cache) was cleaned — never the
ASR `engines-v080*`, production, or compose.

## 7. Verdict
- **9-row CustomVoice language conditioning port: LANDED + COMPILES + PROVEN at the Talker
  language-resolution/dispatch level** (v080-0007). zh→codec_id 2055 / en→codec_id 2050, both
  `prefixRows=9`, distinct speakers — the exact patch-0005 behaviour, now on v0.8.0 with the
  language map flowing all the way from `tensorrt-edgellm-export` → talker engine config → runtime.
- **Coherent v0.8.0 stack assembled**: v0.8.0 runtime + v0.8.0 example binary + v0.8.0-schema talker
  & code_predictor engines (re-exported, `edgellm_version 0.8.0`/`kv_cache_dtype fp16`/`codec_language_id`),
  worker↔runtime↔engine version-matched by construction.
- **Producing correct CustomVoice speech: NOT YET demonstrated** — blocked by (A) the CuTe DSL GEMM
  launch failure (box CUDA-12.6 patch level < the artifact's 12.6.68) and (B) the Code2Wav engine
  build under GPU-memory pressure. Neither is in the ported source; both are device-toolchain /
  resource issues with concrete fixes listed in §5.
- **What remains**: (i) resolve Blocker A (rebuild CuTe DSL artifact for the box's CUDA, or bump
  CUDA to ≥12.6.68) → talker GEMM launches → real codes; (ii) build the Code2Wav engine on a quiet
  GPU; (iii) THEN the energy + ASR-roundtrip gate (zh + en WAV, RMS non-silent, transcript correct);
  (iv) native-batch N=2 (Phase 5b); (v) MOSS path (separate).
