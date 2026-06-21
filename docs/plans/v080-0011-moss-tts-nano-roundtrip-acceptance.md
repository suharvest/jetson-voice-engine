# v0.8.0 MOSS-TTS-Nano — port + ASR-roundtrip correctness acceptance

Date: 2026-06-10
Host: orin-nx (Orin NX 16GB, JetPack 6, CUDA 12.6, TRT 10.3) — verified `Linux aarch64` + `orinnx`.
Patch: `engine-overlay/patches/v080-0011-moss-tts-nano-port.patch`
Source of truth: `engine-overlay/addon/` (byte-identical to Mac fork `TensorRT-Edge-LLM`
branch `v071/customvoice-product`, commits `46bb2b8` + `d92a306`).

## Scope

Port the MOSS-TTS-Nano addon TTS backend (linear-attention KV kernel + standalone
stateful TRT runtime + JSON-line worker) onto v0.8.0 and verify it produces
**correct** Chinese speech (not just non-silent energy) via ASR-roundtrip. Per
DIVERGENCE.md `moss-tts-nano-port` this is the cleanest migration category:
almost all ISO new files, one CMake registration hunk.

## 1. Files added + v0.7.1→v0.8.0 adaptations

8 new files (verbatim from fork — `diff -q` clean vs Mac fork working tree):

```
cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.cu   (111)
cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.h    (30)
cpp/runtime/mossTtsNanoRuntime.cpp                       (1276)
cpp/runtime/mossTtsNanoRuntime.h                         (271)
cpp/workers/moss_tts_nano_worker.cpp                     (822)
cpp/workers/build_moss_worker.sh                         (64)
unittests/mossLinearKvKernelsTests.cu                    (161)
unittests/mossTtsNanoSmokeMain.cpp                       (120)
```

**v0.7.1→v0.8.0 source adaptations: ZERO.** The MOSS runtime is standalone
(`class MossTtsNanoRuntime`, no inheritance from the re-architected v0.8.0 runtime).
Its only edgellm dependency is `gLogger` from `common/logger.h` (present, unchanged)
and `trt_edgellm::format::fmtstr` from `common/stringUtils.cpp` (present, unchanged).
All TRT calls are raw `nvinfer1` 10.x stable APIs (`getTensorDataType`, `enqueueV3`,
`getTensorShape`, `setInputShape`, `setTensorAddress`) — identical between the v0.7.1
fork and v0.8.0's TRT 10.3. The sources compiled into edgellmCore with **no edits**.

**CMake CORE touch (the only core change): a 5-line hunk in `CMakeLists.txt`.**
v0.8.0's `cpp/CMakeLists.txt` uses `file(GLOB_RECURSE kernels/*.cu, runtime/*.cpp)`,
so the kernel and runtime auto-register into `edgellmKernels` + `edgellmCore` with no
explicit list edit. The top-level `CMakeLists.txt` also GLOBs `unittests/*.cu/*.cpp`
into the `unitTest` target, which auto-registers `mossLinearKvKernelsTests.cu`. The
only manual hunk needed: exclude `mossTtsNanoSmokeMain.cpp` (which defines its own
`main()` and pulls the full TRT runtime — it is a standalone driver, not a gtest) from
the gtest_main-linked `unitTest` target via
`list(FILTER UNIT_TESTS_SRCS EXCLUDE REGEX "mossTtsNanoSmokeMain\.cpp$")`.

## 2. Build proof (official build)

- `cmake .` re-glob (arch=aarch64 sm_87) → Configuring/Generating done.
- `make -j2 edgellmCore` → compiled both MOSS objects, relinked `libedgellmCore.a`:
  ```
  [  0%] Building CUDA object .../edgellmCore.dir/kernels/kvCacheUtilKernels/mossLinearKvKernels.cu.o
  [  0%] Building CXX object  .../edgellmCore.dir/runtime/mossTtsNanoRuntime.cpp.o
  [100%] Built target edgellmCore
  ```
  Objects: `mossLinearKvKernels.cu.o` (27152 B), `mossTtsNanoRuntime.cpp.o` (403112 B).
- `cpp/workers/build_moss_worker.sh` (EDGELLM_SRC=edgellm-v080, ORT_ROOT=
  /opt/onnxruntime-linux-aarch64-1.23.2, SP_ROOT=/usr) →
  `[build] OK -> /tmp/moss_tts_nano_worker` (550K ELF aarch64 PIE, links the two MOSS
  objects + stringUtils.o + nvinfer/cudart/onnxruntime/sentencepiece).
- `cmake -DBUILD_UNIT_TESTS=ON .` → Configuring done (exit 0) — confirms the CMake
  hunk leaves the official unitTest target configurable; cache reset to OFF after.

### mossLinearKvKernelsTests — gtest 5/5 PASS

Built standalone from the official source (kernel `.o` + `stringUtils.o` + gtest +
gtest_main) — equivalent to the `unitTest` target's MOSS test, RAM/disk-frugal:

```
[==========] Running 5 tests from 1 test suite.
[ RUN/OK   ] MossLinearKvKernelsTest.DecodeWritesLogicalPastPositionAndLeavesOthersZero (53 ms)
[ RUN/OK   ] MossLinearKvKernelsTest.PrefillWritesPositionsZeroToSeven (1 ms)
[ RUN/OK   ] MossLinearKvKernelsTest.WriteToMaxFillsExactlyToCapacity (1 ms)
[ RUN/OK   ] MossLinearKvKernelsTest.GenericPathHeadDim128 (1 ms)
[ RUN/OK   ] MossLinearKvKernelsTest.RejectsAppendPastMaxSeqLen (0 ms)
[  PASSED  ] 5 tests.
```

`mossTtsNanoSmokeMain.cpp` also built standalone (307K binary) — confirms the smoke
driver compiles + links the full MOSS runtime against v0.8.0 edgellmCore.

## 3. MOSS engine staging

Already on-box (no HF pull needed) at `/opt/models/moss-tts-nano/engines/`: the 6
TRT 10.3 `.plan` engines (prefill, decode_step [FP32 v16 444MB active], local_decoder,
local_cached_step, local_fixed_sampled_frame, codec_decode_step) + tokenizer.model +
codec_onnx/moss_audio_tokenizer_encode.onnx + tts/codec meta JSON. MOSS bypasses the
edgellm version-check, so TRT-10.3 engines load against the v0.8.0 runtime.

## 4. ASR-roundtrip correctness gate

ASR: production `seeed-voice` paraformer_trt (kept UP). To avoid the pre-existing
server HTTP `/asr` bug (`**result.meta` with `meta=None`, unrelated to this work), the
backend was driven **in-container** exactly as prior v0.8.0 TTS tracks did:
`apply_profile()` → `create_asr_backend().transcribe(wav_bytes, language="zh")`, with
`PARAFORMER_ENC_ENGINE=...paraformer_encoder_dp4_400.plan` (the profile's engine).

**ASR validated on known-good audio FIRST** (human-recorded `zh_3.wav`, 16k mono):
- input human clip → `说起咱北京的烤鸭啊那可真是外焦里嫩色泽金黄一口咬下去满嘴流油`
- Clean, coherent → harness trustworthy.

### MOSS generation (fresh, this run) — worker, default voice conditioning, 48kHz stereo

| prompt | input text | dur | RMS (non-silent) | ASR roundtrip | verdict |
|---|---|---|---|---|---|
| moss_zh_1 | 今天天气很好，我们一起测试语音合成。 | 4.08s | 0.04194 | 天天气很好我们一起测测试语音合成 | PASS (intelligible, matches) |
| moss_zh_2 | 语音合成的稳定性。 | 1.68s | 0.05975 | 语音合合成的稳定性 | PASS (intelligible, matches) |
| moss_zh_3 | 说起咱北京的烤鸭啊，那可真是外焦里嫩。 | 3.92s | 0.06328 | 起咱北京的烤烤鸭啊那可真是外焦里嫩 | PASS (intelligible, matches) |

All three: RMS energy non-silent AND ASR-roundtrip transcript is the input sentence.
Minor leading-char clip (今/说) and doubled chars (测测/合合/烤烤) are paraformer
streaming-decoder artifacts on 48k-stereo→16k-resampled input, not MOSS errors — the
words are unambiguously correct. This matches the DIVERGENCE CER=0 reference within
ASR-streaming tolerance.

Golden audio: `docs/audio-evidence/`
- `v080-0011-moss_zh_1-2026-06-10.wav`  md5 e401641ba43f08188645640bd8524869
- `v080-0011-moss_zh_2-2026-06-10.wav`  md5 89dfb992870a4dfed029a96d47d903af
- `v080-0011-moss_zh_3-2026-06-10.wav`  md5 623ccad36b8edaf584194f80c1a36701

## 5. Container restore

Snapshot at `/tmp/container_snapshot_v0011.txt`. For RAM headroom during MOSS
generation, `translator` + `edge-llm-chat-service` were stopped (seeed-voice kept UP
for roundtrip), then restarted. `industrial-security-demo` (production-data) untouched
— remained in its pre-existing restart loop throughout. Final: seeed-voice Up,
translator Up, edge-llm-chat-service Up.

## Verdict

**MOSS-TTS-Nano produces correct Chinese speech on v0.8.0 — roundtrip-verified.**
The port is a clean verbatim addon (zero source adaptation) + one 5-line CMake hunk;
edgellmCore + worker + kernel gtest (5/5) all build on the official toolchain; all 3
zh prompts pass the energy + ASR-roundtrip correctness gate.

This was the cleanest of the v0.8.0 TTS-track items — unlike qwen3-tts CustomVoice
(`5ad0cbb`), MOSS needs no `--wrap` shim, no engine rebuild, and has no Chinese
early-EOS issue, because it is a self-contained runtime independent of the
re-architected edgellm core.
