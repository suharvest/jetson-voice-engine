# Patch state — v0.9.0 serving chain (P4-1 re-pin)

> **CURRENT AUTHORITATIVE STATE.** Supersedes `PATCH-STATE-v080.md` (kept for
> provenance; its §13 was the previous authoritative state on the v0.8.0 pin).

## Pin (supersedes PATCH-STATE-v080.md §13)
```
UPSTREAM_PIN = 1ac0f2b99642045125e1c5ac7b109434ba3b36c7
             = NVIDIA tag v0.9.0 ("Merge PR #122 from jhalabi-nv/dev-release/0.9.0",
               PURE NVIDIA, no fork content)
upstream.remote = https://github.com/NVIDIA/TensorRT-Edge-LLM.git (fork = fallback)
```
ALL fork content travels as patches: `v090-sparktts-0001..0038` =
`git format-patch --no-stat 1ac0f2b9..integration/v090-sparktts` (fork worktree
`~/project/tel-v090-rebase`, HEAD `e8c59c1`). The series is the v0.9.0 rebase of
the former `v080-sparktts-0001..0030` (deleted) plus v0.9.0-specific re-ports:

| range | content | fork commits |
|---|---|---|
| 0001–0003 | sm_87 build/perf: cudart shim into omni exes, cuBLAS-free tiled FP16 GEMM, warp-per-column M=1 GEMV | d18bfa3 e024b2c 07d3f4f |
| 0004–0005 | ASR SessionLaneManager + `maxSessionBatchSize()` (N>1 worker) | d87bf38 ab54e31 |
| 0006–0023 | SparkTTS mixed-precision + INT4-AWQ opt-in set, incl. `mixed_precision` → `bf16_residual` rename (upstream name clash) | eb03efd…d74e35b |
| 0024 | streaming worker **RE-PORTED to the v0.9.0 native streaming API** (slot-pool now `examples/omni/slotPool.h`) | bc3540e |
| 0025–0032 | omni-TTS on the new worker: N<=0 prefill guard, CV 9-row langId conditioning, external speaker-embedding, FP8 e4m3 text_embedding, language + `speaker_embedding_b64` wiring, cooperative cancel, shared-engine ctors, N=2 differentiated chunking | dcf47de…6f0fbfc |
| 0033–0034 | **MOSS-TTS-Nano re-port, now IN-SERIES**: runtime+kernels+worker re-ported as-is from the v0.7.x line, then wired as CMake target `moss_tts_nano_worker` under `examples/omni/` (skipped when ORT/SentencePiece missing) | 2a48e62 d18ddd2 |
| 0035–0038 | checkpoint/export fixes: fp32→half cast in `_set_tensor`, ASR `rope_type: mrope`, base-export guard demotion, `codec_language_id` carry | 13be473 aec6300 aafa7c5 e8c59c1 |

All opt-in: default paths byte-identical to upstream v0.9.0 behaviour.

## Apply chain
`addon/` → `v090-sparktts-0001..0038` (numeric order) → `0001-orin-tegra-build-compat`.

**Dry-run verified 2026-07-04** (Mac, git worktree @ 1ac0f2b9, build.sh contract:
addon copy then `git apply --check` + `git apply` per patch): all 38 patches
CLEAN, then rebased `0001` CLEAN on top, then `0002` OPT-IN check also CLEAN on
the patched tree. Tracked-tree parity: staged worktree vs
`integration/v090-sparktts` diff = **36 files, ALL of them addon files (33) +
0001's 3 build-compat files — zero divergence in the 43 patched/added source
files** (`git diff --stat 1ac0f2b9..integration/v090-sparktts` = 43 files /
+6131 −182).

## Disposition table (v0.8.0 overlay → v0.9.0 overlay)
| patch | disposition | reason |
|---|---|---|
| `v080-sparktts-0001..0030` | **DELETED** | regenerated as `v090-sparktts-0001..0038` from the rebased fork branch `integration/v090-sparktts` (every v080 commit was re-picked/re-ported there; numbering differs because the rebase reordered the series and added the v0.9.0-only commits). |
| `0001-orin-tegra-build-compat` | **KEPT, in chain (last) — REBASED onto v0.9.0** | NOT absorbed upstream: v0.9.0 still lacks the Tegra autodetect, the aarch64 `CMAKE_CUDA_ARCHITECTURES` guard, the cublas/cublasLt link, the macOS-metadata GLOB filter, and still links the cutedsl shim/driver-lib/`--wrap` PRIVATE on static libs. Original patch FAILED `git apply --check` on v0.9.0 (`cmake/CuteDsl.cmake:775`, `cpp/CMakeLists.txt:69` context drift: new XQA-cubins lines + new `--wrap` version condition) → hand-rebased, all three hunks preserved, upstream's new `VERSION_GREATER_EQUAL 12.0` condition kept. Note: official v0.9.0 JP6.2 build docs now instruct `-DEMBEDDED_TARGET=jetson-orin` manually; the autodetect stays as a convenience. |
| `0002-weight-streaming-budget-v080-OPTIN` | **RENAMED → `0002-weight-streaming-budget-v090-OPTIN.patch`, content UNCHANGED, still NOT in chain** | re-verified 2026-07-04: `git apply --check` CLEAN on pristine v0.9.0 AND on the fully-patched tree (upstream v0.9.0 `builderUtils.cpp` still has no weight-streaming flag — not absorbed). OPT-IN only, as before. |
| `0006` / `0007` (server SSE-disconnect + OpenAI API, v0.7.1) | **ARCHIVAL (unchanged)** | still needs re-implementation against the rewritten server, not a rebase. SSE fix remains (a) PR-pending — do NOT auto-submit. |
| `0008-build-misc-example-registration` | **ARCHIVAL (unchanged)** | superseded registrations + v0.7.1 spike API. |
| `v080-0007` / `v080-0008` (pre-runtime-if CV) | **ARCHIVAL (unchanged)** | superseded by `v090-sparktts-0026/0029` (runtime-if on langId). |
| `v080-0001..0006/0010/0012/0024..0027` | **ARCHIVAL/DEFERRED (unchanged)** | incremental-KV spike track, C3 backlog. N>1 ASR = vendored `native/edgellm_voice_worker/qwen3_asr_worker.cpp` (v0.9.0-adapted by `port/v090-workers`, d7aa144) on the vanilla one-shot core. |
| `v080-0011-moss-tts-nano-port` | **ARCHIVAL (unchanged)** | MOSS single-sourced to the series (`v090-sparktts-0033/0034`) — see addon reconciliation. |

## addon/ reconciliation (41 → 33 files)
`v090-sparktts-0033/0034` bring MOSS **into the patch series** (worker moved to
`examples/omni/moss_tts_nano_worker.cpp`, CMake target). The 8 MOSS files were
therefore **REMOVED from `addon/`** (they would collide with the series'
"new file" patches at apply time):
```
cpp/kernels/kvCacheUtilKernels/mossLinearKvKernels.{cu,h}
cpp/runtime/mossTtsNanoRuntime.{cpp,h}
cpp/workers/build_moss_worker.sh          (series ships it as LEGACY reference)
cpp/workers/moss_tts_nano_worker.cpp      (superseded by examples/omni/ location)
unittests/mossLinearKvKernelsTests.cu
unittests/mossTtsNanoSmokeMain.cpp
```
Remaining 33 addon files are all additive and none exists in the v0.9.0 tree
or is created by the series (verified 2026-07-04).

## Dual build configuration (v0.9.0-specific fork) — see README for the full table
- **Voice workers** (build.sh default): `ENABLE_CUTE_DSL=OFF` + our own sm_87
  kernels (`v090-sparktts-0002/0003`).
- **GDN LLM engine** (separate build dir): `ENABLE_CUTE_DSL=ALL` + on-device
  sm_87 CuTe DSL artifact rebuild with cutlass-dsl 4.5.2
  (`python kernelSrcs/build_cutedsl.py --gpu_arch sm_87`; no sm_87 prebuilt
  tarball upstream) + 0001's cudart shim / `--wrap` propagation +
  `-DEMBEDDED_TARGET=jetson-orin`.

## ASR voice worker (vendored, outside the fork patch series)
`native/edgellm_voice_worker/{CMakeLists.txt,qwen3_asr_worker.cpp,spark_tts_worker.cpp}`
adapted to the v0.9.0 runtime API by `port/v090-workers` (d7aa144), merged into
this branch. C3 still owes moving the ASR worker into the fork as single source.

> Produced by **P4-1** (Mac, git/source only — no engine build, no deploy).
> Build-verify on an Orin sm_87 host is the next gate before any image re-bake.
