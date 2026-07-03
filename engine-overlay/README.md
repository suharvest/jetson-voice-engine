# voxedge-engine

Thin **overlay** repository for the edge inference engine: it carries the NVIDIA
TensorRT-Edge-LLM upstream pin plus voxedge's `addon/` files, `patches/`,
build wrapper, divergence ledger, and build-reproduction manifests — **without
vendoring the full NVIDIA source tree**.

## Overlay model

```
voxedge-engine/
  UPSTREAM_PIN       # exact NVIDIA commit (= tag v0.9.0, 1ac0f2b9…)
  upstream.remote    # https://github.com/NVIDIA/TensorRT-Edge-LLM.git
  addon/             # new files (upstream does not have these), original relative paths
  patches/           # v090-sparktts-0001..0038 (THE apply series — fork
                     #   integration/v090-sparktts regenerated as format-patches)
                     #   + 0001 build-compat (rebased onto v0.9.0)
                     #   + 0002 OPT-IN + archival legacy/v080-NNNN patches.
                     #   SEE patches/PATCH-STATE-v090.md for dispositions.
  build.sh           # clone upstream@pin → copy addon → apply patches → build (Jetson host)
  manifests/         # build-reproduction manifests (qwen3-tts / qwen3-asr / customvoice)
  DIVERGENCE.md      # per-topic (a)/(b) classification + upstream-PR / retirement plan
```

> **P4-1 v0.9.0 re-pin (2026-07-04):** pin = pure NVIDIA tag v0.9.0
> `1ac0f2b9`; ALL fork content travels as `patches/v090-sparktts-0001..0038`
> (= `git format-patch --no-stat 1ac0f2b9..integration/v090-sparktts`, fork
> HEAD `e8c59c1`: streaming worker re-ported to the v0.9.0 native streaming
> API + CV 9-row runtime-if + external speaker-embedding + SparkTTS
> bf16/int4/W4A16 mixed-precision (renamed `bf16_residual`) + shared-engine
> ctors + cooperative cancel + MOSS-TTS-Nano in-series + export fixes).
> Apply chain = addon/ + the 38-patch series + `0001-orin-tegra-build-compat`
> (rebased onto v0.9.0 — NOT absorbed upstream); dry-run verified — resulting
> tree == the fork integration branch exactly. MOSS files were REMOVED from
> `addon/` (now created by `v090-sparktts-0033/0034`). Legacy `0002` renamed
> `-v090-OPTIN` (re-verified clean on v0.9.0, unchanged); the old
> `v080-sparktts-0001..0030` series is deleted; `0006`/`0007`/`0008` +
> `v080-NNNN` stay archival. Details: `patches/PATCH-STATE-v090.md`.

## Dual build configuration (v0.9.0)

Two mutually exclusive cmake configurations, per artifact family:

| build | `ENABLE_CUTE_DSL` | notes |
|---|---|---|
| **Voice workers** (this overlay's `build.sh`: TTS streaming worker, ASR worker, MOSS, plugin) | `OFF` (default) | Talker hot path uses our own sm_87 kernels from the series (`v090-sparktts-0002/0003` tiled GEMM + M=1 GEMV); CuTe DSL not needed. `ALL` is a link-time trap on JP6.2/CUDA 12.6 without the artifact rebuild below. |
| **GDN LLM engine** (Qwen3.5 GDN/MTP; separate build dir, not this script) | `ALL` | GDN kernels are CuTe-DSL-only. On Orin sm_87/JP6.2 requires: ① regenerate the sm_87 CuTe DSL artifact on-device with **cutlass-dsl 4.5.2** (`python kernelSrcs/build_cutedsl.py --gpu_arch sm_87` — upstream v0.9.0 ships no sm_87 prebuilt tarball), ② patch `0001`'s cudart shim + `--wrap=_cudaLaunchKernelEx` PUBLIC/INTERFACE propagation (final-exe link fails on CUDA < 12.8 without it), ③ `-DEMBEDDED_TARGET=jetson-orin` (autodetected on Tegra by `0001`). |

Never mix the two configurations in one build dir.

The full source tree is **reconstructed at build time**: clone upstream at
`UPSTREAM_PIN`, copy `addon/` over it, apply `patches/*.patch` in order, build.

## addon vs patch discipline

| change type                     | destination          | rule                              |
|---------------------------------|----------------------|-----------------------------------|
| file upstream does not have     | `addon/<rel-path>`   | copied verbatim onto the checkout |
| modification to an upstream file| `patches/NNNN-*.patch` | minimal, reviewable, replayable |

Extracted from fork `v071/customvoice-product` (HEAD `893ba2a`, 90 commits ahead
of pin). Full diff against pin: **A=40 → addon, M=27 → patches, D=0**.

> The A/M/D counts below describe the original v0.7.1 extraction and are kept
> for provenance; current file inventory is per the v0.9.0 re-pin note above.

## (a) upstreamable vs (b) carried

Every divergence is classified in `DIVERGENCE.md`:

- **(a) upstreamable** — generic fixes / upstream-side bugs → file a PR, and on
  merge **retire the patch** (bump `UPSTREAM_PIN` past the merge, delete the
  patch) to avoid double-apply conflicts.
- **(b) carried** — our product/model/perf-path specifics upstream won't take →
  stay long-term in `addon/` + `patches/`.

Three patches are **mixed** (contain both (a) and (b) hunks in the same files and
must be hunk-split before any upstream PR): `0001` (build-compat + private
kernel/plugin registration), `0002` (weight-streaming + shared-engine ctor /
audioIndexBase), `0006` (SSE-disconnect + OpenAI-API server, interleaved in
commit history so not cleanly base-splittable on the read-only fork).

> **SSE/client-disconnect fix** (commit `0898b5f`, inside `0006`) is **(a)
> PR-pending** — **do NOT auto-submit**; the user assigns a filer.

## Reproduction

```bash
# Materialize the patched source tree only (no CUDA/TRT needed, any host):
./build.sh --apply-only

# Full build (Jetson Orin / sm_87 host with CUDA + TensorRT only):
./build.sh manifests/qwen3-tts-highperf-sm87.toml
```

## Build-verify status

**DEFERRED — requires a Jetson CUDA/TensorRT host (Orin, sm_87).** This overlay
was extracted on a macOS dev box that has no CUDA/TRT toolchain, so only
structure extraction + integrity reconciliation were performed. The compile,
plugin build, engine build, and artifact-checksum steps must run on an Orin
build host. `build.sh` refuses to compile on non-aarch64 and documents the
Jetson entry points.
