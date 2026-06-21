# voxedge-engine

Thin **overlay** repository for the edge inference engine: it carries the NVIDIA
TensorRT-Edge-LLM upstream pin plus voxedge's `addon/` files, `patches/`,
build wrapper, divergence ledger, and build-reproduction manifests — **without
vendoring the full NVIDIA source tree**.

## Overlay model

```
voxedge-engine/
  UPSTREAM_PIN       # exact NVIDIA commit (= v0.8.0, f9cc746…, release/0.8.0 HEAD)
  upstream.remote    # https://github.com/NVIDIA/TensorRT-Edge-LLM.git
  addon/             # new files (upstream does not have these), original relative paths
  patches/           # legacy 000N theme patches (v0.7.1 era) + v080-port-000N
                     #   (fork port/qwen3-tts-base-v080 TTS runtime) + v080-NNNN
                     #   feat patches. SEE patches/PATCH-STATE-v080.md for which
                     #   apply on v0.8.0 and which are pending rebase.
  build.sh           # clone upstream@pin → copy addon → apply patches → build (Jetson host)
  manifests/         # build-reproduction manifests (qwen3-tts / qwen3-asr / customvoice)
  DIVERGENCE.md      # per-topic (a)/(b) classification + upstream-PR / retirement plan
```

> **v0.8.0 re-pin (C2a):** pin moved v0.7.1 `364769…` → v0.8.0 `f9cc746…`. The
> canonical TTS-runtime source is now the fork branch `port/qwen3-tts-base-v080`
> (6 commits / 9 files), regenerated into `patches/v080-port-000N`. Of the legacy
> `000N` patches only `0001` clean-applies on v0.8.0; `0002`–`0007` are superseded
> and `0008` needs a hunk-split. The `v080-NNNN` feat series (ASR/MOSS/CV/TTS-batch)
> is pending rebase. Full apply-check evidence: `patches/PATCH-STATE-v080.md`.

The full source tree is **reconstructed at build time**: clone upstream at
`UPSTREAM_PIN`, copy `addon/` over it, apply `patches/*.patch` in order, build.

## addon vs patch discipline

| change type                     | destination          | rule                              |
|---------------------------------|----------------------|-----------------------------------|
| file upstream does not have     | `addon/<rel-path>`   | copied verbatim onto the checkout |
| modification to an upstream file| `patches/NNNN-*.patch` | minimal, reviewable, replayable |

Extracted from fork `v071/customvoice-product` (HEAD `893ba2a`, 90 commits ahead
of pin). Full diff against pin: **A=40 → addon, M=27 → patches, D=0**.

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
