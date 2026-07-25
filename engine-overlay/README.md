# voxedge-engine

Thin **overlay** repository for the edge inference engine: it carries the NVIDIA
TensorRT-Edge-LLM upstream pin plus voxedge's `addon/` files, `patches/`,
build wrapper, divergence ledger, and build-reproduction manifests — **without
vendoring the full NVIDIA source tree**.

## Overlay model

```
voxedge-engine/
  UPSTREAM_PIN       # exact NVIDIA commit (= tag v0.9.1, 7f061f21…)
  upstream.remote    # https://github.com/NVIDIA/TensorRT-Edge-LLM.git
  patches/upstream-v091-prs/
                     # 7 byte-locked exact commits from PR #118/#145–149
  addon/             # new files (upstream does not have these), original relative paths
  patches/v091-candidate/
                     # explicit sparse 36-patch product series
  build.sh           # pin → upstream PR patches → addon → product patches → build
  manifests/         # build-reproduction manifests (qwen3-tts / qwen3-asr / customvoice)
  DIVERGENCE.md      # per-topic (a)/(b) classification + upstream-PR / retirement plan
```

> **v0.9.1 normalized migration (2026-07-25):** the active base is pure
> NVIDIA tag v0.9.1, `7f061f21`. Seven exact commits from PR #118 and
> #145–149 are vendored with commit/tree/patch-id/SHA-256 provenance, then the
> 36-entry sparse product series is applied. Generic local duplicates `0033`,
> `0034`, `0037`, `0038`, and `0040` were retired; mixed `0009` and `0039`
> retain only their product/residual hunks. See
> `patches/v091-candidate/PATCH-STATE.md`. Old v0.8/v0.9.0 files are retained
> as rollback/history and are never mixed into v0.9.1 images.

## Dual build configuration (v0.9.1)

Two mutually exclusive cmake configurations, per artifact family:

| build | `ENABLE_CUTE_DSL` | notes |
|---|---|---|
| **Voice workers** | `OFF` (fallback) or qualified local SM87 artifact | The local GEMM/GEMV path remains until fresh-engine quality/performance gates show that CuTe can replace it. |
| **GDN LLM engine** | `ALL` | On Orin/JP6.2 regenerate SM87 with the device-qualified cutlass-dsl 4.5.1 toolchain. The packaged v0.9.1 archive was generated with CUDA 13.2 and is not usable with CUDA 12.6. Configure with `AARCH64_BUILD=ON`, SM87, and `EMBEDDED_TARGET=jetson-orin`; PR #118 propagates shim/wrap requirements and residual local `0039` temporarily propagates the CUDA driver edge. |

Never mix the two configurations in one build dir.

The full source tree is **reconstructed at build time**: clone upstream at
`UPSTREAM_PIN`, verify/apply the locked upstream series, copy `addon/`, apply
the explicit local `series`, then build.

## addon vs patch discipline

| change type                     | destination          | rule                              |
|---------------------------------|----------------------|-----------------------------------|
| file upstream does not have     | `addon/<rel-path>`   | copied verbatim onto the checkout |
| modification to an upstream file| `patches/NNNN-*.patch` | minimal, reviewable, replayable |

Extracted from fork `v071/customvoice-product` (HEAD `893ba2a`, 90 commits ahead
of pin). Full diff against pin: **A=40 → addon, M=27 → patches, D=0**.

> The A/M/D counts below describe the original v0.7.1 extraction and are kept
> for provenance; the active inventory is documented in the v0.9.1 patch state.

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

Full builds parse the TOML manifest and verify its pinned upstream/local
series, LOCK, and SHA256SUMS hashes before clone, submodule initialization, or
compilation. `--apply-only` intentionally has no artifact manifest.

Offline provenance gates:

```bash
tests/verify-patch-stack.sh /path/to/official-checkout-with-locked-objects
tests/test-provenance-negative.sh /path/to/official-checkout-with-locked-objects
```

The replay source may be a normal clone or a standard linked Git worktree;
validation uses Git plumbing and does not assume `.git` is a directory. The
temporary replay tree is materialized from the exact PIN only, so unrelated
broken/partial refs in the source cannot poison the gate.

## Build-verify status

The predecessor 41-patch source built on Orin NX in both fallback and
device-generated CuTe configurations. The normalized 7+36 identity has clean
offline replay and must be rebuilt on Orin before its own artifacts are
published. `build.sh` refuses to compile on non-aarch64 and supports
`--apply-only` for source-chain verification.
