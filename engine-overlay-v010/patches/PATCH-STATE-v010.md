# TensorRT-Edge-LLM v0.10.0 overlay patch state

Base: NVIDIA `v0.10.0` at `71dd1bae032e70771265917ec74d3ff4cad07a10`
(`6ade7849c470eb2becf669b9470e7385f1454e3d` tree).

## Proposed-upstream series (active: 4)

| overlay patch | origin | v0.10 disposition |
|---|---|---|
| `upstream-v010-prs/0001-pr118-respect-explicit-cuda-architectures.patch` | PR #118 | retained, rebased onto v0.10.0 |
| `upstream-v010-prs/0002-pr118-static-target-wrap-interface.patch` | PR #118 | retained as the only CuTe delta still missing upstream: static target `INTERFACE` `--wrap=_cudaLaunchKernelEx` propagation; shim link propagation is already in v0.10.0 |
| `upstream-v010-prs/0003-pr146-normalize-linear-mrope.patch` | PR #146 | retained, rebased onto v0.10.0 |
| `upstream-v010-prs/0004-pr149-checkpoint-dtype.patch` | PR #149 | retained, rebased onto v0.10.0 |

The following v0.9.1 proposed-upstream patches are deliberately retired and
must not be copied into the v0.10 series:

| retired topic | reason |
|---|---|
| PR #145 FP4 guard | v0.10.0 already carries the TensorRT-version guard |
| PR #147 stream reader | v0.10.0 already carries the pre-10.7 compatibility path |
| PR #148 legacy FMHA mask-scoped load | v0.10.0 removes that legacy embedded-cubin backend and replaces it with CuTe DSL FMHA-v2 |

`series`, `LOCK`, and `SHA256SUMS` are the source of truth for the four files.
The rebased commit IDs in `LOCK` are overlay-local provenance records; the
patches are applied as diffs to the official v0.10.0 tree.

## Product patch series (active: 32)

The active `patches/v010-candidate/series` contains 32 sparse product patches.
Old 0024, 0035, and 0036 retired because v0.10.0 provides their language,
Base/export, and clone metadata behavior natively. The remaining patches were
rebased rather than copied: Qwen3-TTS clone/language wiring now uses the native
v0.10 request and clone-encoder interfaces, while Spark/MOSS/ASR product paths
retain their explicit local contracts.

`build.sh --apply-only` and `tests/verify-patch-stack.sh` both require and
verify the complete 4+32 chain. Integrated forward/reverse replay against the
official v0.10.0 tree passes; Orin compile, engine regeneration, and runtime
qualification remain release gates.
