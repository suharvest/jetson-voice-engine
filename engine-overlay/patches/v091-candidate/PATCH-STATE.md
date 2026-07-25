# TensorRT Edge-LLM v0.9.1 candidate patch state

Status: **device-qualified source/build chain; artifact publication in progress**

Date: 2026-07-25

Official base: `v0.9.1`
(`7f061f21f0a581ba234a1e233c9315b89d8e47d6`)

`UPSTREAM_PIN` and `engine-overlay/build.sh` now make this the active,
deterministic source/build series. Promotion here means the repository build
contract has moved to v0.9.1; it does **not** mean production cutover or release
qualification. Fresh-engine model/concurrency gates and rollback-safe service
cutover remain mandatory.

## Series shape

The series contains 41 ordered patches:

- 36 semantic rebases of the active v0.9.0 patches;
- 4 independently scoped JetPack 6.2 compatibility fixes;
- 1 local MOSS concurrent-dispatch/cooperative-cancel extension.

The old development-only microbenchmark pair `v090-sparktts-0012` and
`v090-sparktts-0022` is omitted because its net product delta is zero. No
other behavior was deleted merely because runtime evidence is unavailable.

Because those two commits were removed, candidate numbering maps to the old
series as follows:

| Candidate | Previous active series |
|---|---|
| `0001`–`0011` | `0001`–`0011` |
| `0012`–`0020` | `0013`–`0021` |
| `0021` | `0023` |
| `0022`–`0036` | `0024`–`0038` |
| `0037`–`0040` | new v0.9.1 / JP6.2 compatibility fixes |
| `0041` | new local MOSS worker concurrency extension |

The final four patches are deliberately separate generic changes:

1. `0037`: use `IStreamReader` on TensorRT releases before 10.7 while
   preserving the official `IStreamReaderV2` path on newer releases;
2. `0038`: compile without `DataType::kFP4` on TensorRT releases before 10.8;
3. `0039`: propagate the selected CuTe cudart shim, CUDA driver library, and
   `_cudaLaunchKernelEx` wrap option through static-library consumers;
4. `0040`: key and load context-FMHA cubins by attention mask type.

The source for `0037`–`0039` is the read-only Orin evidence patch
`92-v091-jp62-combined-candidate.patch`; `0040` comes from
`115-mask-scoped-fmha-loader.patch`. Both are recorded under
`/home/harvest/validation/edgellm-v091-official-20260724T0418Z` on
`orin-nx`.

`0041` remains a product model extension. NVIDIA v0.9.1 has no MOSS model
target, so this patch is excluded from the upstream bug PR queue.

## Seven-patch rebase cluster

The v0.9.0 patches `0006`, `0014`, `0017`, `0019`, `0020`, `0023`, and
`0031` were semantically rebased instead of accepting rejected hunks:

- `0006`, `0014`, and `0019` place the opt-in BF16 residual fields and
  checkpoint parsing in the rewritten v0.9.1 `ModelConfig`;
- `0017` threads `exclude_attention` through the new multimodal/CodePredictor
  quantization signatures without dropping their new arguments;
- `0020` retains BF16 down-projection output for both AWQ layouts, including
  `int4_awq_modelopt`;
- `0023` renames only the product feature to `bf16_residual`; the unrelated
  official `QUANT_MIXED = "mixed_precision"` parsing remains untouched;
- `0031` retains shared-engine ownership and reverse slot destruction while
  also preserving v0.9.1 auxiliary-stream setup, optional tree-metadata
  registration, and the new `hasIOTensor` path for borrowed engines.

These resolutions preserve source intent only. In particular, shared-engine
ownership, cancellation, PCM isolation, and BF16/W4A16 correctness remain
unverified until the device runtime matrix passes.

## Reproducible apply verification

Verification used a second clean checkout of the exact official SHA. The 33
files from `engine-overlay/addon/` were copied first and committed as the
local baseline, then this directory was replayed in lexical order.

Results:

```text
ordered mail-patch replay:  40/40 clean
sequential reverse replay: 40/40 clean
post-reverse addon match:  yes
sequential forward replay: 40/40 clean
git diff --check:           clean
python compileall:          pass
0041 apply atop 0001-0040: clean
0041 git diff --check:      clean
```

The reverse count is a reverse-order replay of the complete series, not an
independent reverse check of each patch against the final tree. Later commits
intentionally edit fields introduced by earlier commits.

Local Python unit execution is dependency-blocked:

```text
tests/python-unittests/test_loader_dtype_cast.py: skipped (torch unavailable)
tests/test_bf16_residual_int4.py: collection blocked (torch unavailable)
```

This is recorded as `unverified`, not a pass and not a reason to remove either
test or behavior.

## Orin NX build evidence

The complete 40-patch source plus locked v0.9.1 submodules built on JetPack
6.2 / CUDA 12.6 / TensorRT 10.3 for SM87 in two isolated build directories:

- fallback (`ENABLE_CUTE_DSL=OFF`): plugin, core, `llm_build`, Qwen TTS,
  MOSS, Qwen ASR, and SparkTTS targets all returned 0;
- CuTe (`ENABLE_CUTE_DSL=ALL`): the same targets returned 0 using the
  device-generated SM87/CUDA 12.6.68/cutlass-dsl 4.5.1 artifact;
- final audits found no SM80/86/89 target flags; the CuTe build propagated
  the selected artifact and CUDA 12.6 launch shim/wrap into final consumers.

Fresh v0.9.1 engines subsequently passed the full model matrix: GDN+MTP,
Qwen3-ASR b1/b2, CustomVoice FP16/INT4, Base INT4 plus speaker encoder,
SparkTTS BF16/W4A16, MOSS, and SenseVoice. ASR and supported TTS paths passed
their N=2/cancel/recovery gates. The `0041` MOSS worker then passed 50/50 true
N=2 cancel/continue/recovery rounds with no CUDA errors. Full details and
hashes are in `docs/validation/tensorrt-edge-llm-v0.9.1-orin-nx-voice.md` and
`docs/validation/edgellm-v091-model-concurrency-matrix.md`.

## Remaining publication and upstream-retirement gates

The device runtime migration is qualified and production GDN+MTP is already
cut over with the v0.8 image/compose preserved for rollback. Remaining work is
not a model-validation blocker:

- complete and remotely verify the versioned Hugging Face artifact upload;
- build the final voice image from the normalized artifact tree and run its
  profile preload checks;
- compile the TRT-version guards once against a newer TensorRT;
- add a true custom-mask negative test before proposing the FMHA patch;
- split the generic CuTe CMake propagation from the local shim implementation;
- retain the non-CuTe fallback patches until their documented retirement gates
  are met.
