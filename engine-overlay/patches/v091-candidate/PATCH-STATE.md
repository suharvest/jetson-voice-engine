# TensorRT Edge-LLM v0.9.1 patch state

Status: **shrunk local stack; clean replay and Orin product A/B qualified**

Date: 2026-07-25

Official base: NVIDIA tag `v0.9.1`
(`7f061f21f0a581ba234a1e233c9315b89d8e47d6`)

## Active source contract

The active source is reconstructed in this order:

1. official NVIDIA v0.9.1;
2. seven exact commits proposed to NVIDIA in PR #118 and #145–149;
3. additive product files from `addon/`;
4. the explicit sparse 35-patch product series in `series`.

The proposed-upstream patches are byte-locked in
`../upstream-v091-prs/{series,LOCK,SHA256SUMS}`. The local patch bytes are
locked by `SHA256SUMS` beside this file. `LOCK` records official
repository URL, PR, commit, parent, tree, stable patch-id, and file SHA-256.
The build never follows a mutable PR head.

The local series deliberately keeps its historical numbers. Gaps are
retirement evidence, not missing files, so `build.sh` reads `series` instead
of inferring a contiguous glob.

## Retired duplicates

These local patches are fully retired because the exact generic fix is now
applied first from the proposed-upstream set:

| Retired local patch | Replacement |
|---|---|
| `0033` checkpoint destination dtype | PR #149 / upstream patch `0007` |
| `0034` ASR MRoPE normalization | PR #146 / upstream patch `0004` |
| `0037` TensorRT stream reader compatibility | PR #147 / upstream patch `0005` |
| `0038` pre-10.8 FP4 guard | PR #145 / upstream patch `0003` |
| `0039` CUDA driver PUBLIC propagation | Retired by normalized Orin product A/B; PR #118 already supplies the required generic shim/wrap propagation |
| `0040` mask-scoped FMHA cubin load | PR #148 / upstream patch `0006` |

One mixed patch was reduced instead of removed:

- local `0009` now contains only the BF16Linear tied-weight product extension;
  generic destination-dtype preservation comes from PR #149;

Local `0039` is not an upstream candidate. It was a downstream link-interface
workaround, not a generic functional fix. In the normalized Orin product A/B,
the variant without `0039` built the plugin, `llm_inference`, Qwen3 TTS, MOSS,
ASR, and Spark workers. Final link still retained wrap/CuTe/shim/libcuda and
`ldd -r` passed, proving the residual PUBLIC driver edge redundant. PR #118's
generic CuTe shim and `_cudaLaunchKernelEx` wrap propagation remains in the
locked proposed-upstream series.

## Product capability retained

All model/product behavior remains local:

- `0001`–`0003`: Orin voice fallback GEMM/GEMV and executable shim consumer;
- `0004`–`0005`: ASR lane/session concurrency;
- `0006`–`0021`: Spark/BF16/W4A16 modeling, export, kernels, and plugins;
- `0022`–`0030`: Qwen3 TTS streaming, Base/CustomVoice conditioning,
  shared-engine slots, cancellation, and chunk policy;
- `0031`–`0032`, `0041`: MOSS runtime/kernel/worker plus true concurrent
  dispatch and cooperative cancellation;
- `0035`–`0036`: Base export guard and CustomVoice language-id export.

NVIDIA v0.9.1 does not provide the MOSS or Spark model integrations, the Base
speaker-conditioning extension, or these ASR/TTS worker concurrency policies.
They are not duplicates of the seven generic bug fixes.

## Local replay evidence

The stack was tested against a clean checkout of the exact official base:

```text
vendored format-patch byte comparison: 7/7 exact
vendored SHA-256 verification:          7/7 pass
proposed-upstream forward replay:       7/7 clean
local sparse forward replay:           35/35 clean
git diff --check after forward replay: clean
local reverse replay:                  35/35 clean
proposed-upstream reverse replay:       7/7 clean
post-reverse tracked tree:             official v0.9.1
post-reverse untracked tree:           addon/ only
```

`tests/verify-patch-stack.sh` performs the offline integrity checks and can
optionally replay against a supplied clean upstream checkout. `build.sh
--apply-only` exercises the complete materialization contract.

For release builds, `build.sh` additionally parses the selected TOML manifest
and verifies the recorded hashes for upstream `series`, `LOCK`,
`SHA256SUMS`, local `series`, and local `SHA256SUMS` before any clone/build
work. The integrity test binds LOCK and both checksum files to the exact
series order/set and, when official objects are supplied, verifies every
commit's real parent, tree, and patch-id. Provenance inputs are marked
`-text`; exact upstream mail patches are also `-whitespace`, so
`core.autocrlf=true` cannot change release bytes.

Replay materialization uses the source's read-only common object store and
the exact PIN index/tree rather than archive or clone-all-refs. This preserves
the three v0.9.1 gitlinks plus executable/symlink modes and fails unless
`write-tree` equals the PIN tree before apply and after reverse.

## Device and release gate

The former 41-patch chain passed Orin NX fallback/CuTe builds and the complete
model/concurrency matrix. That is strong behavior-preservation evidence, but
it is not attributed to this normalized 7+35 source identity.

Before publishing a new artifact set, the normalized chain must pass:

- Orin NX CUDA 12.6 / TensorRT 10.3 build and focused bug gates;
- fresh engine/model, N=1 baseline, supported N=2, cancellation/recovery, and
  ASR+TTS+LLM co-residency regression;
- a new immutable artifact prefix and manifest. The existing 41-patch artifact
  set and its checksums must not be overwritten.
