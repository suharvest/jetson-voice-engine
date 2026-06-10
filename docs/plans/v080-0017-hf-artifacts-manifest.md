# v080-0017 — v0.8.0 migration artifacts: HF upload + MANIFEST

**Date:** 2026-06-10
**Source host:** orin-nx (`Linux 5.15.148-tegra aarch64`, hostname `orinnx`, Orin NX 16GB, JetPack 6.2, CUDA 12.6.68, TRT 10.3.0.30)
**Relay host:** wsl2-local (`x86_64`) — orin-nx cannot direct-upload to hf.co (ConnectError); upload relayed via wsl2-local through the Clash proxy.
**Scope:** Task #17 — publish every v0.8.0-built runtime artifact (engines + serving worker binaries + edgellm plugin) to Hugging Face so the v0.8.0 Docker image (#7) can pull them at deploy time.

**Toolchain tag:** `sm_87-trt10.3.0.30-cuda12.6.68-jp6.2`
**edgellm base:** `f9cc746` (Merge #101 dev-release/0.8.0) + port patches v080-0007 / v080-0008.

---

## HF repo + layout

- **Repo:** `harvestsu/seeed-local-voice-artifacts` (model repo)
- **Base path:** `sm87-trt10.3-jp6.2/v0.8.0/`

```
sm87-trt10.3-jp6.2/v0.8.0/
  engines/
    asr/llm/        llm.engine + embedding.safetensors + config/tokenizer/chat-template
    asr/audio/      audio_encoder.engine + config.json
    qwen3-tts/talker/         llm.engine + embedding/text_embedding/text_projection + tokenizer
    qwen3-tts/code_predictor/ llm.engine + codec_embeddings + lm_heads + config
    qwen3-tts/code2wav/       code2wav.engine + config.json
    gdn/            llm.engine + embedding + external_int4_ffn_weights + tokenizer
  workers/          qwen3_asr_worker, qwen3_tts_worker, moss_tts_nano_worker
  plugins/          libNvInfer_edgellm_plugin.so
  toolchain/        cutedsl_aarch64_sm_87_cuda12.tar.gz (build-time reference only)
```

## Upload command (run on wsl2-local)

```bash
# HF_ENDPOINT must be overridden — the device default is the hf-mirror.com mirror,
# which does NOT accept uploads. The cached token (~/.cache/huggingface/token,
# user harvestsu) authenticates against both endpoints.
HF_ENDPOINT=https://huggingface.co ~/.local/bin/hf upload \
  harvestsu/seeed-local-voice-artifacts \
  /home/harve/v080-hf-stage  sm87-trt10.3-jp6.2/v0.8.0 \
  --repo-type model \
  --commit-message "v0.8.0 migration artifacts (sm87/trt10.3/jp6.2): engines + workers + plugin"
```

## MANIFEST

Each file: path (relative to base) · size (bytes) · md5. 34 files, 7.577 GB total.

| Path | Size | md5 |
|---|---|---|
| engines/asr/audio/audio_encoder.engine | 377555460 | 5c877cfe58b8fcb7914679c6fe274f90 |
| engines/asr/audio/config.json | 2564 | 0ad147209168659dc23a9081472c718d |
| engines/asr/llm/config.json | 943 | 841508805eaa3709eaa62809e9af4326 |
| engines/asr/llm/embedding.safetensors | 311165016 | 8db9ceda288e2470694a0bc33dbfd381 |
| engines/asr/llm/llm.engine | 1212033660 | b133dff24c8aa96ac1679b95e2f97153 |
| engines/asr/llm/processed_chat_template.json | 672 | 3ed0c38f55940e126860150dc2b83c12 |
| engines/asr/llm/tokenizer.json | 11429653 | 68e0da75e29b5190b2b98c2e8a49d2e2 |
| engines/asr/llm/tokenizer_config.json | 998 | 6017cd191f466ced4e7c63f20f193726 |
| engines/gdn/config.json | 14058 | 8316d35721ce9c815c86989c538bf463 |
| engines/gdn/embedding.safetensors | 1271398496 | e570d701f92a47a8643e4a133cde2b10 |
| engines/gdn/external_int4_ffn_weights.safetensors | 1167874240 | 4d57d41ec0ae54f0d9e2ef097e3b1a04 |
| engines/gdn/llm.engine | 1019519468 | afcb055b67bbc33d3dacf5491e4719d5 |
| engines/gdn/processed_chat_template.json | 763 | 1b8bdc580a0e641f3dd996bd6fb5899c |
| engines/gdn/tokenizer.json | 19989609 | 579ec1208991293fc3a6cf9c70a2663a |
| engines/gdn/tokenizer_config.json | 1098 | b651f82f420582b4717832a688db972d |
| engines/qwen3-tts/code2wav/code2wav.engine | 235287356 | 566c389eea9337195e6b58a846a4ed39 |
| engines/qwen3-tts/code2wav/config.json | 896 | c6535b47286f069ad04f44378a796e82 |
| engines/qwen3-tts/code_predictor/codec_embeddings.safetensors | 62915840 | 0352bb3fe1e228388eae4753f13f8d8f |
| engines/qwen3-tts/code_predictor/config.json | 722 | 92271d44d2cf50d829d58da88a9093ea |
| engines/qwen3-tts/code_predictor/llm.engine | 191037508 | baff21ea48a9f7e8de30e3d698544e4d |
| engines/qwen3-tts/code_predictor/lm_heads.safetensors | 62915920 | 63260bf53193b66edffdf8506a4ae2e6 |
| engines/qwen3-tts/talker/config.json | 1723 | 8436ee0c367d75c8d72514ae302bb2a8 |
| engines/qwen3-tts/talker/embedding.safetensors | 6291544 | 2632299988589ad4d96e151286a4902e |
| engines/qwen3-tts/talker/llm.engine | 906971780 | 471d36d8f730f560a6949c741f49e6ae |
| engines/qwen3-tts/talker/processed_chat_template.json | 560 | 75239a8b80836a43f45f4e57dbb32b9d |
| engines/qwen3-tts/talker/text_embedding.safetensors | 622329952 | 0b2ad4232518334c4aaa2402ca5ed5e7 |
| engines/qwen3-tts/talker/text_projection.safetensors | 12589400 | 64ca22bd6e83bb8c1d242b3e226dd021 |
| engines/qwen3-tts/talker/tokenizer.json | 11424262 | 5a307103ca203cc774119502276c70cb |
| engines/qwen3-tts/talker/tokenizer_config.json | 956 | d21bfd50d2f738da17c7360c4caf3758 |
| plugins/libNvInfer_edgellm_plugin.so | 45948424 | 8f004bb4c9ddcce30ae4eecf2f410624 |
| toolchain/cutedsl_aarch64_sm_87_cuda12.tar.gz | 195655 | cd29d94e411e929b0521dca5a035335c |
| workers/moss_tts_nano_worker | 562272 | 6a03bdf5c7a26b09f60597b95008ebfe |
| workers/qwen3_asr_worker | 13921904 | be7bee91728c63253e5926a5933896c0 |
| workers/qwen3_tts_worker | 14090720 | 22216e8dc724bd8619d4fca26b0c2d5b |

## Notes

- **CuTe DSL sm_87** is statically linked into the plugin/executable at build time via the `--wrap` shim (PORT_NOTES v080-0008 Blocker A). The `toolchain/cutedsl_aarch64_sm_87_cuda12.tar.gz` tarball is shipped as a **build-time reference only**, NOT a runtime dependency.
- **MOSS-TTS-Nano engines** already live on HF at `harvestsu/seeed-local-voice-artifacts/models/moss-tts-nano/engines` (SKIP re-upload). The MOSS *worker binary* (`moss_tts_nano_worker`, md5 `6a03bdf5`, built by `cpp/workers/build_moss_worker.sh` to `/tmp/moss_tts_nano_worker`) IS included here under `workers/`.
- **ASR plugin:** v0.8.0 uses a single `libNvInfer_edgellm_plugin.so`; no separate `_asr.so` was built for v080 (the pre-v080 deploy tree's `_asr.so` is stale, May-16, and was not uploaded).
- All 34 md5s were verified identical orin-nx (source) == wsl2-local (relay) == HF (post-upload) before this doc was finalized.

## Verdict

All v0.8.0 runtime artifacts (ASR / TTS / GDN engines + sidecars, 3 serving workers, edgellm plugin) are published under `sm87-trt10.3-jp6.2/v0.8.0/`. MOSS engines already present separately. **Ready for #7 Docker image to pull at deploy.**
