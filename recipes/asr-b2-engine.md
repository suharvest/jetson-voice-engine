# Qwen3-ASR thinker engine — `maxBatchSize=2` (N>1 ASR)

The N>1 ASR concurrency engine: the same v0.8.0 Qwen3-ASR thinker ONNX, rebuilt
with `--maxBatchSize 2` so two concurrent ASR sessions share one native-batch
engine (verified GREEN: two concurrent sessions, isolation OK, 0 CUDA errors).
`--maxBatchSize` is a **build-time-only** `llm_build` flag — no re-export, same
ONNX as the maxBatch=1 engine.

- **Artifact:** `engines-v080-b2/llm/llm.engine` (+ `config.json` / tokenizer /
  chat_template auto-copied by `llm_build`). ~1.21 GB.
- **Inputs:** v0.8.0 ASR thinker ONNX `onnx-v080/llm`, exported from
  `Qwen/Qwen3-ASR-0.6B` with the v0.8.0 schema (`edgellm_version 0.8.0`,
  `kv_cache_dtype fp16`) via `tensorrt-edgellm-export-llm --fp8_embedding`.
  (Export recipe: [`../docs/asr-thinker-engine-build-recipe.md`](../docs/asr-thinker-engine-build-recipe.md).)
- **Pin:** no fork driver — pure `llm_build` engine build. Builder = the C++
  `build/examples/llm/llm_build` from the overlay build (see
  [`slot-pool-worker.md`](slot-pool-worker.md), overlay `UPSTREAM_PIN`).
- **md5:** `4122dfcc666fe82b8b0cae4b93c97b70` — `engines-v080-b2/llm/llm.engine`
  (1.21 GB). (maxBatch=1 reference `b133dff24c8aa96ac1679b95e2f97153`.)
  **Download-integrity only — a rebuild will not match (TRT autotuning).**
- **HF:** `harvestsu/seeed-local-voice-artifacts` → `asr-b2`.
- **Build device:** **orin-nx** (`Linux aarch64`, hostname `orinnx`, Orin NX
  16GB, JetPack 6, CUDA 12.6, TRT 10.3, SM 87). Loadable on Orin Nano (also
  SM 87) given matching TRT minor + plugin .so.
- **Provenance:** `edgellm-v080-migration/docs/plans/v080-phase5a-acceptance-results.md`.

## Command

```bash
WS=<workspace>/Qwen3-ASR-0.6B
./build/examples/llm/llm_build \
    --onnxDir   $WS/onnx-v080/llm \
    --engineDir $WS/engines-v080-b2/llm \
    --maxBatchSize 2 --maxInputLen 4096 --maxKVCacheCapacity 4096
# → [llm_build.cpp:246:main] LLM engine built successfully.
```

`parseEngineConfig` should report `maxBatch=2 maxInputLen=4096 maxKVCapacity=4096`.

> Build entrypoint is the repo's C++ `llm_build` ONLY — never a hand-rolled
> cmake/quantization script. The build-time plugin `.so` must be byte-compatible
> with the runtime worker's plugin (same overlay pin).

## Verify

```bash
jq '{max_input_len, max_kv_cache_capacity, max_batch_size}' $WS/engines-v080-b2/llm/config.json
md5sum $WS/engines-v080-b2/llm/llm.engine    # integrity check vs published md5
```

Functional gate: two concurrent ASR sessions over the b2 engine transcribe
their own audio correctly (no cross-talk).
