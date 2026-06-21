# Qwen3-TTS Talker engine — `maxBatchSize=2` (N>1 TTS slot-pool)

The N>1 TTS engine: one `maxBatchSize=2` Talker engine that natively batches two
lanes, replacing per-session runtime replication for the TTS slot-pool.
Verified GREEN: two concurrent lanes produce per-lane RVQ codes **and** full
audio waveforms byte-identical to their solo runs (0 cross-talk, 0 CUDA errors).
Same build-flag-only path as the ASR b2 engine — no re-export, reuses the
on-box maxBatch=1 Talker ONNX.

- **Artifact:** `engines-v080-tts-b2/talker/llm.engine` (+ config / tokenizer /
  chat_template auto-copied by `llm_build`, identical set to maxBatch=1 dir).
- **Inputs:** the on-box Talker ONNX `onnx/llm/` (high-perf fork layout:
  `model_type: llm`, Talker is `talker/llm.engine` built from `onnx/llm/`).
- **Pin:** no fork driver — pure `llm_build` engine build. Builder = overlay
  `build/examples/llm/llm_build` (overlay `UPSTREAM_PIN` `a361221`; the slot-pool
  lane-concurrency runtime is patch `v080-0010-tts-batch-lane-concurrency.patch`).
- **md5:** `f7339e02a32c89fc5f96254ce2719f4e` —
  `engines-v080-tts-b2/talker/llm.engine`. (maxBatch=1 reference
  `471d36d8f730f560a6949c741f49e6ae`.)
  **Download-integrity only — TRT autotuning means a rebuild will not match.**
- **HF:** `harvestsu/seeed-local-voice-artifacts` → `talker-b2`.
- **Build device:** **orin-nx** (Orin NX 16GB, JetPack 6, CUDA 12.6, TRT 10.3,
  SM 87). Loadable on Orin Nano (SM 87) given matching TRT minor + plugin .so.
- **Provenance:** `edgellm-v080-migration/docs/plans/v080-phase5b-acceptance-results.md`.

## Command

```bash
WS=<tts-workspace>
build/examples/llm/llm_build \
    --onnxDir   $WS/onnx/llm \
    --engineDir $WS/engines-v080-tts-b2/talker \
    --maxBatchSize 2 --maxInputLen 4096 --maxKVCacheCapacity 4096
# → [llm_build.cpp:246:main] LLM engine built successfully.
```

> Only the Talker rebuilds to b2; the CodePredictor / Code2Wav engines are
> unchanged. The slot-pool runtime feeds `response.batchRvqCodes[k]` per
> admitted lane.

## Verify

```bash
md5sum $WS/engines-v080-tts-b2/talker/llm.engine    # integrity vs published md5
```

Functional gate (the real reproducibility judgement): solo and concurrent runs
over the b2 Talker produce **byte-identical audio** per lane (zh "今天天气真不错",
en) and each roundtrips byte-exact through production `paraformer_trt` ASR.
