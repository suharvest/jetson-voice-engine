# Hugging Face Artifacts

Target repo: `harvestsu/qwen3-edgellm-jetson-artifacts`.

This repository should contain large generated artifacts only:

- ASR thinker engines and config/tokenizer sidecars
- ASR audio encoder engines
- TTS Talker config/tokenizer/text embedding sidecars
- TTS W8A16 explicit Talker engines
- TTS CodePredictor engines and auxiliary tensors
- TTS stateful Code2Wav engines and configs
- per-device manifests/checksums

Do not store these large files in GitHub. Keep the required relative paths in `deploy/artifacts/qwen3_manifest.json` aligned with Jetson Voice profiles.

## Publication status

| Artifact set | HF status |
|---|---|
| `orin-nano-highperf-2026-05-10` | complete |
| `orin-nx-highperf-2026-05-11` | complete |
| `orin-nano-official-2026-05-10` | complete |
| `orin-nx-edgellm-v091-jp62-trt103-sm87-20260725` | 143-file staging set verified on Orin NX and WSL; original 116-file payload is on HF, while 24 sidecars plus the default-512 Code2Wav engine/config/sidecar and updated controls await explicit upload approval |

Do not mark a profile as reproducible until every path in its
`required_files` list exists in the HF repo.

The v0.9.1 full-runtime set is stored under
`orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/v091`. Unlike the older
Qwen3-only sets, it includes version-matched runtime binaries, plugin, pybind,
GDN+MTP, ASR, CustomVoice, Base, SparkTTS, MOSS, and SenseVoice artifacts.
Its `manifest.json` inventories every payload file; `SHA256SUMS` is the
portable verification list. The set is bound to SM87, JetPack 6.2 /
L4T R36.4.3, CUDA 12.6, and TensorRT 10.3.0.30.

The original publication was verified through the official
`https://huggingface.co` endpoint. The expanded 143-file staging manifest is
deliberately marked `published_to_hf=false` until the sidecars and default-512
Code2Wav files have been uploaded, the remote tree has been rechecked, and the
downloaded manifest matches byte-for-byte.

## Stage and upload

If the source directory already matches the manifest-relative layout:

```bash
python3 scripts/package_qwen3_artifacts.py \
  --set orin-nano-highperf-2026-05-10 \
  --source-root /opt/models/qwen3-edgellm \
  --out /tmp/qwen3-hf-upload

hf upload harvestsu/qwen3-edgellm-jetson-artifacts /tmp/qwen3-hf-upload . \
  --repo-type model \
  --commit-message "Upload orin-nano highperf artifacts"
```

For scattered build outputs, repeat `--map RELATIVE_PATH=/actual/source/file` for each file that is not already under `--source-root`. The packager writes `checksums/<artifact-set>.json` with file sizes and SHA-256 digests.
