from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE_BUILDER = ROOT / "engine-overlay/build-engines-for-device.sh"
SPARK_FETCH = ROOT / "engine-overlay/drivers/fetch-sparktts-v091-inputs.sh"


def test_engine_builder_requires_hugging_face_mirror():
    text = ENGINE_BUILDER.read_text()

    assert 'HF="${HF_ENDPOINT:-}"' in text
    assert 'HF_ENDPOINT=https://hf-mirror.com' in text
    assert 'HF_ENDPOINT:-https://huggingface.co' not in text
    assert 'HF_ENDPOINT="${HF}" hf download' in text


def test_spark_shared_engines_both_receive_release_sidecars():
    text = ENGINE_BUILDER.read_text()

    assert "--minShapes=semantic_tokens:1x1,d_vector:1x1024" in text
    assert "--maxShapes=semantic_tokens:1x600,d_vector:1x1024" in text
    assert "--minShapes=semantic_tokens:1x50" not in text
    assert '_meta "${eng}/bicodec_decoder_dynT.fp16.engine"' in text
    assert '_meta "${eng}/sparktts_speaker_decoder.fp32.engine"' in text


def test_spark_fetcher_pins_source_and_uses_mirror_with_resume_cache():
    text = SPARK_FETCH.read_text()

    assert "2f1ea9082400547242641f5271b6f941c9f439d1" in text
    assert "642071559bfc6346c2359d19dcb6be3f9dd8a05d" in text
    assert 'HF_ENDPOINT:-}" != "https://hf-mirror.com"' in text
    assert "bash -c" in text
    assert "hf download" in text
    assert '--revision "${SPARK_MODEL_REVISION}"' in text
    assert '--local-dir "${model_dir}"' in text
    assert "BiCodec/config.yaml" in text
    assert "BiCodec/model.safetensors" in text
    assert "rm -" not in text


def test_spark_fetcher_materializes_a_new_no_checkout_clone_even_at_pinned_head():
    text = SPARK_FETCH.read_text()

    fetch_guard = 'if [ "${actual_source}" != "${SPARK_SOURCE_SHA}" ]; then'
    checkout_guard = 'if [ "${source_created}" = 1 ]; then'
    checkout = 'git -C "${source_dir}" checkout --detach "${SPARK_SOURCE_SHA}"'

    assert text.index(fetch_guard) < text.index(checkout_guard) < text.index(checkout)
    assert text.index(checkout) < text.index('actual_source="$(git -C "${source_dir}" rev-parse HEAD)"')
