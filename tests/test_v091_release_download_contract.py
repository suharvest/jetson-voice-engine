from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE_BUILDER = ROOT / "engine-overlay/build-engines-for-device.sh"


def test_engine_builder_requires_hugging_face_mirror():
    text = ENGINE_BUILDER.read_text()

    assert 'HF="${HF_ENDPOINT:-}"' in text
    assert 'HF_ENDPOINT=https://hf-mirror.com' in text
    assert 'HF_ENDPOINT:-https://huggingface.co' not in text
    assert 'HF_ENDPOINT="${HF}" hf download' in text


def test_spark_shared_engines_both_receive_release_sidecars():
    text = ENGINE_BUILDER.read_text()

    assert '_meta "${eng}/bicodec_decoder_dynT.fp16.engine"' in text
    assert '_meta "${eng}/sparktts_speaker_decoder.fp32.engine"' in text
