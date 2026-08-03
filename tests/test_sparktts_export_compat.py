import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sparktts_export_compat import install_unused_torchaudio_stub_if_missing


class _FakeModule:
    def __init__(self, *args, **kwargs):
        pass


class _FakeTorch:
    class nn:
        Module = _FakeModule


def test_missing_torchaudio_stub_is_decoder_only_and_fail_closed(monkeypatch):
    real_find_spec = __import__("importlib.util", fromlist=["find_spec"]).find_spec
    monkeypatch.setattr(
        "sparktts_export_compat.importlib.util.find_spec",
        lambda name: None if name == "torchaudio" else real_find_spec(name),
    )
    monkeypatch.delitem(sys.modules, "torchaudio", raising=False)
    monkeypatch.delitem(sys.modules, "torchaudio.transforms", raising=False)

    assert install_unused_torchaudio_stub_if_missing(_FakeTorch) is True

    import torchaudio.transforms as transforms

    mel = transforms.MelSpectrogram(16000, 1024)
    with pytest.raises(RuntimeError, match="must not execute the mel frontend"):
        mel.forward(None)
