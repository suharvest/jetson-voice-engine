"""Narrow compatibility helpers for the SparkTTS decoder-only exporters."""

import importlib.util
import sys
import types


def install_unused_torchaudio_stub_if_missing(torch) -> bool:
    """Allow decoder-only model loading when torchaudio is not installed.

    BiCodec constructs a MelSpectrogram during initialization, although neither
    decoder exporter calls the mel frontend. The stub is deliberately unusable:
    an accidental encoder/frontend call fails instead of producing fake values.
    """

    if importlib.util.find_spec("torchaudio") is not None:
        return False

    class UnusedMelSpectrogram(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError(
                "torchaudio is unavailable; the decoder-only SparkTTS export "
                "must not execute the mel frontend"
            )

    torchaudio = types.ModuleType("torchaudio")
    transforms = types.ModuleType("torchaudio.transforms")
    transforms.MelSpectrogram = UnusedMelSpectrogram
    torchaudio.transforms = transforms
    sys.modules["torchaudio"] = torchaudio
    sys.modules["torchaudio.transforms"] = transforms
    return True
