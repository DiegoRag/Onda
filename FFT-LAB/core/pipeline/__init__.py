"""High-level orchestration of the transmit and receive pipelines.

`Transmitter` composes: optional denoise -> AES -> framing -> scramble seed ->
OFDM modulation -> chirp prepend -> WAV write or speaker playback.

`Receiver` composes: WAV read or mic capture -> chirp detection -> scramble
permutation -> OFDM demodulation -> AES decrypt -> int16 sample recovery.
"""

from core.pipeline.receiver import RecoveredAudio, Receiver
from core.pipeline.transmitter import TransmissionResult, Transmitter

__all__ = [
    "RecoveredAudio",
    "Receiver",
    "TransmissionResult",
    "Transmitter",
]
