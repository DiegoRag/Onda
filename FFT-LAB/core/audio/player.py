"""Speaker playback via `sounddevice`. Mono only."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

import global_configs

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Wraps `sounddevice.play` for predictable playback semantics."""

    def __init__(
        self,
        sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
    ) -> None:
        self._sample_rate: int = sample_rate

    def play(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
        blocking: bool = True,
    ) -> None:
        """Play `samples` through the default output device.

        Parameters
        ----------
        samples : np.ndarray
            int16 or float32, mono.
        sample_rate : int, optional
            Override the constructor sample rate. Useful for TX (48 kHz) vs
            voice playback (16 kHz) without juggling two players.
        blocking : bool
            Wait for the buffer to drain before returning.
        """
        rate = sample_rate if sample_rate is not None else self._sample_rate
        logger.info(
            "Playing %d samples @ %d Hz (%s)",
            samples.size, rate, samples.dtype,
        )
        sd.play(samples, samplerate=rate)
        if blocking:
            sd.wait()

    @staticmethod
    def stop() -> None:
        """Abort any in-progress playback immediately."""
        sd.stop()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
