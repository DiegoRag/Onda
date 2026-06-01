"""Reprodução pelo alto-falante via `sounddevice`. Apenas mono."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

import global_configs

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Embrulha `sounddevice.play` para uma semântica de reprodução previsível."""

    def __init__(
        self,
        sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
    ) -> None:
        # Taxa padrão usada quando o chamador não sobrescrever no play().
        self._sample_rate: int = sample_rate

    def play(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
        blocking: bool = True,
    ) -> None:
        """Toca `samples` no dispositivo de saída padrão.

        Parameters
        ----------
        samples : np.ndarray
            int16 ou float32, mono.
        sample_rate : int, opcional
            Sobrescreve a taxa do construtor. Útil para TX (48 kHz) vs reprodução de
            voz (16 kHz) sem precisar de dois players.
        blocking : bool
            Espera o buffer esvaziar antes de retornar.
        """
        # Usa a taxa passada na chamada; se None, cai na taxa do construtor.
        rate = sample_rate if sample_rate is not None else self._sample_rate
        logger.info(
            "Playing %d samples @ %d Hz (%s)",
            samples.size, rate, samples.dtype,
        )
        # Dispara a reprodução; se blocking, espera terminar.
        sd.play(samples, samplerate=rate)
        if blocking:
            sd.wait()

    @staticmethod
    def stop() -> None:
        """Aborta qualquer reprodução em andamento imediatamente."""
        sd.stop()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
