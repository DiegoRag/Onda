"""WAV file I/O and int16-bytes conversions.

The WAV format is a thin wrapper around interleaved PCM samples. We use
`scipy.io.wavfile` because it is part of the existing dependency set,
handles int16/int32/float32 transparently, and is faster than the stdlib
`wave` module for large files.

This module is also responsible for the int16<->bytes interconversion that
sits at the boundary between "audio samples" (numpy arrays) and "data
bytes" (the unit on which AES operates). The conversion is bit-exact in
both directions, so a clean WAV-loopback roundtrip will produce identical
samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

import global_configs


class WavIO:
    """File-system reads and writes plus int16 <-> bytes conversions."""

    INT16_MAX: int = global_configs.INT16_MAX
    HEADROOM: float = global_configs.INT16_HEADROOM

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    @classmethod
    def write(
        cls,
        path: str | Path,
        signal: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Save a float signal as a mono int16 WAV.

        The signal is peak-normalized to HEADROOM (90%) of the int16 range to
        avoid clipping. We multiply by `INT16_MAX` (32767), not 32768, because
        +32768 would wrap to -32768 in two's-complement.
        """
        if signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {signal.shape}.")

        peak = float(np.max(np.abs(signal)))
        if peak == 0.0:
            scaled = np.zeros_like(signal, dtype=np.int16)
        else:
            scaled = (
                signal.astype(np.float64) / peak * cls.HEADROOM * cls.INT16_MAX
            ).astype(np.int16)

        wavfile.write(str(path), sample_rate, scaled)

    @classmethod
    def read(cls, path: str | Path) -> tuple[int, np.ndarray]:
        """Load a WAV and return `(sample_rate, float32_samples in [-1, 1])`.

        Stereo input is downmixed to the left channel only (consistent with
        the rest of the pipeline, which is mono throughout).
        """
        sample_rate, samples = wavfile.read(str(path))

        # Stereo -> mono (left channel).
        if samples.ndim > 1:
            samples = samples[:, 0]

        # Normalize to float32 in [-1, 1] based on the source dtype.
        if samples.dtype == np.int16:
            float_samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype == np.int32:
            float_samples = samples.astype(np.float32) / 2147483648.0
        elif samples.dtype == np.float32:
            float_samples = samples
        elif samples.dtype == np.float64:
            float_samples = samples.astype(np.float32)
        else:
            raise ValueError(f"Unsupported WAV dtype {samples.dtype}.")

        return int(sample_rate), float_samples

    # ------------------------------------------------------------------
    # int16 <-> bytes  (used by the encrypt/decrypt path)
    # ------------------------------------------------------------------
    @staticmethod
    def int16_to_bytes(samples: np.ndarray) -> bytes:
        """Serialize a 1-D int16 array to little-endian raw bytes.

        2 bytes per sample. The conversion is reversible bit-exact via
        `bytes_to_int16`.
        """
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        # np tobytes is always native byte order; force little-endian to keep
        # the WAV-compatible representation regardless of host platform.
        return samples.astype("<i2").tobytes()

    @staticmethod
    def bytes_to_int16(data: bytes) -> np.ndarray:
        """Reverse of `int16_to_bytes`. Returns a *writable* copy."""
        if len(data) % 2 != 0:
            raise ValueError(
                f"byte length must be even (2 bytes/sample), got {len(data)}."
            )
        # np.frombuffer returns a read-only view of the input buffer; copy so
        # the caller can modify the result.
        return np.frombuffer(data, dtype="<i2").astype(np.int16).copy()
