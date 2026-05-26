"""Synchronization preamble: a linear chirp + cross-correlation detector.

Over-the-air transmission means the sender and receiver do NOT share a clock.
The receiver must figure out *when* the OFDM data begins inside the captured
audio. We solve this with a known waveform — a linear chirp — placed
immediately before the data. The receiver cross-correlates the captured
signal against a local copy of the chirp; the correlation peak marks the end
of the chirp (i.e., the start of the data).

Why a chirp?
------------
A chirp sweeps frequency over time. Its autocorrelation has a sharp narrow
peak (close to a delta), giving precise timing. A pure sinusoid would also
correlate well, but its autocorrelation is a sinusoid — many peaks, no
unambiguous timing. A noise-like sequence works too (PN sequences are used
in CDMA) but a chirp doubles as a built-in channel-frequency-response probe
since it visits every frequency in the band.

Ramps at the edges
------------------
The chirp is multiplied by a half-Hann window over the first and last
PREAMBLE_RAMP_S seconds. Without ramps, the abrupt start of the sinusoid
acts as a step function and splatters energy outside the 6-10 kHz band,
which can leak audible clicks and waste TX power. The ramp tapers the
amplitude smoothly to zero at the edges.

Detection rule
--------------
We use normalized cross-correlation (peak amplitude bounded to ~1.0 when
the signal exactly matches the preamble). The peak is accepted if it
exceeds `threshold_factor * mean(|correlation|)`. Default factor = 10×.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

import global_configs


class PreambleSync:
    """Generates and detects the chirp preamble used to align over-air frames."""

    FS: int = global_configs.FS
    DURATION_S: float = global_configs.PREAMBLE_DURATION_S
    F_START: int = global_configs.PREAMBLE_F_START
    F_END: int = global_configs.PREAMBLE_F_END
    RAMP_S: float = global_configs.PREAMBLE_RAMP_S

    def __init__(self) -> None:
        # The preamble is deterministic; build once and cache.
        self._preamble: np.ndarray = self._build_preamble()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_preamble(self) -> np.ndarray:
        """Synthesize the linear chirp with half-Hann ramps at both ends."""
        n_samples = int(self.DURATION_S * self.FS)
        t = np.arange(n_samples) / self.FS

        # Instantaneous frequency: f(t) = F_START + (F_END - F_START) * t / T
        # Integrated phase: phi(t) = 2*pi * (F_START*t + (F_END - F_START)/(2T) * t^2)
        sweep_rate = (self.F_END - self.F_START) / self.DURATION_S
        phase = 2.0 * np.pi * (self.F_START * t + 0.5 * sweep_rate * t * t)
        chirp = np.cos(phase).astype(np.float32)

        # Half-Hann ramps. A Hann window is 0.5 * (1 - cos(pi * t / T_ramp))
        # for t in [0, T_ramp], rising from 0 to 1. We use the same shape
        # mirrored for the trailing ramp.
        ramp_n = int(self.RAMP_S * self.FS)
        if ramp_n > 0 and 2 * ramp_n <= n_samples:
            ramp_idx = np.arange(ramp_n)
            ramp = 0.5 * (1.0 - np.cos(np.pi * ramp_idx / ramp_n))
            chirp[:ramp_n] *= ramp.astype(np.float32)
            chirp[-ramp_n:] *= ramp[::-1].astype(np.float32)

        return chirp

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def preamble(self) -> np.ndarray:
        """Return a copy of the preamble waveform (float32)."""
        return self._preamble.copy()

    @property
    def length(self) -> int:
        """Preamble length in samples."""
        return self._preamble.size

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(
        self,
        captured_signal: np.ndarray,
        threshold_factor: float = 10.0,
    ) -> int:
        """Locate the end of the preamble inside `captured_signal`.

        Parameters
        ----------
        captured_signal : np.ndarray
            1-D float array (the microphone capture or a loaded WAV).
        threshold_factor : float
            How much the correlation peak must exceed the mean to be accepted.

        Returns
        -------
        int
            Index of the first sample AFTER the preamble, or -1 if no peak
            exceeds the acceptance threshold.
        """
        if captured_signal.ndim != 1:
            raise ValueError(
                f"captured_signal must be 1-D, got shape {captured_signal.shape}."
            )
        if captured_signal.size < self._preamble.size:
            return -1

        # Cross-correlate using scipy's FFT-based implementation. mode='valid'
        # gives a peak index k that means: preamble best aligns at
        # captured_signal[k : k + preamble_len].
        # We use 'fft' method to keep this fast even for second-long captures.
        corr = sps.correlate(
            captured_signal.astype(np.float64),
            self._preamble.astype(np.float64),
            mode="valid",
            method="fft",
        )
        abs_corr = np.abs(corr)
        mean_corr = abs_corr.mean()
        peak_idx = int(np.argmax(abs_corr))
        peak_val = abs_corr[peak_idx]

        if mean_corr <= 0 or peak_val < threshold_factor * mean_corr:
            return -1

        # Convert "preamble start index" to "first sample AFTER preamble".
        return peak_idx + self._preamble.size
