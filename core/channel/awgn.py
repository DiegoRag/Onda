"""Additive White Gaussian Noise channel — for SNR characterization only.

AWGN ("everything that isn't the signal") is the standard mathematical
abstraction for channel impairment. We use it to measure how much noise the
OFDM/QPSK link can tolerate before bit errors creep in (a BER-vs-SNR sweep).

This module is *never* in the live transmit/receive pipeline; the real
channel is acoustic and adds its own noise + multipath. AWGN gives us a
clean, reproducible benchmark.

SNR convention used here:
    SNR_dB = 10 * log10( signal_power / noise_power )
    noise_power = signal_power / 10**(SNR_dB / 10)

Signal power is measured as `mean(signal**2)` (i.e., variance for zero-mean
signals plus the DC component squared).
"""

from __future__ import annotations

import numpy as np


class AWGNChannel:
    """Adds white Gaussian noise to a signal at a target SNR."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        """Use a fixed-seed Generator for reproducible tests; otherwise None
        falls back to the default global-state-free new-style RNG."""
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    def add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Return `signal + n` where `n` is Gaussian with the target SNR.

        Parameters
        ----------
        signal : np.ndarray
            Real-valued 1-D array. Output dtype matches input dtype.
        snr_db : float
            Target signal-to-noise ratio in dB. Higher means cleaner.

        Returns
        -------
        np.ndarray
            Same shape and dtype as `signal`, with additive Gaussian noise.
        """
        signal = np.asarray(signal)
        if signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {signal.shape}.")

        # Compute noise variance to hit the target SNR.
        signal_power = float(np.mean(signal.astype(np.float64) ** 2))
        if signal_power <= 0.0:
            # Signal is silent; just return a copy (no SNR is meaningful).
            return signal.copy()

        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = self._rng.normal(
            loc=0.0,
            scale=np.sqrt(noise_power),
            size=signal.shape,
        )

        # Preserve input dtype (caller may be using float32).
        return (signal.astype(np.float64) + noise).astype(signal.dtype)

    @staticmethod
    def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
        """Estimate the SNR (in dB) given the clean and noisy versions.

        Useful for verifying that a target SNR was actually achieved.
        """
        noise = noisy.astype(np.float64) - clean.astype(np.float64)
        signal_power = float(np.mean(clean.astype(np.float64) ** 2))
        noise_power = float(np.mean(noise ** 2))
        if noise_power <= 0:
            return float("inf")
        return 10.0 * np.log10(signal_power / noise_power)
