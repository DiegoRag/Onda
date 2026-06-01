"""Spectral subtraction denoising — single-channel, magnitude-domain.

Algorithm (Boll, 1979) in three steps:

    1. STFT of the noisy input.
    2. Estimate the noise magnitude spectrum |N(f)| from the first
       DENOISE_NOISE_FLOOR_FRAMES frames (assumed to be silence).
    3. For each subsequent frame Y(f), subtract a scaled copy of the noise:
            |S(f)| = max( |Y(f)| - alpha * |N(f)|,  beta * |Y(f)| )
            S(f)  = |S(f)| * exp(j * phase(Y(f)))
       The `beta` floor preserves a tiny residual to avoid "musical noise"
       (random isolated frequency bins surviving subtraction).
    4. ISTFT to reconstruct the time-domain signal.

Where Fourier appears
---------------------
The STFT is a sequence of FFTs on short overlapping windows. The denoiser is
the cleanest example of *using Fourier as a filter*: we move into the
frequency domain, do an operation that would be impossible in the time
domain (subtract specific frequency components), and come back. Mention this
in the report.

Assumption / limitation
-----------------------
The first ~80 ms of the input MUST be noise (silence). If the user starts
speaking immediately at recording onset, the noise floor estimate contains
voice energy and the algorithm subtracts the wrong thing. Recommend a brief
pause before speaking.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

import global_configs


class SpectralDenoiser:
    """Subtracts a noise floor estimated from the first frames of the signal."""

    FRAME_SIZE: int = global_configs.DENOISE_FRAME_SIZE
    HOP_SIZE: int = global_configs.DENOISE_HOP_SIZE
    NOISE_FLOOR_FRAMES: int = global_configs.DENOISE_NOISE_FLOOR_FRAMES
    ALPHA: float = global_configs.DENOISE_OVERSUBTRACTION_FACTOR
    BETA: float = global_configs.DENOISE_SPECTRAL_FLOOR

    # scipy's stft uses `noverlap` instead of `hop`. They are related by:
    #     noverlap = nperseg - hop
    _NOVERLAP: int = FRAME_SIZE - HOP_SIZE  # 1024 - 256 = 768  (75% overlap)

    def __init__(self) -> None:
        # Hann window pre-computed once.
        self._window: np.ndarray = sps.get_window("hann", self.FRAME_SIZE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def denoise(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply spectral subtraction. Returns same-shape, same-dtype array.

        Parameters
        ----------
        signal : np.ndarray
            1-D array of int16 or float32 audio samples. The first
            ~`NOISE_FLOOR_FRAMES * HOP_SIZE` samples should be silence.
        sample_rate : int
            Used only by `scipy.signal.stft` for bookkeeping; no resampling
            is performed.

        Returns
        -------
        np.ndarray
            Denoised signal, same shape and dtype as the input.
        """
        if signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {signal.shape}.")

        original_dtype = signal.dtype
        float_signal = signal.astype(np.float64)

        # --- STFT ---
        # boundary=None, padded=False -> output time bins exactly match input.
        # We control the windowing/overlap to match the ISTFT inverse below.
        freqs, times, Zxx = sps.stft(
            float_signal,
            fs=sample_rate,
            window=self._window,
            nperseg=self.FRAME_SIZE,
            noverlap=self._NOVERLAP,
            boundary="zeros",
            padded=True,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # --- Noise floor: average of the first N frames ---
        n_floor = min(self.NOISE_FLOOR_FRAMES, magnitude.shape[1])
        if n_floor <= 0:
            # No frames available -> nothing meaningful to subtract.
            return signal.copy()
        noise_mag = magnitude[:, :n_floor].mean(axis=1, keepdims=True)

        # --- Spectral subtraction with floor ---
        cleaned_mag = np.maximum(
            magnitude - self.ALPHA * noise_mag,
            self.BETA * magnitude,
        )

        # --- Reconstruct complex spectrum and ISTFT ---
        cleaned_complex = cleaned_mag * np.exp(1j * phase)
        _, time_signal = sps.istft(
            cleaned_complex,
            fs=sample_rate,
            window=self._window,
            nperseg=self.FRAME_SIZE,
            noverlap=self._NOVERLAP,
            boundary=True,
        )

        # scipy.signal.istft may return a slightly longer array due to
        # overlap-add. Crop / pad to match the original length.
        if time_signal.size >= signal.size:
            time_signal = time_signal[: signal.size]
        else:
            pad = np.zeros(signal.size - time_signal.size, dtype=time_signal.dtype)
            time_signal = np.concatenate([time_signal, pad])

        return time_signal.astype(original_dtype)

    # ------------------------------------------------------------------
    # Diagnostic helper
    # ------------------------------------------------------------------
    def estimate_noise_floor(self, signal: np.ndarray) -> np.ndarray:
        """Return the noise magnitude spectrum estimated from the input head.

        Output length is `FRAME_SIZE // 2 + 1` (one-sided spectrum). Useful
        for plotting the noise floor in diagnostics.
        """
        if signal.size < self.NOISE_FLOOR_FRAMES * self.HOP_SIZE:
            raise ValueError(
                "signal too short to estimate noise floor "
                f"(need >= {self.NOISE_FLOOR_FRAMES * self.HOP_SIZE} samples)."
            )
        _, _, Zxx = sps.stft(
            signal.astype(np.float64),
            window=self._window,
            nperseg=self.FRAME_SIZE,
            noverlap=self._NOVERLAP,
            boundary="zeros",
            padded=True,
        )
        n_floor = min(self.NOISE_FLOOR_FRAMES, Zxx.shape[1])
        return np.abs(Zxx[:, :n_floor]).mean(axis=1)
