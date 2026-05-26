"""Value object that holds the three signals of the FFT Lab demo.

Separating the STATE (what we are showing) from the VIEW (how we are showing
it) is a classic OOP pattern. The view reads from this state and asks the
canvas to redraw; the state knows nothing about widgets or matplotlib.

A single mutable dataclass is enough here — the FFT Lab is single-threaded
from the perspective of these fields (recordings happen on a worker but
write back to the state via `self.after(0, ...)`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignalState:
    """Holds the three signals manipulated in the FFT Lab tab.

    Fields
    ------
    time_signal : np.ndarray | None
        Source audio in the time domain (float32, mono, range [-1, 1]).
    sample_rate : int | None
        Sample rate of `time_signal`. Required to convert FFT bin index to Hz.
    freq_spectrum : np.ndarray | None
        Complex spectrum produced by `np.fft.fft(time_signal)`. Length equals
        `len(time_signal)`. Plot uses only the first half (positive freqs).
    recovered_signal : np.ndarray | None
        Time-domain reconstruction produced by `np.fft.ifft(freq_spectrum).real`.
        Should match `time_signal` to within floating-point precision.
    """

    time_signal: np.ndarray | None = None
    sample_rate: int | None = None
    freq_spectrum: np.ndarray | None = None
    recovered_signal: np.ndarray | None = None

    def clear(self) -> None:
        """Reset to empty state — used when loading a fresh signal."""
        self.time_signal = None
        self.sample_rate = None
        self.freq_spectrum = None
        self.recovered_signal = None

    @property
    def has_time(self) -> bool:
        return self.time_signal is not None and self.sample_rate is not None

    @property
    def has_freq(self) -> bool:
        return self.freq_spectrum is not None

    @property
    def has_recovered(self) -> bool:
        return self.recovered_signal is not None
