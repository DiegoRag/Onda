"""QPSK (4-QAM, Gray-coded) modulation: bits <-> complex symbols.

QPSK maps pairs of bits to one of four points on the complex unit circle:

         imag
          |
    (1,0) o   o (0,0)
          |
    ------+------ real
          |
    (1,1) o   o (0,1)
          |

Adjacent constellation points differ by exactly one bit (Gray coding), which
minimizes bit errors when a symbol is decoded into a neighbor due to noise.

Closed-form mapping:
    s = ( (1 - 2*b0) + 1j * (1 - 2*b1) ) / sqrt(2)

The 1/sqrt(2) normalization makes every symbol have unit power, so an
average-symbol-power constraint is automatically satisfied.

WATCH OUT: bits must be cast to a SIGNED integer type before `1 - 2*b`. With
`np.uint8`, `1 - 2*1 = -1` wraps to 255, silently producing garbage symbols.
This module casts to `np.int8` explicitly to make the bug impossible.
"""

from __future__ import annotations

import numpy as np


class QPSKModem:
    """Stateless QPSK modulator/demodulator.

    Implemented as a class for API discoverability and to keep the codebase
    consistent with the other "modem" objects (OFDM, Scrambler). There is no
    per-instance state — feel free to share a single QPSKModem across threads.
    """

    BITS_PER_SYMBOL: int = 2

    # Pre-computed normalization factor. 1/sqrt(2) ~ 0.7071.
    _NORM: float = 1.0 / np.sqrt(2.0)

    # ------------------------------------------------------------------
    # Forward direction: bits -> complex symbols
    # ------------------------------------------------------------------
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Map a bit-stream to QPSK symbols.

        Parameters
        ----------
        bits : np.ndarray
            uint8 or int array of shape (2N,). Each element must be 0 or 1.

        Returns
        -------
        np.ndarray
            complex128 array of shape (N,) of unit-magnitude QPSK symbols.
        """
        bits = np.asarray(bits)
        if bits.ndim != 1:
            raise ValueError(f"bits must be 1-D, got shape {bits.shape}.")
        if bits.size % self.BITS_PER_SYMBOL != 0:
            raise ValueError(
                f"bit count must be a multiple of {self.BITS_PER_SYMBOL}, "
                f"got {bits.size}."
            )

        # CRITICAL: cast to signed type. uint8 silently overflows on `1 - 2*b`.
        b_signed = bits.astype(np.int8)
        b0 = b_signed[0::2]
        b1 = b_signed[1::2]

        # Closed-form mapping: s = (1 - 2*b0) + 1j*(1 - 2*b1), normalized.
        symbols = ((1 - 2 * b0) + 1j * (1 - 2 * b1)) * self._NORM
        return symbols.astype(np.complex128)

    # ------------------------------------------------------------------
    # Reverse direction: complex symbols -> bits
    # ------------------------------------------------------------------
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Hard-decision QPSK demodulation: nearest-quadrant lookup.

        b0 = 1 iff real(s) < 0   (left half-plane)
        b1 = 1 iff imag(s) < 0   (bottom half-plane)

        Parameters
        ----------
        symbols : np.ndarray
            complex array of shape (N,). Need not have unit magnitude — the
            decision uses only the sign of real and imaginary parts.

        Returns
        -------
        np.ndarray
            uint8 array of shape (2N,) containing the recovered bits.
        """
        symbols = np.asarray(symbols)
        if symbols.ndim != 1:
            raise ValueError(f"symbols must be 1-D, got shape {symbols.shape}.")

        n = symbols.size
        bits = np.empty(2 * n, dtype=np.uint8)
        bits[0::2] = (symbols.real < 0).astype(np.uint8)
        bits[1::2] = (symbols.imag < 0).astype(np.uint8)
        return bits
