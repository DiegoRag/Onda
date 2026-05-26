"""OFDM modulation/demodulation — the heart of the Fourier story.

OFDM (Orthogonal Frequency-Division Multiplexing) sends data by encoding it
into the AMPLITUDES of many narrowband subcarriers, then summing them. The
sum is computed by an IFFT: a single inverse FFT yields the time-domain
signal that contains all subcarriers at once.

Where Fourier appears here
--------------------------
    Transmit:  data symbols -> place into FFT bins -> IFFT -> real signal
               (Fourier as a SIGNAL GENERATOR)

    Receive:   real signal -> FFT -> read FFT bins -> data symbols
               (Fourier as a SIGNAL ANALYZER)

This is the cleanest illustration of Fourier as an *engineering* tool, not
just a mathematical analysis.

Frame layout
------------
Each OFDM frame is `N_FFT + N_CP` samples long (256 + 64 = 320). The cyclic
prefix is the LAST N_CP samples of the IFFT output, prepended at the
beginning. Two reasons:
    1. It absorbs timing jitter / multipath echoes (delayed copies of the
       signal interfere within the CP region, not with the data region).
    2. It converts linear convolution with the channel into circular
       convolution, which preserves orthogonality of the FFT bins.

Conjugate symmetry
------------------
We want a REAL time-domain signal (we are going to feed it into a soundcard).
A real signal's spectrum is conjugate-symmetric: X[N-k] = conj(X[k]). So we
explicitly mirror every bin we set into its negative-frequency twin before
calling IFFT. The IFFT output then has zero imaginary part (modulo
floating-point noise ~1e-15).

Channel estimation via pilot
----------------------------
Real channels (speakers/mics/air/walls) introduce an unknown complex gain
H(f). We send a KNOWN symbol on PILOT_BIN (value 1+0j); the receiver
measures the received value at PILOT_BIN and computes H = received/expected.
Dividing the data bins by H equalizes the channel. This is the simplest
possible channel estimator — works as long as H is approximately flat across
the 22-bin band (which is the case for our 4 kHz-wide band over short
distances).
"""

from __future__ import annotations

import numpy as np

import global_configs
from core.modulation.qpsk import QPSKModem


class OFDMModem:
    """OFDM transmitter and receiver tied to the spec's numerical parameters.

    Parameters are pulled from `global_configs` at construction time. The
    modem is stateless across frames — calling `modulate_frame` repeatedly
    with the same arguments always produces the same output.
    """

    # All shape-defining constants snapshot from config (see global_configs.py).
    N_FFT: int = global_configs.N_FFT
    N_CP: int = global_configs.N_CP
    FRAME_LEN: int = N_FFT + N_CP                       # 320

    PILOT_BIN: int = global_configs.PILOT_BIN
    PILOT_VALUE: complex = global_configs.PILOT_VALUE
    DATA_BINS: np.ndarray = np.asarray(global_configs.DATA_BINS, dtype=np.intp)
    N_DATA_BINS: int = len(global_configs.DATA_BINS)    # 21

    BITS_PER_OFDM_FRAME: int = global_configs.BITS_PER_OFDM_FRAME  # 42
    BYTES_PER_OFDM_FRAME: int = BITS_PER_OFDM_FRAME // 8           # not used, see _pad_bits

    def __init__(self, qpsk: QPSKModem | None = None) -> None:
        """Optionally inject a custom QPSKModem; default is a fresh instance."""
        self._qpsk: QPSKModem = qpsk if qpsk is not None else QPSKModem()

    # ==================================================================
    # SINGLE FRAME — modulation
    # ==================================================================
    def modulate_frame(
        self,
        data_symbols: np.ndarray,
        permutation: np.ndarray,
    ) -> np.ndarray:
        """Build one OFDM time-domain frame from 21 QPSK symbols.

        Parameters
        ----------
        data_symbols : np.ndarray
            complex array of shape (21,). Already QPSK-mapped.
        permutation : np.ndarray
            int array of shape (21,). Maps logical-index -> physical-bin index
            within DATA_BINS. Identity permutation `np.arange(21)` disables
            scrambling.

        Returns
        -------
        np.ndarray
            float32 array of length 320 (= N_FFT + N_CP).
        """
        if data_symbols.shape != (self.N_DATA_BINS,):
            raise ValueError(
                f"data_symbols must have shape ({self.N_DATA_BINS},), "
                f"got {data_symbols.shape}."
            )
        if permutation.shape != (self.N_DATA_BINS,):
            raise ValueError(
                f"permutation must have shape ({self.N_DATA_BINS},), "
                f"got {permutation.shape}."
            )

        # 1) Empty spectrum.
        spectrum = np.zeros(self.N_FFT, dtype=np.complex128)

        # 2) Place pilot.
        spectrum[self.PILOT_BIN] = self.PILOT_VALUE

        # 3) Place scrambled data symbols.
        #    output[permutation[i]] = data_symbols[i]  -->  use fancy indexing
        scrambled = np.empty_like(data_symbols)
        scrambled[permutation] = data_symbols
        spectrum[self.DATA_BINS] = scrambled

        # 4) Conjugate symmetry: mirror positive freqs into negative-freq bins
        #    so the IFFT output is purely real. For k in 1..N/2-1:
        #    spectrum[N-k] = conj(spectrum[k]).
        positive = np.arange(1, self.N_FFT // 2)
        spectrum[self.N_FFT - positive] = np.conj(spectrum[positive])

        # 5) IFFT: synthesize the time-domain signal.
        time_signal = np.fft.ifft(spectrum)

        # 6) Take real part (imag should be ~1e-15 due to conjugate symmetry;
        #    we strip the noise rather than carrying a complex array around).
        real_signal = time_signal.real

        # 7) Cyclic prefix: prepend the last N_CP samples to the front.
        cp = real_signal[-self.N_CP:]
        frame = np.concatenate([cp, real_signal]).astype(np.float32)

        assert frame.shape == (self.FRAME_LEN,)
        return frame

    # ==================================================================
    # SINGLE FRAME — demodulation
    # ==================================================================
    def demodulate_frame(
        self,
        received: np.ndarray,
        permutation: np.ndarray,
    ) -> tuple[np.ndarray, complex]:
        """Recover 21 QPSK symbols from one 320-sample frame.

        Parameters
        ----------
        received : np.ndarray
            float array of shape (320,) — one CP + IFFT block.
        permutation : np.ndarray
            int array of shape (21,) — the same permutation used at TX.

        Returns
        -------
        symbols : np.ndarray
            complex array of shape (21,) — equalized, unscrambled symbols.
        h_estimate : complex
            The channel response measured on the pilot. Useful for diagnostics
            (magnitude indicates link strength, phase indicates carrier offset).
        """
        if received.shape != (self.FRAME_LEN,):
            raise ValueError(
                f"received must have shape ({self.FRAME_LEN},), "
                f"got {received.shape}."
            )

        # 1) Drop the cyclic prefix.
        useful = received[self.N_CP:]

        # 2) FFT to recover the spectrum.
        spectrum = np.fft.fft(useful)

        # 3) Channel estimate from the pilot. H carries any combined gain
        #    introduced by speaker, air, and microphone. As long as H is
        #    consistent across nearby bins, dividing equalizes the channel.
        h_estimate = spectrum[self.PILOT_BIN] / self.PILOT_VALUE

        # Guard against the degenerate case of a zero pilot (e.g., feeding a
        # silent signal). Without this we'd produce nan/inf symbols.
        if h_estimate == 0:
            equalized = spectrum[self.DATA_BINS]
        else:
            equalized = spectrum[self.DATA_BINS] / h_estimate

        # 4) Unscramble: symbols[i] = equalized[permutation[i]].
        unscrambled = equalized[permutation]

        return unscrambled, complex(h_estimate)

    # ==================================================================
    # MANY FRAMES — bytes <-> signal (convenience)
    # ==================================================================
    def bytes_to_signal(self, data: bytes, permutation: np.ndarray) -> np.ndarray:
        """Encode a byte stream into a long OFDM signal.

        Pads the bit stream with zeros up to a multiple of BITS_PER_OFDM_FRAME,
        QPSK-modulates, and emits one OFDM frame per chunk. The same
        permutation is applied to every frame.
        """
        # Bytes -> bits (MSB first, matching np.packbits convention).
        bit_array = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

        # Pad with zeros so every frame is full.
        pad = (-len(bit_array)) % self.BITS_PER_OFDM_FRAME
        if pad:
            bit_array = np.concatenate([bit_array, np.zeros(pad, dtype=np.uint8)])

        n_frames = len(bit_array) // self.BITS_PER_OFDM_FRAME

        # Pre-allocate the full output — avoids the n-square cost of repeated
        # np.concatenate inside a loop.
        signal = np.empty(n_frames * self.FRAME_LEN, dtype=np.float32)

        for i in range(n_frames):
            chunk = bit_array[
                i * self.BITS_PER_OFDM_FRAME : (i + 1) * self.BITS_PER_OFDM_FRAME
            ]
            symbols = self._qpsk.modulate(chunk)
            frame = self.modulate_frame(symbols, permutation)
            signal[i * self.FRAME_LEN : (i + 1) * self.FRAME_LEN] = frame

        return signal

    def signal_to_bytes(
        self,
        signal: np.ndarray,
        n_bytes: int,
        permutation: np.ndarray,
    ) -> bytes:
        """Decode an OFDM signal back into the original byte stream.

        Trailing padding (from `bytes_to_signal`) is dropped by truncating to
        exactly `n_bytes`.
        """
        if signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {signal.shape}.")

        n_frames = signal.size // self.FRAME_LEN
        if n_frames == 0:
            return b""

        bit_buffer = np.empty(
            n_frames * self.BITS_PER_OFDM_FRAME, dtype=np.uint8
        )

        for i in range(n_frames):
            frame = signal[i * self.FRAME_LEN : (i + 1) * self.FRAME_LEN]
            symbols, _ = self.demodulate_frame(frame, permutation)
            bits = self._qpsk.demodulate(symbols)
            bit_buffer[
                i * self.BITS_PER_OFDM_FRAME : (i + 1) * self.BITS_PER_OFDM_FRAME
            ] = bits

        # Bits -> bytes. np.packbits pads with zeros at the end automatically
        # if the bit count is not a multiple of 8; we then truncate to the
        # caller's exact request.
        packed = np.packbits(bit_buffer).tobytes()
        return packed[:n_bytes]

    # ==================================================================
    # Diagnostics
    # ==================================================================
    @classmethod
    def frame_duration_s(cls, fs: int = global_configs.FS) -> float:
        """Wall-clock seconds occupied by one OFDM frame at sample rate `fs`."""
        return cls.FRAME_LEN / fs
