"""Signal modulation: QPSK, OFDM, frequency-domain scrambling, and preamble sync.

This package is where Fourier shows up most clearly:

    - OFDMModem.modulate_frame   -> IFFT synthesizes a real signal from bins
    - OFDMModem.demodulate_frame -> FFT recovers symbols from the time signal
    - Scrambler.build_permutation -> key-controlled permutation of FFT bins
    - PreambleSync.generate       -> linear chirp for sender/receiver alignment
"""

from core.modulation.ofdm import OFDMModem
from core.modulation.qpsk import QPSKModem
from core.modulation.scrambler import Scrambler
from core.modulation.sync import PreambleSync

__all__ = ["OFDMModem", "QPSKModem", "Scrambler", "PreambleSync"]
