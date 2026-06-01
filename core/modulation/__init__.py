"""Modulação de sinal: QPSK, OFDM, embaralhamento na frequência e sincronização.

Este pacote é onde o Fourier aparece de forma mais clara:

    - OFDMModem.modulate_frame   -> IFFT sintetiza um sinal real a partir dos bins
    - OFDMModem.demodulate_frame -> FFT recupera os símbolos a partir do sinal no tempo
    - Scrambler.build_permutation -> permutação dos bins da FFT controlada por chave
    - PreambleSync.generate       -> chirp linear para alinhar transmissor/receptor
"""

from core.modulation.ofdm import OFDMModem
from core.modulation.qpsk import QPSKModem
from core.modulation.scrambler import Scrambler
from core.modulation.sync import PreambleSync

__all__ = ["OFDMModem", "QPSKModem", "Scrambler", "PreambleSync"]
