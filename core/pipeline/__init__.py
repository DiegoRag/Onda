"""Orquestração de alto nível das pipelines de transmissão e recepção.

`Transmitter` compõe: AES -> framing -> semente de embaralhamento -> modulação OFDM ->
prefixar chirp -> escrever WAV ou tocar no alto-falante.

`Receiver` compõe: ler WAV ou capturar do microfone -> detecção do chirp -> permutação
de embaralhamento -> demodulação OFDM -> decrypt AES -> recuperação das amostras int16.
"""

from core.pipeline.receiver import RecoveredAudio, Receiver
from core.pipeline.transmitter import TransmissionResult, Transmitter

__all__ = [
    "RecoveredAudio",
    "Receiver",
    "TransmissionResult",
    "Transmitter",
]
