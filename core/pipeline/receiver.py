"""Pipeline de recepção de alto nível: arquivo WAV ou microfone -> amostras de voz.

Inverso do `Transmitter`:

    sinal float32 capturado / carregado @ 48 kHz
        |
        | PreambleSync.detect  -> índice de início do OFDM
        v
    região OFDM do sinal
        |
        | demodula primeiro HEADER_SIZE_BYTES bytes em frames
        v
    FrameHeader (nonce, sample_rate, num_samples, ciphertext_length)
        |
        | demodula os ciphertext_length bytes restantes
        v
    bytes de payload (cabeçalho + ciphertext)
        |
        | Framer.extract_ciphertext
        v
    bytes de ciphertext
        |
        | AESCipher.decrypt com o nonce do cabeçalho
        v
    bytes de plaintext (int16 cru, little-endian)
        |
        | WavIO.bytes_to_int16 -> trunca para num_samples
        v
    amostras de voz recuperadas (int16)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import global_configs
from core.audio.player import AudioPlayer
from core.audio.recorder import AudioRecorder
from core.audio.wav_io import WavIO
from core.crypto.cipher import AESCipher
from core.crypto.framer import FrameHeader, Framer
from core.modulation.ofdm import OFDMModem
from core.modulation.scrambler import Scrambler
from core.modulation.sync import PreambleSync

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RecoveredAudio:
    """Resultado de uma operação de recepção."""

    samples: np.ndarray            # int16, comprimento == header.num_samples
    sample_rate: int               # taxa da voz de origem (vinda do cabeçalho)
    header: FrameHeader            # cabeçalho completo para diagnóstico
    h_estimate_magnitude: float    # |H| no primeiro frame — força do enlace
    preamble_index: int            # deslocamento (amostra) onde o OFDM começa


class ReceptionError(Exception):
    """Levantada quando o preâmbulo não é localizado ou o frame está malformado."""


class Receiver:
    """Decodifica arquivos WAV ou capturas ao vivo do microfone de volta em voz."""

    FS: int = global_configs.FS

    def __init__(
        self,
        password: str,
        *,
        cipher: AESCipher | None = None,
        framer: Framer | None = None,
        ofdm: OFDMModem | None = None,
        scrambler: Scrambler | None = None,
        preamble: PreambleSync | None = None,
        recorder: AudioRecorder | None = None,
        player: AudioPlayer | None = None,
    ) -> None:
        """Constrói um receptor amarrado a `password`.

        O gravador, por padrão, é um `AudioRecorder` novo configurado para a taxa do
        OFDM (48 kHz, mono, float32), NÃO para a taxa da voz.
        """
        # Componentes injetáveis; padrões derivados da senha quando não fornecidos.
        self._cipher: AESCipher = cipher or AESCipher.from_password(password)
        self._framer: Framer = framer or Framer()
        self._ofdm: OFDMModem = ofdm or OFDMModem()
        self._scrambler: Scrambler = scrambler or Scrambler()
        self._preamble: PreambleSync = preamble or PreambleSync()
        # Gravador na taxa do OFDM (48 kHz), pois é o que chega pelo ar.
        self._recorder: AudioRecorder = recorder or AudioRecorder(
            sample_rate=self.FS,
            channels=1,
            dtype="float32",
        )
        self._player: AudioPlayer = player or AudioPlayer()

        # Mesma permutação do transmissor, derivada da mesma senha.
        self._permutation: np.ndarray = self._scrambler.permutation_for_password(
            password
        )

    # ==================================================================
    # Fontes: arquivo / microfone
    # ==================================================================
    def from_wav(self, path: str | Path) -> RecoveredAudio:
        """Decodifica um WAV de transmissão salvo antes (fluxo de loopback)."""
        sample_rate, signal = WavIO.read(path)
        # O WAV precisa estar na taxa do OFDM; senão a demodulação não casa.
        if sample_rate != self.FS:
            raise ReceptionError(
                f"expected transmission sample rate {self.FS} Hz, "
                f"WAV reports {sample_rate} Hz."
            )
        return self.decode_signal(signal)

    def from_microphone(self, duration_s: float) -> RecoveredAudio:
        """Captura `duration_s` segundos do microfone e decodifica.

        Duração recomendada: alguns segundos além do comprimento de TX esperado para
        absorver o atraso de partida entre transmissor e receptor.
        """
        # Grava do microfone e garante float32 antes de decodificar.
        signal = self._recorder.record(duration_s)
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)
        return self.decode_signal(signal)

    # ==================================================================
    # Decodificador central
    # ==================================================================
    def decode_signal(self, signal: np.ndarray) -> RecoveredAudio:
        """Roda detecção de preâmbulo + demod OFDM + decrypt AES num sinal @ 48 kHz."""
        # Trabalha com um sinal mono 1-D.
        if signal.ndim != 1:
            raise ReceptionError(
                f"signal must be 1-D, got shape {signal.shape}."
            )

        # ---- Estágio 1: detecção do preâmbulo ----
        # Acha onde o chirp termina = onde os dados OFDM começam.
        ofdm_start = self._preamble.detect(signal.astype(np.float32))
        if ofdm_start < 0:
            raise ReceptionError("preamble not detected in captured signal.")
        logger.info("Preamble located; OFDM payload begins at sample %d.", ofdm_start)

        # Tudo a partir do início do OFDM.
        payload_signal = signal[ofdm_start:]

        # ---- Estágio 2: demodula frames suficientes para ler o cabeçalho ----
        # Primeiro só o necessário para os 28 bytes do cabeçalho.
        header_frames_needed = self._frames_for_bytes(
            self._framer.HEADER_SIZE_BYTES
        )
        header_signal = payload_signal[
            : header_frames_needed * self._ofdm.FRAME_LEN
        ]
        # Sem frames suficientes -> sinal curto demais.
        if header_signal.size < header_frames_needed * self._ofdm.FRAME_LEN:
            raise ReceptionError(
                "captured signal too short to contain a frame header."
            )
        # Demodula -> bytes -> interpreta o cabeçalho.
        header_bytes = self._ofdm.signal_to_bytes(
            header_signal,
            self._framer.HEADER_SIZE_BYTES,
            self._permutation,
        )
        header = self._framer.parse_header(header_bytes)
        logger.info(
            "Header: sample_rate=%d, num_samples=%d, ciphertext_length=%d",
            header.sample_rate, header.num_samples, header.ciphertext_length,
        )

        # ---- Estágio 3: demodula o payload completo ----
        # Agora sabemos o tamanho total (cabeçalho + ciphertext) pelo cabeçalho.
        total_bytes_needed = self._framer.HEADER_SIZE_BYTES + header.ciphertext_length
        total_frames_needed = self._frames_for_bytes(total_bytes_needed)
        full_signal = payload_signal[: total_frames_needed * self._ofdm.FRAME_LEN]
        if full_signal.size < total_frames_needed * self._ofdm.FRAME_LEN:
            raise ReceptionError(
                "captured signal too short to contain the announced ciphertext."
            )
        payload_bytes = self._ofdm.signal_to_bytes(
            full_signal, total_bytes_needed, self._permutation
        )

        # ---- Estágio 4: decrypt AES-256-CTR ----
        # Recorta o ciphertext e decripta com o nonce que veio no cabeçalho.
        ciphertext = self._framer.extract_ciphertext(
            payload_bytes, header.ciphertext_length
        )
        plaintext = self._cipher.decrypt(header.nonce, ciphertext)

        # ---- Estágio 5: bytes -> int16, trunca para a contagem original ----
        samples = WavIO.bytes_to_int16(plaintext)
        samples = samples[: header.num_samples]

        # Diagnóstico: mede |H| no primeiro frame para dar feedback na UI.
        first_frame = full_signal[: self._ofdm.FRAME_LEN]
        _, h_first = self._ofdm.demodulate_frame(first_frame, self._permutation)

        return RecoveredAudio(
            samples=samples,
            sample_rate=header.sample_rate,
            header=header,
            h_estimate_magnitude=float(abs(h_first)),
            preamble_index=ofdm_start,
        )

    # ==================================================================
    # Saídas: arquivo / alto-falante
    # ==================================================================
    def save_wav(self, recovered: RecoveredAudio, path: str | Path) -> None:
        """Salva as amostras recuperadas como WAV mono int16 na taxa de origem."""
        WavIO.write(path, recovered.samples.astype(np.float32), recovered.sample_rate)

    def playback(self, recovered: RecoveredAudio, blocking: bool = True) -> None:
        """Toca as amostras recuperadas pela saída de áudio padrão."""
        self._player.play(
            recovered.samples, sample_rate=recovered.sample_rate, blocking=blocking
        )

    # ==================================================================
    # Auxiliares
    # ==================================================================
    def _frames_for_bytes(self, n_bytes: int) -> int:
        """Número de frames OFDM necessários para carregar `n_bytes` de payload."""
        # bits totais / bits por frame, arredondando para cima (ceil).
        bits = n_bytes * 8
        return math.ceil(bits / self._ofdm.BITS_PER_OFDM_FRAME)
