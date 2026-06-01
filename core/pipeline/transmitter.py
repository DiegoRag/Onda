"""Pipeline de transmissão de alto nível: amostras de voz -> arquivo WAV ou alto-falante.

Estágios da pipeline (em ordem):

    amostras de voz (int16)
        |
        | int16 -> bytes crus (little-endian)
        v
    bytes de plaintext
        |
        | AES-256-CTR com chave = SHA-256(senha)
        v
    (nonce, ciphertext)
        |
        | embrulha em FrameHeader -> Framer.build
        v
    bytes de payload (cabeçalho de 28 bytes + ciphertext)
        |
        | OFDMModem.bytes_to_signal com permutação = Scrambler(senha)
        v
    sinal OFDM no tempo @ 48 kHz
        |
        | prefixa [silêncio, chirp]; acrescenta [silêncio]
        v
    sinal completo de transmissão @ 48 kHz
        |
        | -> WavIO.write   (teste em loopback)
        | -> AudioPlayer.play  (pelo ar, no alto-falante)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import global_configs
from core.audio.player import AudioPlayer
from core.audio.wav_io import WavIO
from core.crypto.cipher import AESCipher
from core.crypto.framer import FrameHeader, Framer
from core.modulation.ofdm import OFDMModem
from core.modulation.scrambler import Scrambler
from core.modulation.sync import PreambleSync

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TransmissionResult:
    """Resumo do que acabou de ser transmitido (útil para feedback na UI)."""

    voice_sample_rate: int       # taxa de amostragem da fonte (Hz)
    voice_num_samples: int       # contagem original de amostras
    ciphertext_bytes: int        # bytes após o AES (== bytes de plaintext no CTR)
    ofdm_frames: int             # número de símbolos OFDM emitidos
    transmission_samples: int    # comprimento do sinal de TX completo (48 kHz)
    transmission_duration_s: float


class Transmitter:
    """Constrói e emite um sinal OFDM pronto-pro-ar a partir de amostras de voz."""

    FS: int = global_configs.FS
    PRE_SILENCE_S: float = global_configs.PRE_SILENCE_S
    POST_SILENCE_S: float = global_configs.POST_SILENCE_S

    def __init__(
        self,
        password: str,
        *,
        cipher: AESCipher | None = None,
        framer: Framer | None = None,
        ofdm: OFDMModem | None = None,
        scrambler: Scrambler | None = None,
        preamble: PreambleSync | None = None,
        player: AudioPlayer | None = None,
    ) -> None:
        """Constrói um transmissor amarrado a `password`.

        Todas as dependências são injetáveis para teste; os padrões vêm dos módulos
        deste pacote.
        """
        # Cada componente: usa o injetado, ou cria um padrão derivado da senha.
        self._cipher: AESCipher = cipher or AESCipher.from_password(password)
        self._framer: Framer = framer or Framer()
        self._ofdm: OFDMModem = ofdm or OFDMModem()
        self._scrambler: Scrambler = scrambler or Scrambler()
        self._preamble: PreambleSync = preamble or PreambleSync()
        self._player: AudioPlayer = player or AudioPlayer()

        # A permutação depende só da senha; pré-calcula uma vez.
        self._permutation: np.ndarray = self._scrambler.permutation_for_password(
            password
        )

    # ==================================================================
    # Monta o sinal completo de transmissão (NÃO toca disco / alto-falante)
    # ==================================================================
    def build_signal(
        self,
        voice_samples: np.ndarray,
        voice_sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
    ) -> tuple[np.ndarray, TransmissionResult]:
        """Roda a pipeline e retorna o sinal final @ 48 kHz + um resumo."""
        # A pipeline trabalha com voz mono 1-D.
        if voice_samples.ndim != 1:
            raise ValueError(
                f"voice_samples must be 1-D, got shape {voice_samples.shape}."
            )

        # Guarda a contagem original para o receptor cortar o padding depois.
        original_count = voice_samples.size

        # ---- Estágio 1: int16 -> bytes ----
        # Garante int16 e serializa as amostras como bytes crus (o que o AES come).
        int16_samples = voice_samples.astype(np.int16)
        plaintext = WavIO.int16_to_bytes(int16_samples)

        # ---- Estágio 2: AES-256-CTR ----
        # Confidencialidade real: devolve o nonce aleatório + o ciphertext.
        nonce, ciphertext = self._cipher.encrypt(plaintext)

        # ---- Estágio 3: framing ----
        # Cabeçalho com tudo que o receptor precisa para remontar o áudio.
        header = FrameHeader(
            nonce=nonce,
            sample_rate=voice_sample_rate,
            num_samples=original_count,
            ciphertext_length=len(ciphertext),
        )
        # payload = cabeçalho (28 bytes) + ciphertext.
        payload = self._framer.build(header, ciphertext)

        # ---- Estágio 4: modulação OFDM ----
        # Bytes -> sinal no tempo via IFFT (o coração de Fourier), com a permutação.
        ofdm_signal = self._ofdm.bytes_to_signal(payload, self._permutation)

        # ---- Estágio 5: prefixa silêncio + chirp; acrescenta silêncio ----
        # Silêncios curtos nas pontas dão folga ao receptor e evitam clipar o fim.
        pre_silence = np.zeros(
            int(self.PRE_SILENCE_S * self.FS), dtype=np.float32
        )
        post_silence = np.zeros(
            int(self.POST_SILENCE_S * self.FS), dtype=np.float32
        )
        # Sinal final: [silêncio][chirp de sincronização][dados OFDM][silêncio].
        full = np.concatenate(
            [pre_silence, self._preamble.preamble, ofdm_signal, post_silence]
        )

        # Monta o resumo da transmissão (quantos frames, duração, etc.).
        n_frames = ofdm_signal.size // self._ofdm.FRAME_LEN
        result = TransmissionResult(
            voice_sample_rate=voice_sample_rate,
            voice_num_samples=original_count,
            ciphertext_bytes=len(ciphertext),
            ofdm_frames=n_frames,
            transmission_samples=full.size,
            transmission_duration_s=full.size / self.FS,
        )
        logger.info(
            "Transmission built: %d voice samples -> %d ciphertext bytes -> "
            "%d OFDM frames -> %.2fs @ %d Hz",
            original_count, len(ciphertext), n_frames,
            result.transmission_duration_s, self.FS,
        )
        return full, result

    # ==================================================================
    # Saídas: arquivo / alto-falante
    # ==================================================================
    def to_wav(
        self,
        voice_samples: np.ndarray,
        path: str | Path,
        voice_sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
    ) -> TransmissionResult:
        """Monta o sinal e grava em `path` como um WAV int16 de 48 kHz."""
        # Monta o sinal e escreve no disco (fluxo de loopback).
        signal, result = self.build_signal(voice_samples, voice_sample_rate)
        WavIO.write(path, signal, self.FS)
        logger.info("Wrote transmission WAV to %s", path)
        return result

    def to_speaker(
        self,
        voice_samples: np.ndarray,
        voice_sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
        blocking: bool = True,
    ) -> TransmissionResult:
        """Monta o sinal e o toca pela saída de áudio padrão.

        Para transmissão pelo ar PC-a-PC, aponte o alto-falante para o microfone da
        outra máquina e chame `Receiver.from_microphone(...)` nessa máquina em
        poucos segundos.
        """
        # Monta o sinal e toca no alto-falante (fluxo pelo ar).
        signal, result = self.build_signal(voice_samples, voice_sample_rate)
        self._player.play(signal, sample_rate=self.FS, blocking=blocking)
        return result

    # ==================================================================
    # Acessores somente-leitura
    # ==================================================================
    @property
    def permutation(self) -> np.ndarray:
        """A permutação de embaralhamento derivada da senha (cópia imutável)."""
        return self._permutation.copy()
