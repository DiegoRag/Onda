"""High-level transmit pipeline: voice samples -> WAV file or speaker.

Pipeline stages (in order):

    voice samples (int16)
        |
        | optional: SpectralDenoiser
        v
    cleaned int16 samples
        |
        | int16 -> raw bytes (little-endian)
        v
    plaintext bytes
        |
        | AES-256-CTR with key = SHA-256(password)
        v
    (nonce, ciphertext)
        |
        | wrap in FrameHeader -> Framer.build
        v
    payload bytes (28-byte header + ciphertext)
        |
        | OFDMModem.bytes_to_signal with permutation = Scrambler(password)
        v
    OFDM time-domain signal @ 48 kHz
        |
        | prepend [silence, chirp]; append [silence]
        v
    full transmission signal @ 48 kHz
        |
        | -> WavIO.write   (loopback testing)
        | -> AudioPlayer.play  (over-air to speaker)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import global_configs
from core.audio.denoiser import SpectralDenoiser
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
    """Summary of what was just transmitted (useful for UI feedback)."""

    voice_sample_rate: int       # source audio sample rate (Hz)
    voice_num_samples: int       # original sample count (pre-denoise)
    ciphertext_bytes: int        # bytes after AES (== plaintext bytes for CTR)
    ofdm_frames: int             # number of OFDM symbols emitted
    transmission_samples: int    # length of the full TX signal (48 kHz)
    transmission_duration_s: float
    denoise_applied: bool


class Transmitter:
    """Builds and emits an over-air-ready OFDM signal from voice samples."""

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
        denoiser: SpectralDenoiser | None = None,
        player: AudioPlayer | None = None,
    ) -> None:
        """Build a transmitter bound to `password`.

        All dependencies are injectable for testing; defaults pull from the
        modules in this package.
        """
        self._cipher: AESCipher = cipher or AESCipher.from_password(password)
        self._framer: Framer = framer or Framer()
        self._ofdm: OFDMModem = ofdm or OFDMModem()
        self._scrambler: Scrambler = scrambler or Scrambler()
        self._preamble: PreambleSync = preamble or PreambleSync()
        self._denoiser: SpectralDenoiser = denoiser or SpectralDenoiser()
        self._player: AudioPlayer = player or AudioPlayer()

        # Permutation depends only on the password; precompute once.
        self._permutation: np.ndarray = self._scrambler.permutation_for_password(
            password
        )

    # ==================================================================
    # Build the full transmission signal (does NOT touch disk / speakers)
    # ==================================================================
    def build_signal(
        self,
        voice_samples: np.ndarray,
        voice_sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
        denoise: bool = False,
    ) -> tuple[np.ndarray, TransmissionResult]:
        """Run the pipeline and return the final 48 kHz signal + a summary."""
        if voice_samples.ndim != 1:
            raise ValueError(
                f"voice_samples must be 1-D, got shape {voice_samples.shape}."
            )

        original_count = voice_samples.size

        # ---- Stage 1: optional spectral denoise ----
        if denoise:
            logger.info("Applying spectral subtraction denoise.")
            cleaned = self._denoiser.denoise(voice_samples, voice_sample_rate)
        else:
            cleaned = voice_samples

        # ---- Stage 2: int16 -> bytes ----
        int16_samples = cleaned.astype(np.int16)
        plaintext = WavIO.int16_to_bytes(int16_samples)

        # ---- Stage 3: AES-256-CTR ----
        nonce, ciphertext = self._cipher.encrypt(plaintext)

        # ---- Stage 4: framing ----
        header = FrameHeader(
            nonce=nonce,
            sample_rate=voice_sample_rate,
            num_samples=original_count,
            ciphertext_length=len(ciphertext),
        )
        payload = self._framer.build(header, ciphertext)

        # ---- Stage 5: OFDM modulate ----
        ofdm_signal = self._ofdm.bytes_to_signal(payload, self._permutation)

        # ---- Stage 6: prepend silence + chirp; append silence ----
        pre_silence = np.zeros(
            int(self.PRE_SILENCE_S * self.FS), dtype=np.float32
        )
        post_silence = np.zeros(
            int(self.POST_SILENCE_S * self.FS), dtype=np.float32
        )
        full = np.concatenate(
            [pre_silence, self._preamble.preamble, ofdm_signal, post_silence]
        )

        n_frames = ofdm_signal.size // self._ofdm.FRAME_LEN
        result = TransmissionResult(
            voice_sample_rate=voice_sample_rate,
            voice_num_samples=original_count,
            ciphertext_bytes=len(ciphertext),
            ofdm_frames=n_frames,
            transmission_samples=full.size,
            transmission_duration_s=full.size / self.FS,
            denoise_applied=denoise,
        )
        logger.info(
            "Transmission built: %d voice samples -> %d ciphertext bytes -> "
            "%d OFDM frames -> %.2fs @ %d Hz",
            original_count, len(ciphertext), n_frames,
            result.transmission_duration_s, self.FS,
        )
        return full, result

    # ==================================================================
    # Sinks: file / speaker
    # ==================================================================
    def to_wav(
        self,
        voice_samples: np.ndarray,
        path: str | Path,
        voice_sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
        denoise: bool = False,
    ) -> TransmissionResult:
        """Build the signal and write it to `path` as a 48 kHz int16 WAV."""
        signal, result = self.build_signal(
            voice_samples, voice_sample_rate, denoise=denoise
        )
        WavIO.write(path, signal, self.FS)
        logger.info("Wrote transmission WAV to %s", path)
        return result

    def to_speaker(
        self,
        voice_samples: np.ndarray,
        voice_sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
        denoise: bool = False,
        blocking: bool = True,
    ) -> TransmissionResult:
        """Build the signal and play it through the default audio output.

        For PC-to-PC over-air transmission, point the speaker at the other
        machine's microphone, then call `Receiver.from_microphone(...)` on
        that machine within a few seconds.
        """
        signal, result = self.build_signal(
            voice_samples, voice_sample_rate, denoise=denoise
        )
        self._player.play(signal, sample_rate=self.FS, blocking=blocking)
        return result

    # ==================================================================
    # Read-only accessors
    # ==================================================================
    @property
    def permutation(self) -> np.ndarray:
        """The scramble permutation derived from the password (immutable copy)."""
        return self._permutation.copy()
