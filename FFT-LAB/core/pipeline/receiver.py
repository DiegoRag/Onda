"""High-level receive pipeline: WAV file or microphone -> voice samples.

Inverse of `Transmitter`:

    captured / loaded float32 signal @ 48 kHz
        |
        | PreambleSync.detect  -> index of OFDM start
        v
    OFDM region of the signal
        |
        | demodulate HEADER_SIZE_BYTES bytes worth of frames first
        v
    FrameHeader (nonce, sample_rate, num_samples, ciphertext_length)
        |
        | demodulate the remaining ciphertext_length bytes
        v
    payload bytes (header + ciphertext)
        |
        | Framer.extract_ciphertext
        v
    ciphertext bytes
        |
        | AESCipher.decrypt with nonce from header
        v
    plaintext bytes (raw little-endian int16)
        |
        | WavIO.bytes_to_int16 -> truncate to num_samples
        v
    recovered voice samples (int16)
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
    """Result of a receive operation."""

    samples: np.ndarray            # int16, length == header.num_samples
    sample_rate: int               # source voice sample rate (from header)
    header: FrameHeader            # full header for diagnostics
    h_estimate_magnitude: float    # |H| at the first frame — link strength
    preamble_index: int            # sample offset where OFDM starts


class ReceptionError(Exception):
    """Raised when the preamble cannot be located or the frame is malformed."""


class Receiver:
    """Decodes WAV files or live microphone captures back into voice samples."""

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
        """Build a receiver bound to `password`.

        The recorder defaults to a fresh `AudioRecorder` configured for the
        OFDM rate (48 kHz, mono, float32), NOT the voice rate.
        """
        self._cipher: AESCipher = cipher or AESCipher.from_password(password)
        self._framer: Framer = framer or Framer()
        self._ofdm: OFDMModem = ofdm or OFDMModem()
        self._scrambler: Scrambler = scrambler or Scrambler()
        self._preamble: PreambleSync = preamble or PreambleSync()
        self._recorder: AudioRecorder = recorder or AudioRecorder(
            sample_rate=self.FS,
            channels=1,
            dtype="float32",
        )
        self._player: AudioPlayer = player or AudioPlayer()

        self._permutation: np.ndarray = self._scrambler.permutation_for_password(
            password
        )

    # ==================================================================
    # Sources: file / microphone
    # ==================================================================
    def from_wav(self, path: str | Path) -> RecoveredAudio:
        """Decode a previously-saved transmission WAV (loopback flow)."""
        sample_rate, signal = WavIO.read(path)
        if sample_rate != self.FS:
            raise ReceptionError(
                f"expected transmission sample rate {self.FS} Hz, "
                f"WAV reports {sample_rate} Hz."
            )
        return self.decode_signal(signal)

    def from_microphone(self, duration_s: float) -> RecoveredAudio:
        """Capture `duration_s` seconds from the mic and decode.

        Recommended duration: a couple of seconds beyond the expected TX
        length to absorb sender/receiver kickoff delay.
        """
        signal = self._recorder.record(duration_s)
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)
        return self.decode_signal(signal)

    # ==================================================================
    # Core decoder
    # ==================================================================
    def decode_signal(self, signal: np.ndarray) -> RecoveredAudio:
        """Run preamble detection + OFDM demod + AES decrypt on a 48 kHz signal."""
        if signal.ndim != 1:
            raise ReceptionError(
                f"signal must be 1-D, got shape {signal.shape}."
            )

        # ---- Stage 1: preamble detection ----
        ofdm_start = self._preamble.detect(signal.astype(np.float32))
        if ofdm_start < 0:
            raise ReceptionError("preamble not detected in captured signal.")
        logger.info("Preamble located; OFDM payload begins at sample %d.", ofdm_start)

        payload_signal = signal[ofdm_start:]

        # ---- Stage 2: demodulate enough frames to read the header ----
        header_frames_needed = self._frames_for_bytes(
            self._framer.HEADER_SIZE_BYTES
        )
        header_signal = payload_signal[
            : header_frames_needed * self._ofdm.FRAME_LEN
        ]
        if header_signal.size < header_frames_needed * self._ofdm.FRAME_LEN:
            raise ReceptionError(
                "captured signal too short to contain a frame header."
            )
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

        # ---- Stage 3: demodulate the full payload ----
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

        # ---- Stage 4: AES-256-CTR decrypt ----
        ciphertext = self._framer.extract_ciphertext(
            payload_bytes, header.ciphertext_length
        )
        plaintext = self._cipher.decrypt(header.nonce, ciphertext)

        # ---- Stage 5: bytes -> int16, truncate to original sample count ----
        samples = WavIO.bytes_to_int16(plaintext)
        samples = samples[: header.num_samples]

        # Diagnostic: measure |H| on the first frame for UI feedback.
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
    # Sinks: file / speaker
    # ==================================================================
    def save_wav(self, recovered: RecoveredAudio, path: str | Path) -> None:
        """Save recovered samples to a mono int16 WAV at the source rate."""
        WavIO.write(path, recovered.samples.astype(np.float32), recovered.sample_rate)

    def playback(self, recovered: RecoveredAudio, blocking: bool = True) -> None:
        """Play recovered samples through the default audio output."""
        self._player.play(
            recovered.samples, sample_rate=recovered.sample_rate, blocking=blocking
        )

    # ==================================================================
    # Helpers
    # ==================================================================
    def _frames_for_bytes(self, n_bytes: int) -> int:
        """Number of OFDM frames required to carry `n_bytes` payload bytes."""
        bits = n_bytes * 8
        return math.ceil(bits / self._ofdm.BITS_PER_OFDM_FRAME)
