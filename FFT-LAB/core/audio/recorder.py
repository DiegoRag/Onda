"""Microphone capture via `sounddevice`.

Three recording modes:
    - `record(duration)`       blocking, returns the full take at the end.
    - `record_async(duration)` non-blocking with a fixed duration.
    - `start_stream(on_chunk)` open-ended live capture; the caller decides when
                               to call `stop_stream()`. Each audio block is
                               delivered to the `on_chunk` callback (which runs
                               on the audio thread — keep it short).

Sample rate / channels / dtype default to `global_configs.AUDIO_RECORD_*`.
For TX/RX over-air operation the receiver typically wants a HIGHER sample
rate (FS = 48 kHz) to capture the OFDM band cleanly — pass an override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd

import global_configs

logger = logging.getLogger(__name__)


@dataclass
class AsyncRecording:
    """Handle to an in-progress recording (returned by `record_async`)."""

    buffer: np.ndarray             # pre-allocated; filled in place
    sample_rate: int

    def wait_until_done(self) -> np.ndarray:
        """Block until the recording finishes; return the captured samples."""
        sd.wait()
        return self.buffer.flatten()

    def cancel(self) -> None:
        """Stop the recording immediately. The buffer may be partially filled."""
        sd.stop()


class AudioRecorder:
    """Wraps `sounddevice` for blocking and async microphone capture."""

    def __init__(
        self,
        sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
        channels: int = global_configs.AUDIO_CHANNELS,
        dtype: str = global_configs.AUDIO_DATA_TYPE,
    ) -> None:
        self._sample_rate: int = sample_rate
        self._channels: int = channels
        self._dtype: str = dtype
        self._stream: sd.InputStream | None = None

    # ------------------------------------------------------------------
    # Blocking
    # ------------------------------------------------------------------
    def record(
        self,
        duration_s: float = global_configs.AUDIO_RECORD_SAMPLE_DURATION,
    ) -> np.ndarray:
        """Record for `duration_s` seconds. Returns a 1-D array of `dtype`."""
        n_samples = int(duration_s * self._sample_rate)
        logger.info(
            "Recording %.2fs @ %d Hz (%s)",
            duration_s, self._sample_rate, self._dtype,
        )
        samples = sd.rec(
            n_samples,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
        )
        sd.wait()
        logger.info("Recording finished (%d samples)", n_samples)
        return samples.flatten()

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------
    def record_async(self, duration_s: float) -> AsyncRecording:
        """Start a non-blocking recording. Returns immediately."""
        n_samples = int(duration_s * self._sample_rate)
        buffer = sd.rec(
            n_samples,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
        )
        return AsyncRecording(buffer=buffer, sample_rate=self._sample_rate)

    # ------------------------------------------------------------------
    # Streaming (push-to-talk style — caller decides when to stop)
    # ------------------------------------------------------------------
    def start_stream(
        self,
        on_chunk: Callable[[np.ndarray], None],
        blocksize: int = 512,
    ) -> None:
        """Open a live microphone stream and fire `on_chunk(samples)` per block.

        IMPORTANT: the callback runs on the audio thread, NOT the UI thread.
        Keep it short — no GUI calls. Typical pattern: push the samples to a
        thread-safe queue/list and let the UI thread poll on a timer.

        Parameters
        ----------
        on_chunk : callable
            Function called with a 1-D numpy array of `dtype` samples per block.
            Receives a *copy* of the audio buffer, so the callback can keep
            references safely.
        blocksize : int
            Samples per audio block. Smaller = lower latency but more callback
            overhead. 512 ~ 32 ms at 16 kHz, a good default for live UI.
        """
        if self._stream is not None:
            raise RuntimeError(
                "stream is already running; call stop_stream() first."
            )

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("InputStream status: %s", status)
            # indata.shape == (frames, channels). Mono -> column 0.
            # .copy() because sounddevice may reuse the buffer after we return.
            on_chunk(indata[:, 0].copy())

        logger.info("Starting live stream @ %d Hz (%s)", self._sample_rate, self._dtype)
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
            blocksize=blocksize,
            callback=_callback,
        )
        self._stream.start()

    def stop_stream(self) -> None:
        """Stop the stream started with `start_stream`. Idempotent."""
        if self._stream is None:
            return
        logger.info("Stopping live stream.")
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    @property
    def is_streaming(self) -> bool:
        return self._stream is not None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def list_input_devices() -> list[dict]:
        """Enumerate available input devices (useful when troubleshooting)."""
        return [
            d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0
        ]

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def dtype(self) -> str:
        return self._dtype
