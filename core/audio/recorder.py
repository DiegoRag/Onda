"""Captura de microfone via `sounddevice`.

Três modos de gravação:
    - `record(duration)`       bloqueante, devolve a captura inteira no fim.
    - `record_async(duration)` não-bloqueante, com duração fixa.
    - `start_stream(on_chunk)` captura ao vivo, sem fim definido; o chamador decide
                               quando chamar `stop_stream()`. Cada bloco de áudio é
                               entregue ao callback `on_chunk` (que roda na thread de
                               áudio — mantenha-o curto).

Taxa de amostragem / canais / dtype usam por padrão `global_configs.AUDIO_RECORD_*`.
Para operação TX/RX pelo ar, o receptor normalmente quer uma taxa MAIOR (FS = 48 kHz)
para capturar a banda OFDM limpa — passe uma sobrescrita.
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
    """Referência para uma gravação em andamento (retornada por `record_async`)."""

    buffer: np.ndarray             # pré-alocado; preenchido no lugar
    sample_rate: int

    def wait_until_done(self) -> np.ndarray:
        """Bloqueia até a gravação terminar; retorna as amostras capturadas."""
        # sd.wait() bloqueia até o buffer ser totalmente preenchido.
        sd.wait()
        return self.buffer.flatten()

    def cancel(self) -> None:
        """Para a gravação imediatamente. O buffer pode estar preenchido pela metade."""
        sd.stop()


class AudioRecorder:
    """Embrulha o `sounddevice` para captura de microfone bloqueante e assíncrona."""

    def __init__(
        self,
        sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
        channels: int = global_configs.AUDIO_CHANNELS,
        dtype: str = global_configs.AUDIO_DATA_TYPE,
    ) -> None:
        # Guarda os parâmetros de captura; nenhum stream aberto ainda.
        self._sample_rate: int = sample_rate
        self._channels: int = channels
        self._dtype: str = dtype
        self._stream: sd.InputStream | None = None

    # ------------------------------------------------------------------
    # Bloqueante
    # ------------------------------------------------------------------
    def record(
        self,
        duration_s: float = global_configs.AUDIO_RECORD_SAMPLE_DURATION,
    ) -> np.ndarray:
        """Grava por `duration_s` segundos. Retorna um array 1-D de `dtype`."""
        # Quantidade de amostras = duração * taxa.
        n_samples = int(duration_s * self._sample_rate)
        logger.info(
            "Recording %.2fs @ %d Hz (%s)",
            duration_s, self._sample_rate, self._dtype,
        )
        # Dispara a gravação e espera terminar (bloqueante).
        samples = sd.rec(
            n_samples,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
        )
        sd.wait()
        logger.info("Recording finished (%d samples)", n_samples)
        # flatten: (N,1) -> (N,) para ficar mono 1-D.
        return samples.flatten()

    # ------------------------------------------------------------------
    # Assíncrono
    # ------------------------------------------------------------------
    def record_async(self, duration_s: float) -> AsyncRecording:
        """Inicia uma gravação não-bloqueante. Retorna imediatamente."""
        # sd.rec já devolve o buffer que será preenchido em segundo plano;
        # embrulhamos numa AsyncRecording para o chamador esperar/cancelar depois.
        n_samples = int(duration_s * self._sample_rate)
        buffer = sd.rec(
            n_samples,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
        )
        return AsyncRecording(buffer=buffer, sample_rate=self._sample_rate)

    # ------------------------------------------------------------------
    # Streaming (estilo "aperte para falar" — o chamador decide quando parar)
    # ------------------------------------------------------------------
    def start_stream(
        self,
        on_chunk: Callable[[np.ndarray], None],
        blocksize: int = 512,
    ) -> None:
        """Abre um stream de microfone ao vivo e dispara `on_chunk(samples)` por bloco.

        IMPORTANTE: o callback roda na thread de ÁUDIO, NÃO na thread de UI. Mantenha-o
        curto — nada de chamadas de GUI. Padrão típico: empurrar as amostras para uma
        fila/lista thread-safe e deixar a thread da UI consultar num timer.

        Parameters
        ----------
        on_chunk : callable
            Função chamada com um array numpy 1-D de amostras `dtype` por bloco.
            Recebe uma *cópia* do buffer de áudio, então o callback pode guardar
            referências com segurança.
        blocksize : int
            Amostras por bloco de áudio. Menor = menos latência, mas mais overhead de
            callback. 512 ~ 32 ms a 16 kHz, um bom padrão para UI ao vivo.
        """
        # Não permite abrir dois streams ao mesmo tempo.
        if self._stream is not None:
            raise RuntimeError(
                "stream is already running; call stop_stream() first."
            )

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            # Roda na thread de áudio a cada bloco capturado.
            if status:
                logger.debug("InputStream status: %s", status)
            # indata.shape == (frames, channels). Mono -> coluna 0.
            # .copy() porque o sounddevice pode reusar o buffer depois do return.
            on_chunk(indata[:, 0].copy())

        logger.info("Starting live stream @ %d Hz (%s)", self._sample_rate, self._dtype)
        # Abre e inicia o stream de entrada com o callback acima.
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
            blocksize=blocksize,
            callback=_callback,
        )
        self._stream.start()

    def stop_stream(self) -> None:
        """Para o stream iniciado com `start_stream`. Idempotente."""
        # Se não há stream, não faz nada (idempotente).
        if self._stream is None:
            return
        logger.info("Stopping live stream.")
        # Para e fecha; garante limpar a referência mesmo se algo falhar.
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    @property
    def is_streaming(self) -> bool:
        return self._stream is not None

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------
    @staticmethod
    def list_input_devices() -> list[dict]:
        """Lista os dispositivos de entrada disponíveis (útil para diagnóstico)."""
        # Filtra os dispositivos que têm ao menos 1 canal de entrada.
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
