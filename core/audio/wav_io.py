"""I/O de arquivos WAV e conversões int16 <-> bytes.

O formato WAV é uma casca fina em volta de amostras PCM intercaladas. Usamos
`scipy.io.wavfile` porque já faz parte das dependências, lida com int16/int32/float32
de forma transparente e é mais rápido que o módulo `wave` da stdlib para arquivos
grandes.

Este módulo também é responsável pela interconversão int16<->bytes que fica na fronteira
entre "amostras de áudio" (arrays numpy) e "bytes de dados" (a unidade sobre a qual o
AES opera). A conversão é bit-exata nos dois sentidos, então um roundtrip limpo via
WAV-loopback produz amostras idênticas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

import global_configs


class WavIO:
    """Leituras/escritas no sistema de arquivos mais conversões int16 <-> bytes."""

    INT16_MAX: int = global_configs.INT16_MAX
    HEADROOM: float = global_configs.INT16_HEADROOM

    # ------------------------------------------------------------------
    # I/O de arquivo
    # ------------------------------------------------------------------
    @classmethod
    def write(
        cls,
        path: str | Path,
        signal: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Salva um sinal float como WAV mono int16.

        O sinal é normalizado pelo pico para HEADROOM (90%) da faixa do int16, para
        evitar clipping. Multiplicamos por `INT16_MAX` (32767), não 32768, porque
        +32768 daria a volta para -32768 em complemento de dois.
        """
        # Trabalhamos só com sinal mono 1-D.
        if signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {signal.shape}.")

        # Acha o pico (maior valor absoluto) para normalizar.
        peak = float(np.max(np.abs(signal)))
        if peak == 0.0:
            # Sinal totalmente silencioso -> grava zeros (evita divisão por zero).
            scaled = np.zeros_like(signal, dtype=np.int16)
        else:
            # Normaliza para [-1,1], aplica a margem (90%) e escala para int16.
            scaled = (
                signal.astype(np.float64) / peak * cls.HEADROOM * cls.INT16_MAX
            ).astype(np.int16)

        wavfile.write(str(path), sample_rate, scaled)

    @classmethod
    def read(cls, path: str | Path) -> tuple[int, np.ndarray]:
        """Carrega um WAV e retorna `(sample_rate, amostras_float32 em [-1, 1])`.

        Entrada estéreo é reduzida ao canal esquerdo (consistente com o resto da
        pipeline, que é mono do começo ao fim).
        """
        sample_rate, samples = wavfile.read(str(path))

        # Estéreo -> mono (canal esquerdo).
        if samples.ndim > 1:
            samples = samples[:, 0]

        # Normaliza para float32 em [-1, 1] conforme o dtype de origem
        # (cada tipo tem um valor máximo diferente para dividir).
        if samples.dtype == np.int16:
            float_samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype == np.int32:
            float_samples = samples.astype(np.float32) / 2147483648.0
        elif samples.dtype == np.float32:
            float_samples = samples
        elif samples.dtype == np.float64:
            float_samples = samples.astype(np.float32)
        else:
            raise ValueError(f"Unsupported WAV dtype {samples.dtype}.")

        return int(sample_rate), float_samples

    # ------------------------------------------------------------------
    # int16 <-> bytes  (usado no caminho de encriptar/decriptar)
    # ------------------------------------------------------------------
    @staticmethod
    def int16_to_bytes(samples: np.ndarray) -> bytes:
        """Serializa um array int16 1-D em bytes crus little-endian.

        2 bytes por amostra. A conversão é reversível e bit-exata via
        `bytes_to_int16`.
        """
        # Garante o dtype int16 antes de serializar.
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        # tobytes usa sempre a ordem de bytes nativa; forçamos little-endian ('<i2')
        # para manter a representação compatível com WAV em qualquer plataforma.
        return samples.astype("<i2").tobytes()

    @staticmethod
    def bytes_to_int16(data: bytes) -> np.ndarray:
        """Inverso de `int16_to_bytes`. Retorna uma cópia *gravável*."""
        # Cada amostra int16 ocupa 2 bytes; um total ímpar seria inválido.
        if len(data) % 2 != 0:
            raise ValueError(
                f"byte length must be even (2 bytes/sample), got {len(data)}."
            )
        # frombuffer devolve uma view somente-leitura do buffer; copiamos para o
        # chamador poder modificar o resultado.
        return np.frombuffer(data, dtype="<i2").astype(np.int16).copy()
