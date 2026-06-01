"""Preâmbulo de sincronização: um chirp linear + detector por correlação cruzada.

Transmissão pelo ar significa que transmissor e receptor NÃO compartilham um relógio.
O receptor precisa descobrir *quando* os dados OFDM começam dentro do áudio capturado.
Resolvemos isso com uma forma de onda conhecida — um chirp linear — colocada
imediatamente antes dos dados. O receptor faz a correlação cruzada do sinal capturado
contra uma cópia local do chirp; o pico de correlação marca o fim do chirp (ou seja,
o início dos dados).

Por que um chirp?
-----------------
Um chirp varre a frequência ao longo do tempo. Sua autocorrelação tem um pico estreito
e nítido (próximo de um delta), dando um instante de tempo preciso. Uma senoide pura
também correlaciona bem, mas sua autocorrelação é uma senoide — vários picos, sem
instante inequívoco. Uma sequência tipo ruído também funciona (sequências PN são usadas
no CDMA), mas um chirp ainda serve de sonda da resposta em frequência do canal, já que
visita todas as frequências da banda.

Rampas nas bordas
------------------
O chirp é multiplicado por uma meia-janela de Hann nos primeiros e últimos
PREAMBLE_RAMP_S segundos. Sem as rampas, o início abrupto da senoide age como uma função
degrau e espalha energia para fora da banda de 6-10 kHz, o que pode vazar cliques
audíveis e desperdiçar potência de transmissão. A rampa reduz a amplitude suavemente
até zero nas bordas.

Regra de detecção
-----------------
Usamos correlação cruzada (com pico de amplitude limitado a ~1.0 quando o sinal casa
exatamente com o preâmbulo). O pico é aceito se exceder
`threshold_factor * mean(|correlação|)`. Fator padrão = 10×.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

import global_configs


class PreambleSync:
    """Gera e detecta o preâmbulo de chirp usado para alinhar os frames pelo ar."""

    FS: int = global_configs.FS
    DURATION_S: float = global_configs.PREAMBLE_DURATION_S
    F_START: int = global_configs.PREAMBLE_F_START
    F_END: int = global_configs.PREAMBLE_F_END
    RAMP_S: float = global_configs.PREAMBLE_RAMP_S

    def __init__(self) -> None:
        # O preâmbulo é determinístico; constrói uma vez e guarda em cache.
        self._preamble: np.ndarray = self._build_preamble()

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------
    def _build_preamble(self) -> np.ndarray:
        """Sintetiza o chirp linear com meias-rampas de Hann nas duas pontas."""
        # Eixo do tempo: n_samples pontos espaçados de 1/FS segundos.
        n_samples = int(self.DURATION_S * self.FS)
        t = np.arange(n_samples) / self.FS

        # Frequência instantânea: f(t) = F_START + (F_END - F_START) * t / T
        # Fase integrada: phi(t) = 2*pi * (F_START*t + (F_END - F_START)/(2T) * t^2)
        # A fase é a integral da frequência; o chirp é o cosseno dessa fase.
        sweep_rate = (self.F_END - self.F_START) / self.DURATION_S
        phase = 2.0 * np.pi * (self.F_START * t + 0.5 * sweep_rate * t * t)
        chirp = np.cos(phase).astype(np.float32)

        # Meias-rampas de Hann. A janela de Hann é 0.5 * (1 - cos(pi * t / T_ramp))
        # para t em [0, T_ramp], subindo de 0 a 1. Usamos a mesma forma espelhada
        # para a rampa final. Só aplica se houver espaço para as duas rampas.
        ramp_n = int(self.RAMP_S * self.FS)
        if ramp_n > 0 and 2 * ramp_n <= n_samples:
            ramp_idx = np.arange(ramp_n)
            ramp = 0.5 * (1.0 - np.cos(np.pi * ramp_idx / ramp_n))
            # Rampa de subida no início e a mesma rampa invertida no fim.
            chirp[:ramp_n] *= ramp.astype(np.float32)
            chirp[-ramp_n:] *= ramp[::-1].astype(np.float32)

        return chirp

    # ------------------------------------------------------------------
    # Acessores públicos
    # ------------------------------------------------------------------
    @property
    def preamble(self) -> np.ndarray:
        """Retorna uma cópia da forma de onda do preâmbulo (float32)."""
        # Cópia para o chamador não conseguir mutar o preâmbulo em cache.
        return self._preamble.copy()

    @property
    def length(self) -> int:
        """Comprimento do preâmbulo em amostras."""
        return self._preamble.size

    # ------------------------------------------------------------------
    # Detecção
    # ------------------------------------------------------------------
    def detect(
        self,
        captured_signal: np.ndarray,
        threshold_factor: float = 10.0,
    ) -> int:
        """Localiza o fim do preâmbulo dentro de `captured_signal`.

        Parameters
        ----------
        captured_signal : np.ndarray
            array float 1-D (a captura do microfone ou um WAV carregado).
        threshold_factor : float
            Quanto o pico de correlação precisa exceder a média para ser aceito.

        Returns
        -------
        int
            Índice da primeira amostra DEPOIS do preâmbulo, ou -1 se nenhum pico
            exceder o limiar de aceitação.
        """
        # Valida formato e descarta sinais curtos demais para conter o preâmbulo.
        if captured_signal.ndim != 1:
            raise ValueError(
                f"captured_signal must be 1-D, got shape {captured_signal.shape}."
            )
        if captured_signal.size < self._preamble.size:
            return -1

        # Correlação cruzada via implementação baseada em FFT do scipy. mode='valid'
        # dá um índice de pico k que significa: o preâmbulo casa melhor em
        # captured_signal[k : k + preamble_len].
        # method='fft' mantém isso rápido mesmo em capturas de vários segundos.
        corr = sps.correlate(
            captured_signal.astype(np.float64),
            self._preamble.astype(np.float64),
            mode="valid",
            method="fft",
        )
        # Trabalhamos com o valor absoluto da correlação para achar o pico.
        abs_corr = np.abs(corr)
        mean_corr = abs_corr.mean()
        peak_idx = int(np.argmax(abs_corr))
        peak_val = abs_corr[peak_idx]

        # Rejeita se não houver correlação ou se o pico não se destacar o bastante
        # da média (sinal sem o preâmbulo -> retorna -1).
        if mean_corr <= 0 or peak_val < threshold_factor * mean_corr:
            return -1

        # Converte "índice de início do preâmbulo" em "primeira amostra DEPOIS dele".
        return peak_idx + self._preamble.size
