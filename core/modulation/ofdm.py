"""Modulação/demodulação OFDM — o coração da história de Fourier.

INTUIÇÃO NO MUNDO REAL (sem fórmulas):
    Imagine um coral com 256 cantores, cada um capaz de segurar UMA nota (uma
    frequência) específica. Para enviar dados, damos a alguns cantores uma "receita":
    com que VOLUME cantar (amplitude) e em que momento do ciclo COMEÇAR (fase). Cada
    receita dessas é um número complexo — uma seta com tamanho e ângulo (os símbolos
    QPSK do módulo qpsk.py).

    - A IFFT é o maestro que MISTURA todas essas notas numa única onda sonora no
      tempo: ela soma as senoides de todos os cantores num só sinal. Fourier aqui é
      um GERADOR de som a partir de "quanto de cada frequência".

    - A FFT é o ouvido afinadíssimo que, ao escutar a onda misturada, descobre de
      volta QUANTO de cada frequência havia e com qual fase. Fourier aqui é um
      ANALISADOR.

    No nosso caso usamos 21 cantores "úteis" (as 21 subportadoras de dados) + 1
    cantor de referência (o piloto). Cada um dos 21 segura uma frequência entre 6 e
    10 kHz, e o número complexo que entregamos a ele diz a amplitude e a fase daquela
    frequência. São, portanto, 21 números complexos = 21 frequências com fase+amplitude.

Por que a saída precisa ser "real"
    Som de verdade é um número real por amostra (uma pressão de ar), não um número
    complexo. Uma onda real é feita de cossenos, e cada cosseno, no mundo de Fourier,
    é a soma de duas rotações: uma de frequência positiva e outra negativa, "gêmeas
    espelhadas". Por isso, antes de misturar, colocamos em cada frequência negativa o
    "espelho conjugado" da positiva: assim as partes imaginárias se cancelam na soma
    e sobra um som real, tocável.

Prefixo cíclico (a "margem de segurança")
    Copiamos o finalzinho de cada bloco para o começo dele. Isso cria um respiro: se
    o som chega com eco ou um pequeno atraso, a bagunça cai nessa margem e não estraga
    os dados. É como deixar um espaço em branco entre palavras para elas não se
    colarem.

Piloto (a "nota de afinação")
    Mandamos uma nota conhecida (sempre a mesma) num cantor reservado. O receptor
    sabe exatamente como ela deveria soar; comparando como ela CHEGOU, ele mede o
    quanto o ambiente (alto-falante, ar, microfone) distorceu tudo e corrige as
    outras notas na mesma medida.
"""

from __future__ import annotations

# numpy: arrays de números (o espectro, o sinal no tempo, etc.).
import numpy as np

# Constantes do projeto (tamanho da FFT, bins, piloto...).
import global_configs
# NOSSA FFT/IFFT feita à mão (sem usar a da biblioteca numpy).
from core.modulation.fft_from_scratch import fft_iterative, ifft_iterative
# O mapeador de bits <-> setas (símbolos QPSK).
from core.modulation.qpsk import QPSKModem


class OFDMModem:
    """Transmissor e receptor OFDM, amarrados aos parâmetros numéricos da spec."""

    # Tamanho da (I)FFT = quantos "cantores"/frequências existem no total: 256.
    N_FFT: int = global_configs.N_FFT
    # Tamanho do prefixo cíclico (a margem de segurança), em amostras: 64.
    N_CP: int = global_configs.N_CP
    # Comprimento total de um frame transmitido: margem + corpo = 64 + 256 = 320.
    FRAME_LEN: int = N_FFT + N_CP                       # 320

    # Qual cantor guarda a nota de referência (o piloto): o bin 32.
    PILOT_BIN: int = global_configs.PILOT_BIN
    # Qual nota o piloto canta (sempre a mesma, conhecida): 1+0j (volume 1, fase 0).
    PILOT_VALUE: complex = global_configs.PILOT_VALUE
    # Quais cantores carregam dados de fato (os 21 bins), como vetor de índices.
    DATA_BINS: np.ndarray = np.asarray(global_configs.DATA_BINS, dtype=np.intp)
    # Quantos cantores de dados existem: 21.
    N_DATA_BINS: int = len(global_configs.DATA_BINS)    # 21

    # Quantos bits cabem em um frame: 21 cantores x 2 bits/seta = 42 bits.
    BITS_PER_OFDM_FRAME: int = global_configs.BITS_PER_OFDM_FRAME  # 42
    # (Não usado diretamente; só informativo.)
    BYTES_PER_OFDM_FRAME: int = BITS_PER_OFDM_FRAME // 8           # não usado

    def __init__(self, qpsk: QPSKModem | None = None) -> None:
        """Guarda o mapeador QPSK (cria um padrão se nenhum for passado)."""
        # Se o chamador injetou um QPSKModem (ex.: em teste), usa ele; senão, cria um.
        self._qpsk: QPSKModem = qpsk if qpsk is not None else QPSKModem()

    # ==================================================================
    # UM FRAME — modulação (dados -> som)
    # ==================================================================
    def modulate_frame(
        self,
        data_symbols: np.ndarray,
        permutation: np.ndarray,
    ) -> np.ndarray:
        """Pega 21 setas (símbolos QPSK) e gera um pedaço de som (320 amostras)."""
        # Confere que vieram exatamente 21 setas (uma por cantor de dados).
        if data_symbols.shape != (self.N_DATA_BINS,):
            raise ValueError(
                f"data_symbols must have shape ({self.N_DATA_BINS},), "
                f"got {data_symbols.shape}."
            )
        # Confere que a permutação (a ordem embaralhada) também tem 21 posições.
        if permutation.shape != (self.N_DATA_BINS,):
            raise ValueError(
                f"permutation must have shape ({self.N_DATA_BINS},), "
                f"got {permutation.shape}."
            )

        # Começa com TODOS os 256 cantores em silêncio (zero = nenhuma energia
        # naquela frequência). Vamos "ligar" só o piloto e os 21 de dados.
        spectrum = np.zeros(self.N_FFT, dtype=np.complex128)

        # Liga o piloto: entrega a nota de referência conhecida ao cantor reservado.
        spectrum[self.PILOT_BIN] = self.PILOT_VALUE

        # Embaralha QUAL cantor recebe QUAL seta (controlado por chave/senha).
        # "scrambled[permutation[i]] = data_symbols[i]" reposiciona as 21 setas;
        # a permutação identidade (0,1,2,...) significaria "sem embaralhar".
        scrambled = np.empty_like(data_symbols)
        scrambled[permutation] = data_symbols
        # Entrega as 21 setas (já embaralhadas) aos 21 cantores de dados.
        spectrum[self.DATA_BINS] = scrambled

        # Cria os "gêmeos espelhados" nas frequências negativas (simetria conjugada),
        # para a mistura final dar um som REAL (sem parte imaginária sobrando).
        # 'positive' são os índices das frequências positivas (1 até a metade-1).
        positive = np.arange(1, self.N_FFT // 2)
        # Em cada frequência negativa (N-k) colocamos o conjugado da positiva (k).
        spectrum[self.N_FFT - positive] = np.conj(spectrum[positive])

        # A IFFT (o maestro): mistura todas as frequências numa única onda no tempo.
        # Aqui é a SÍNTESE — "quanto de cada frequência" vira som. Usamos NOSSA IFFT
        # manual; ela devolve uma lista de complexos, que voltamos a virar array.
        time_signal = np.asarray(ifft_iterative(spectrum), dtype=np.complex128)

        # Por causa da simetria conjugada, a parte imaginária é ~zero (lixo numérico
        # ~1e-15). Ficamos só com a parte real, que é o som de verdade.
        real_signal = time_signal.real

        # Prefixo cíclico: copia as últimas 64 amostras para o começo (a margem de
        # segurança contra ecos/atrasos). Resultado: 64 + 256 = 320 amostras.
        cp = real_signal[-self.N_CP:]
        frame = np.concatenate([cp, real_signal]).astype(np.float32)

        # Garante que o frame saiu com o tamanho certo (320).
        assert frame.shape == (self.FRAME_LEN,)
        # Devolve o pedaço de som pronto.
        return frame

    # ==================================================================
    # UM FRAME — demodulação (som -> dados)
    # ==================================================================
    def demodulate_frame(
        self,
        received: np.ndarray,
        permutation: np.ndarray,
    ) -> tuple[np.ndarray, complex]:
        """Pega um pedaço de som (320 amostras) e recupera as 21 setas QPSK."""
        # Confere que veio exatamente um frame inteiro (320 amostras).
        if received.shape != (self.FRAME_LEN,):
            raise ValueError(
                f"received must have shape ({self.FRAME_LEN},), "
                f"got {received.shape}."
            )

        # Joga fora a margem de segurança (as primeiras 64 amostras do prefixo
        # cíclico); o que importa são as 256 amostras do corpo.
        useful = received[self.N_CP:]

        # A FFT (o ouvido afinado): escuta a onda e descobre QUANTO de cada
        # frequência havia, com qual fase. Aqui é a ANÁLISE. Usamos NOSSA FFT manual.
        spectrum = np.asarray(fft_iterative(useful), dtype=np.complex128)

        # Mede a distorção do ambiente comparando a nota de referência (piloto)
        # recebida com a que sabíamos ter mandado. 'h_estimate' é "o quanto o
        # caminho mexeu no som" (volume e fase).
        h_estimate = spectrum[self.PILOT_BIN] / self.PILOT_VALUE

        # Se o piloto chegou zerado (ex.: sinal mudo), não dá para dividir — usaríamos
        # os bins crus para evitar nan/inf.
        if h_estimate == 0:
            equalized = spectrum[self.DATA_BINS]
        else:
            # Divide cada cantor de dados pela distorção medida -> "desfaz" o efeito
            # do ambiente e recupera as setas como foram enviadas (equalização).
            equalized = spectrum[self.DATA_BINS] / h_estimate

        # Desembaralha: coloca as 21 setas de volta na ordem lógica original
        # (inverso do embaralhamento feito na modulação).
        unscrambled = equalized[permutation]

        # Devolve as 21 setas recuperadas + a distorção medida (útil p/ diagnóstico:
        # tamanho de h = força do sinal; ângulo de h = desvio de fase).
        return unscrambled, complex(h_estimate)

    # ==================================================================
    # MUITOS FRAMES — bytes <-> sinal (conveniência)
    # ==================================================================
    def bytes_to_signal(self, data: bytes, permutation: np.ndarray) -> np.ndarray:
        """Codifica um monte de bytes num sinal OFDM longo (vários frames)."""
        # Explode os bytes em bits individuais (8 bits por byte, do mais significativo
        # ao menos). Essa ordem PRECISA bater com a usada na recepção.
        bit_array = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

        # Calcula quantos bits faltam para fechar o último frame (múltiplo de 42).
        pad = (-len(bit_array)) % self.BITS_PER_OFDM_FRAME
        # Se faltarem, completa com zeros (um frame parcial não pode ser transmitido).
        if pad:
            bit_array = np.concatenate([bit_array, np.zeros(pad, dtype=np.uint8)])

        # Agora a quantidade de bits é múltiplo exato de 42 -> sabemos quantos frames.
        n_frames = len(bit_array) // self.BITS_PER_OFDM_FRAME

        # Reserva de uma vez todo o espaço do sinal final (mais rápido que ir colando
        # pedaço por pedaço dentro do loop).
        signal = np.empty(n_frames * self.FRAME_LEN, dtype=np.float32)

        # Para cada frame:
        for i in range(n_frames):
            # Recorta os 42 bits desse frame.
            chunk = bit_array[
                i * self.BITS_PER_OFDM_FRAME : (i + 1) * self.BITS_PER_OFDM_FRAME
            ]
            # Transforma os 42 bits em 21 setas (símbolos QPSK).
            symbols = self._qpsk.modulate(chunk)
            # Transforma as 21 setas em 320 amostras de som (a IFFT acontece aqui).
            frame = self.modulate_frame(symbols, permutation)
            # Encaixa esse pedaço de som na posição certa do sinal final.
            signal[i * self.FRAME_LEN : (i + 1) * self.FRAME_LEN] = frame

        # Devolve o sinal OFDM completo.
        return signal

    def signal_to_bytes(
        self,
        signal: np.ndarray,
        n_bytes: int,
        permutation: np.ndarray,
    ) -> bytes:
        """Decodifica um sinal OFDM de volta nos bytes originais."""
        # O decodificador espera uma fila simples (1 dimensão) de amostras.
        if signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {signal.shape}.")

        # Quantos frames inteiros de 320 amostras cabem no sinal.
        n_frames = signal.size // self.FRAME_LEN
        # Sem nem um frame completo, não há o que decodificar.
        if n_frames == 0:
            return b""

        # Reserva espaço para todos os bits que vamos recuperar (42 por frame).
        bit_buffer = np.empty(
            n_frames * self.BITS_PER_OFDM_FRAME, dtype=np.uint8
        )

        # Para cada frame:
        for i in range(n_frames):
            # Recorta as 320 amostras desse frame.
            frame = signal[i * self.FRAME_LEN : (i + 1) * self.FRAME_LEN]
            # Recupera as 21 setas (a FFT acontece aqui); ignoramos a distorção (_).
            symbols, _ = self.demodulate_frame(frame, permutation)
            # Decide os 42 bits a partir das 21 setas.
            bits = self._qpsk.demodulate(symbols)
            # Guarda esses 42 bits na posição certa do buffer.
            bit_buffer[
                i * self.BITS_PER_OFDM_FRAME : (i + 1) * self.BITS_PER_OFDM_FRAME
            ] = bits

        # Junta os bits de volta em bytes (8 a 8). Se sobrar, packbits completa com
        # zeros; por isso cortamos exatamente no tamanho pedido (descarta o padding).
        packed = np.packbits(bit_buffer).tobytes()
        return packed[:n_bytes]

    # ==================================================================
    # Diagnóstico
    # ==================================================================
    @classmethod
    def frame_duration_s(cls, fs: int = global_configs.FS) -> float:
        """Quantos segundos um frame OFDM ocupa, na taxa de amostragem `fs`."""
        # amostras-por-frame dividido por amostras-por-segundo = segundos-por-frame.
        return cls.FRAME_LEN / fs
