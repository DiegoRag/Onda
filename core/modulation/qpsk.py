"""Modulação QPSK (4-QAM, com código de Gray): bits <-> símbolos complexos.

INTUIÇÃO NO MUNDO REAL (sem fórmulas):
    Cada "símbolo" QPSK é um número complexo, e um número complexo é simplesmente
    uma seta no plano: ela tem um TAMANHO (a amplitude) e um ÂNGULO (a fase). No
    OFDM, cada símbolo desses vira a "receita" de UMA onda senoidal: o tamanho da
    seta diz com que VOLUME aquela frequência vai tocar, e o ângulo diz em que ponto
    do ciclo ela COMEÇA (o adiantamento/atraso da onda).

    O QPSK é a forma mais simples de guardar informação nessa seta: mantemos o
    tamanho sempre igual (volume fixo) e usamos só 4 ÂNGULOS possíveis. Como são 4
    opções, cada seta carrega 2 bits (4 = 2x2). As 4 setas apontam para as 4
    "quinas" do plano:

         imag
          |
    (1,0) o   o (0,0)
          |
    ------+------ real
          |
    (1,1) o   o (0,1)
          |

    Pontos vizinhos diferem por exatamente UM bit (código de Gray). Por quê? Porque
    o ruído costuma empurrar uma seta para a quina VIZINHA; se vizinhos diferem por 1
    bit só, um deslize pequeno erra 1 bit em vez de 2. Minimiza o estrago.

CUIDADO (armadilha de tipo): os bits precisam virar inteiros COM SINAL antes da
conta. Com inteiros sem sinal (uint8), a operação "1 menos 2" não dá -1: ela "dá a
volta" e vira 255, produzindo setas erradas silenciosamente. Por isso convertemos
para int8 explicitamente.
"""

from __future__ import annotations

# numpy: trabalhamos com arrays (vetores) de bits e de números complexos.
import numpy as np


class QPSKModem:
    """Modulador/demodulador QPSK sem estado (não guarda nada entre chamadas)."""

    # Quantos bits cada símbolo (cada seta) carrega: 2.
    BITS_PER_SYMBOL: int = 2

    # Fator que mantém o TAMANHO da seta igual a 1 (volume normalizado). 1/raiz(2).
    # No mundo real: garante que todos os símbolos tenham a mesma "potência".
    _NORM: float = 1.0 / np.sqrt(2.0)

    # ------------------------------------------------------------------
    # Sentido direto: bits -> símbolos complexos (setas)
    # ------------------------------------------------------------------
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Transforma um fluxo de bits em setas (símbolos QPSK)."""
        # Garante que a entrada é um array numpy (aceita lista, etc.).
        bits = np.asarray(bits)
        # Precisa ser uma fila simples de bits (1 dimensão), não uma matriz.
        if bits.ndim != 1:
            raise ValueError(f"bits must be 1-D, got shape {bits.shape}.")
        # Como cada símbolo usa 2 bits, a quantidade de bits tem que ser par.
        if bits.size % self.BITS_PER_SYMBOL != 0:
            raise ValueError(
                f"bit count must be a multiple of {self.BITS_PER_SYMBOL}, "
                f"got {bits.size}."
            )

        # Converte para inteiro COM SINAL (int8) — sem isso, a conta abaixo erra
        # (uint8 transborda em "1 - 2*bit" e vira 255 em vez de -1).
        b_signed = bits.astype(np.int8)
        # Separa os bits: os de posição par (0,2,4,...) controlam o eixo horizontal
        # (parte real da seta); os ímpares (1,3,5,...) controlam o eixo vertical
        # (parte imaginária). Ou seja, cada par de bits define 1 quina do plano.
        b0 = b_signed[0::2]
        b1 = b_signed[1::2]

        # Monta a seta. "1 - 2*bit" converte o bit em sinal: bit 0 -> +1, bit 1 -> -1.
        # b0 decide se a seta vai para a direita (+) ou esquerda (-) no eixo real;
        # b1 decide se vai para cima (+) ou para baixo (-) no eixo imaginário.
        # Multiplicar por 1/raiz(2) encolhe a seta para tamanho 1 (volume padrão).
        # Resultado: uma das 4 quinas, com amplitude 1 e fase em 45/135/225/315 graus.
        symbols = ((1 - 2 * b0) + 1j * (1 - 2 * b1)) * self._NORM
        # Devolve como complex128 (precisão dupla) para casar com o resto da pipeline.
        return symbols.astype(np.complex128)

    # ------------------------------------------------------------------
    # Sentido inverso: símbolos complexos (setas) -> bits
    # ------------------------------------------------------------------
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Decide quais bits cada seta representa, pela quina mais próxima.

        Como as 4 quinas estão uma em cada quadrante, basta olhar o SINAL das partes
        real e imaginária da seta recebida — não precisa medir ângulo exato:
            - se a seta caiu no lado esquerdo (parte real negativa) -> b0 = 1
            - se a seta caiu embaixo (parte imaginária negativa)    -> b1 = 1
        Isso é a "decisão dura": jogamos a seta ruidosa para a quina mais próxima.
        """
        # Garante array numpy e que seja uma fila simples (1 dimensão).
        symbols = np.asarray(symbols)
        if symbols.ndim != 1:
            raise ValueError(f"symbols must be 1-D, got shape {symbols.shape}.")

        # Cada seta vira 2 bits, então o vetor de saída tem o dobro de elementos.
        n = symbols.size
        bits = np.empty(2 * n, dtype=np.uint8)
        # b0 (posições pares): 1 quando a seta está à esquerda (real < 0), senão 0.
        bits[0::2] = (symbols.real < 0).astype(np.uint8)
        # b1 (posições ímpares): 1 quando a seta está embaixo (imag < 0), senão 0.
        bits[1::2] = (symbols.imag < 0).astype(np.uint8)
        # Devolve os bits intercalados de volta na ordem original (b0,b1,b0,b1,...).
        return bits
