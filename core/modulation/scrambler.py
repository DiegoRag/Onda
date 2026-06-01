"""Embaralhamento no domínio da frequência: permutação de subportadoras com chave.

  *** ISTO NÃO É CRIPTOGRAFIA. ***

O que faz
---------
Sem embaralhamento, o OFDM coloca o i-ésimo símbolo QPSK de um frame na i-ésima
subportadora de dados (DATA_BINS[i]). O scrambler embaralha esse mapeamento com uma
permutação derivada da senha do usuário: o i-ésimo símbolo vai para
DATA_BINS[permutation[i]]. O receptor, sabendo a senha, regenera a mesma permutação e
desfaz o embaralhamento.

Por que NÃO é criptográfico
---------------------------
O embaralhamento é REVERSÍVEL SEM A CHAVE por qualquer um que:
  - saiba que esse esquema é usado (o princípio de Kerckhoffs diz que devemos supor
    que o atacante sabe), e
  - consiga coletar ciphertext suficiente para fazer análise estatística da energia/
    correlação por bin.

A confidencialidade real neste projeto vem do AES-256-CTR (ver core/crypto/cipher.py).
A camada de embaralhamento existe só para demonstrar que o domínio de Fourier é, ele
próprio, um domínio onde dá para manipular dados com uma chave — útil didaticamente,
desprezível criptograficamente.

Separação de chaves
-------------------
A semente que dirige a permutação NÃO é a chave AES. Fazemos o hash da senha junto com
uma tag de separação de domínio (SCRAMBLE_SEED_TAG) para que o scrambler e a cifra usem
material derivado distinto:

    aes_key       = SHA-256(password)
    scramble_seed = SHA-256(password + b"scramble-v1")

Isso é boa prática em qualquer sistema que derive múltiplos segredos de uma única senha
— evita reutilizar os mesmos bytes de chave em contextos diferentes.
"""

from __future__ import annotations

import hashlib

import numpy as np

import global_configs


class Scrambler:
    """Gera e aplica permutações de subportadoras OFDM controladas por chave."""

    SEED_TAG: bytes = global_configs.SCRAMBLE_SEED_TAG

    # Número de subportadoras de dados (padrão = 21).
    DEFAULT_LENGTH: int = len(global_configs.DATA_BINS)

    def __init__(self, length: int | None = None) -> None:
        """Configura o scrambler para permutações de tamanho `length`.

        Por padrão usa o número de bins de dados do OFDM (21 na spec). Pode ser
        usado com outros tamanhos em testes.
        """
        # Se nada for passado, cai no tamanho padrão (21).
        self._length: int = length if length is not None else self.DEFAULT_LENGTH

    # ------------------------------------------------------------------
    # Derivação da semente
    # ------------------------------------------------------------------
    @classmethod
    def derive_seed(cls, password: str) -> int:
        """Retorna um inteiro de 8 bytes derivado de `password`.

        Derivado de SHA-256(password + SEED_TAG), independente da derivação da chave
        AES. Truncado para 64 bits porque o RNG do numpy aceita sementes inteiras
        arbitrárias, mas 8 bytes já são entropia de sobra para uma permutação de
        21 elementos (21! ~ 5.1e19 ~ 65.7 bits — semente de 64 bits é confortável).
        """
        # Hash da senha CONCATENADA com a tag (separação de domínio) e pega 8 bytes.
        digest = hashlib.sha256(password.encode("utf-8") + cls.SEED_TAG).digest()
        return int.from_bytes(digest[:8], "little")

    # ------------------------------------------------------------------
    # Permutação
    # ------------------------------------------------------------------
    def build_permutation(self, seed: int) -> np.ndarray:
        """Retorna uma permutação determinística de [0, 1, ..., length-1].

        A mesma semente sempre gera a mesma permutação. Usa
        numpy.random.default_rng (PCG64), que é reprodutível entre plataformas
        (diferente do antigo MT19937 em alguns casos extremos).
        """
        # RNG semeado de forma determinística -> mesma semente, mesma permutação.
        rng = np.random.default_rng(seed)
        return rng.permutation(self._length)

    # ------------------------------------------------------------------
    # Aplicar / inverter
    # ------------------------------------------------------------------
    @staticmethod
    def scramble(symbols: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        """Reordena os símbolos de forma que output[permutation[i]] = symbols[i].

        Equivalente a: `output = np.empty_like(symbols); output[permutation] = symbols`.
        O numpy expressa isso como indexação avançada do LADO ESQUERDO.
        """
        # Os dois arrays precisam ter o mesmo formato (um destino por símbolo).
        if symbols.shape != permutation.shape:
            raise ValueError(
                f"symbols.shape={symbols.shape} and permutation.shape="
                f"{permutation.shape} must match."
            )
        # Escreve cada símbolo na sua posição permutada.
        output = np.empty_like(symbols)
        output[permutation] = symbols
        return output

    @staticmethod
    def unscramble(symbols: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        """Inverso de `scramble`: output[i] = symbols[permutation[i]]."""
        # Mesma validação de formato.
        if symbols.shape != permutation.shape:
            raise ValueError(
                f"symbols.shape={symbols.shape} and permutation.shape="
                f"{permutation.shape} must match."
            )
        # Lê os símbolos na ordem da permutação -> desfaz o embaralhamento.
        return symbols[permutation]

    # ------------------------------------------------------------------
    # Conveniência
    # ------------------------------------------------------------------
    def permutation_for_password(self, password: str) -> np.ndarray:
        """Conveniência: deriva a semente E constrói a permutação numa só chamada."""
        return self.build_permutation(self.derive_seed(password))

    @property
    def length(self) -> int:
        """Comprimento da permutação (número de subportadoras de dados)."""
        return self._length
