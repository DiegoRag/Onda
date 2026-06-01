"""Cifra AES-256-CTR: a camada de confidencialidade criptográfica de verdade.

Por que CTR (Counter Mode)?
    - Cifra de fluxo: o ciphertext tem o MESMO tamanho do plaintext (sem overhead).
    - Erros de um único bit no ciphertext viram erros de um único bit no plaintext
      (sem propagação de erro). Isso importa porque o canal OFDM pode introduzir
      erros de bit ocasionais, e queremos que a voz continue inteligível mesmo se
      alguns bits virarem.
    - Sem verificação de integridade embutida. Uma chave/nonce errada produz lixo na
      saída SEM levantar exceção — falha silenciosa. Isso é aceitável aqui porque o
      modelo de ameaça do projeto é acadêmico; sistemas de produção precisam embrulhar
      o CTR com HMAC ou usar um modo AEAD (GCM, ChaCha20-Poly1305).

Por que uma classe em vez de funções soltas?
    - A chave de 32 bytes fica amarrada ao objeto cifra de uma vez. O chamador não
      consegue passar a chave errada no encrypt() vs decrypt() por acidente.
    - Variantes futuras (ex.: modo autenticado) podem herdar sem mexer nos pontos
      de chamada.
"""

from __future__ import annotations

import hashlib
import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import global_configs


class AESCipher:
    """Encripta/decripta em AES-256-CTR com uma chave fixa de 32 bytes.

    Uso típico:
        cipher = AESCipher.from_password("hunter2")
        nonce, ciphertext = cipher.encrypt(b"hello")
        recovered = cipher.decrypt(nonce, ciphertext)
        assert recovered == b"hello"
    """

    KEY_SIZE_BYTES: int = global_configs.AES_KEY_SIZE_BYTES
    NONCE_SIZE_BYTES: int = global_configs.AES_NONCE_SIZE_BYTES

    def __init__(self, key: bytes) -> None:
        """Cria uma cifra amarrada a `key` (precisa ter exatamente KEY_SIZE_BYTES)."""
        # Trava de segurança: AES-256 exige 32 bytes; qualquer outro tamanho é erro.
        if len(key) != self.KEY_SIZE_BYTES:
            raise ValueError(
                f"AES-256 requires a {self.KEY_SIZE_BYTES}-byte key, "
                f"got {len(key)} bytes."
            )
        self._key: bytes = key

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------
    @classmethod
    def from_password(cls, password: str) -> "AESCipher":
        """Deriva uma chave de 32 bytes de uma senha de tamanho arbitrário via SHA-256.

        Nota (acadêmico): SHA-256 da senha crua serve para um trabalho de faculdade,
        mas sistemas de produção PRECISAM usar PBKDF2 / Argon2 / scrypt com um salt
        aleatório e muitas iterações. Senão, a mesma senha sempre gera a mesma chave,
        o que torna ataques de rainbow-table triviais.
        """
        # SHA-256 sempre devolve 32 bytes = tamanho exato da chave AES-256.
        key = hashlib.sha256(password.encode("utf-8")).digest()
        return cls(key)

    # ------------------------------------------------------------------
    # Encriptação / decriptação
    # ------------------------------------------------------------------
    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes]:
        """Encripta `plaintext`. Retorna `(nonce, ciphertext)`.

        Um nonce aleatório novo é gerado a cada chamada. Isso faz encriptar o mesmo
        plaintext duas vezes produzir ciphertexts *diferentes* — propriedade desejável
        que dificulta ataques de análise de tráfego.
        """
        # Nonce aleatório (16 bytes): inicializa o contador do modo CTR.
        nonce = os.urandom(self.NONCE_SIZE_BYTES)
        # Monta o motor AES-CTR e processa tudo (update + finalize fecha o stream).
        encryptor = Cipher(algorithms.AES(self._key), modes.CTR(nonce)).encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return nonce, ciphertext

    def decrypt(self, nonce: bytes, ciphertext: bytes) -> bytes:
        """Decripta `ciphertext` com o `nonce` dado. Retorna o plaintext.

        O CTR não tem verificação de integridade: chave/nonce errados produzem lixo
        silenciosamente em vez de levantar erro. Cabe ao chamador qualquer checagem
        de sanidade da saída (ex.: isso soa como voz?).
        """
        # O nonce precisa ter o tamanho certo, senão a construção do CTR falha.
        if len(nonce) != self.NONCE_SIZE_BYTES:
            raise ValueError(
                f"CTR nonce must be {self.NONCE_SIZE_BYTES} bytes, "
                f"got {len(nonce)}."
            )
        # No CTR, decriptar é o mesmo processo de encriptar (XOR com o keystream).
        decryptor = Cipher(algorithms.AES(self._key), modes.CTR(nonce)).decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    # ------------------------------------------------------------------
    # Acessores somente-leitura (a chave em si NÃO é exposta de propósito)
    # ------------------------------------------------------------------
    @property
    def key_size_bytes(self) -> int:
        """Retorna o tamanho configurado da chave AES, em bytes (sempre 32 aqui)."""
        return self.KEY_SIZE_BYTES
