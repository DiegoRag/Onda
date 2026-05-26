"""AES-256-CTR cipher: the actual cryptographic confidentiality layer.

Why CTR (Counter Mode)?
    - Stream cipher: ciphertext has the SAME length as plaintext (no overhead).
    - Single-bit errors in ciphertext map to single-bit errors in plaintext
      (no error propagation). This matters because the OFDM channel may
      introduce occasional bit errors, and we want voice to remain
      intelligible even if a few bits flip.
    - No built-in integrity check. A wrong key/nonce produces garbage output
      WITHOUT raising — silent failure. This is acceptable here because the
      project's threat model is academic; production systems must wrap CTR
      with HMAC or use an AEAD mode (GCM, ChaCha20-Poly1305).

Why a class instead of free functions?
    - The 32-byte key is bound to the cipher object once. Callers cannot
      accidentally pass the wrong key into encrypt() vs decrypt().
    - Future variants (e.g., authenticated mode) can subclass without
      touching call sites.
"""

from __future__ import annotations

import hashlib
import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import global_configs


class AESCipher:
    """AES-256-CTR encrypt/decrypt with a fixed 32-byte key.

    Typical usage:
        cipher = AESCipher.from_password("hunter2")
        nonce, ciphertext = cipher.encrypt(b"hello")
        recovered = cipher.decrypt(nonce, ciphertext)
        assert recovered == b"hello"
    """

    KEY_SIZE_BYTES: int = global_configs.AES_KEY_SIZE_BYTES
    NONCE_SIZE_BYTES: int = global_configs.AES_NONCE_SIZE_BYTES

    def __init__(self, key: bytes) -> None:
        """Create a cipher bound to `key` (must be exactly KEY_SIZE_BYTES)."""
        if len(key) != self.KEY_SIZE_BYTES:
            raise ValueError(
                f"AES-256 requires a {self.KEY_SIZE_BYTES}-byte key, "
                f"got {len(key)} bytes."
            )
        self._key: bytes = key

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_password(cls, password: str) -> "AESCipher":
        """Derive a 32-byte key from an arbitrary-length password via SHA-256.

        Note (academic): SHA-256 of the raw password is fine for a class
        project, but production systems MUST use PBKDF2 / Argon2 / scrypt with
        a random salt and a high iteration count. Otherwise the same password
        always produces the same key, which makes rainbow-table attacks
        trivial.
        """
        key = hashlib.sha256(password.encode("utf-8")).digest()
        return cls(key)

    # ------------------------------------------------------------------
    # Encryption / decryption
    # ------------------------------------------------------------------
    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes]:
        """Encrypt `plaintext`. Returns `(nonce, ciphertext)`.

        A fresh random nonce is generated per call. This means encrypting the
        same plaintext twice produces *different* ciphertexts — a desirable
        property that prevents traffic-analysis attacks.
        """
        nonce = os.urandom(self.NONCE_SIZE_BYTES)
        encryptor = Cipher(algorithms.AES(self._key), modes.CTR(nonce)).encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return nonce, ciphertext

    def decrypt(self, nonce: bytes, ciphertext: bytes) -> bytes:
        """Decrypt `ciphertext` with the given `nonce`. Returns plaintext.

        CTR has no integrity check: a wrong key/nonce silently produces
        garbage rather than raising. The caller is responsible for any
        sanity-check on the decrypted output (e.g., does it sound like voice?).
        """
        if len(nonce) != self.NONCE_SIZE_BYTES:
            raise ValueError(
                f"CTR nonce must be {self.NONCE_SIZE_BYTES} bytes, "
                f"got {len(nonce)}."
            )
        decryptor = Cipher(algorithms.AES(self._key), modes.CTR(nonce)).decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    # ------------------------------------------------------------------
    # Read-only accessors (the key itself is intentionally NOT exposed)
    # ------------------------------------------------------------------
    @property
    def key_size_bytes(self) -> int:
        """Return the configured AES key size, in bytes (always 32 here)."""
        return self.KEY_SIZE_BYTES
