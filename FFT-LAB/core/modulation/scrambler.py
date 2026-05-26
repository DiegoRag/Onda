"""Frequency-domain scrambling: key-controlled subcarrier permutation.

  *** THIS IS NOT CRYPTOGRAPHY. ***

What it does
------------
Without scrambling, OFDM places the i-th QPSK symbol of a frame into the i-th
data subcarrier bin (DATA_BINS[i]). The scrambler shuffles that mapping with
a permutation derived from the user's password: the i-th symbol goes into
DATA_BINS[permutation[i]]. The receiver, knowing the password, regenerates
the same permutation and undoes the shuffle.

Why it is NOT cryptographic
---------------------------
Scrambling is REVERSIBLE WITHOUT THE KEY by anyone who:
  - knows that this scheme is used (Kerckhoffs's principle says we must
    assume the attacker does), and
  - can collect enough ciphertext to do statistical analysis on per-bin
    energy / correlation.

Real confidentiality in this project is provided by AES-256-CTR (see
core/crypto/cipher.py). The scramble layer exists only to demonstrate that
the Fourier domain is itself a domain in which we can manipulate data with a
key — pedagogically useful, cryptographically negligible.

Key separation
--------------
The seed used to drive the permutation is NOT the AES key. We hash the
password together with a domain-separation tag (SCRAMBLE_SEED_TAG) so that
the scrambler and the cipher use distinct derived material:

    aes_key       = SHA-256(password)
    scramble_seed = SHA-256(password + b"scramble-v1")

This is good practice in any system that derives multiple secrets from one
password — it avoids reuse of the same key bytes in different contexts.
"""

from __future__ import annotations

import hashlib

import numpy as np

import global_configs


class Scrambler:
    """Generates and applies key-controlled permutations of OFDM subcarriers."""

    SEED_TAG: bytes = global_configs.SCRAMBLE_SEED_TAG

    # Number of data subcarriers (default = 21).
    DEFAULT_LENGTH: int = len(global_configs.DATA_BINS)

    def __init__(self, length: int | None = None) -> None:
        """Configure the scrambler for permutations of size `length`.

        Defaults to the number of OFDM data bins (21 in the spec). Could be
        used with different lengths for testing.
        """
        self._length: int = length if length is not None else self.DEFAULT_LENGTH

    # ------------------------------------------------------------------
    # Seed derivation
    # ------------------------------------------------------------------
    @classmethod
    def derive_seed(cls, password: str) -> int:
        """Return an 8-byte integer derived from `password`.

        Derived from SHA-256(password + SEED_TAG), independent of the AES key
        derivation. Truncated to 64 bits because numpy's RNG accepts arbitrary
        integer seeds but 8 bytes is plenty of entropy for a permutation of
        21 elements (21! ~ 5.1e19 ~ 65.7 bits — 64-bit seed is comfortable).
        """
        digest = hashlib.sha256(password.encode("utf-8") + cls.SEED_TAG).digest()
        return int.from_bytes(digest[:8], "little")

    # ------------------------------------------------------------------
    # Permutation
    # ------------------------------------------------------------------
    def build_permutation(self, seed: int) -> np.ndarray:
        """Return a deterministic permutation of [0, 1, ..., length-1].

        The same seed always yields the same permutation. Uses
        numpy.random.default_rng (PCG64) which is reproducible across
        platforms (unlike the legacy MT19937 in some edge cases).
        """
        rng = np.random.default_rng(seed)
        return rng.permutation(self._length)

    # ------------------------------------------------------------------
    # Apply / invert
    # ------------------------------------------------------------------
    @staticmethod
    def scramble(symbols: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        """Reorder symbols so that output[permutation[i]] = symbols[i].

        Equivalently: `output = np.empty_like(symbols); output[permutation] = symbols`.
        Numpy expresses this as fancy indexing on the LEFT side.
        """
        if symbols.shape != permutation.shape:
            raise ValueError(
                f"symbols.shape={symbols.shape} and permutation.shape="
                f"{permutation.shape} must match."
            )
        output = np.empty_like(symbols)
        output[permutation] = symbols
        return output

    @staticmethod
    def unscramble(symbols: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        """Inverse of `scramble`: output[i] = symbols[permutation[i]]."""
        if symbols.shape != permutation.shape:
            raise ValueError(
                f"symbols.shape={symbols.shape} and permutation.shape="
                f"{permutation.shape} must match."
            )
        return symbols[permutation]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def permutation_for_password(self, password: str) -> np.ndarray:
        """Convenience: derive seed AND build permutation in one call."""
        return self.build_permutation(self.derive_seed(password))

    @property
    def length(self) -> int:
        """Permutation length (number of data subcarriers)."""
        return self._length
