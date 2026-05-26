"""Cryptographic primitives: AES-256-CTR cipher and binary frame layout.

The split between `cipher` (raw crypto) and `framer` (metadata serialization)
is deliberate: the cipher only knows about bytes-in, bytes-out, while the
framer carries the audio metadata (sample rate, sample count, nonce) needed
for the receiver to reconstruct the original audio.
"""

from core.crypto.cipher import AESCipher
from core.crypto.framer import FrameHeader, Framer

__all__ = ["AESCipher", "FrameHeader", "Framer"]
