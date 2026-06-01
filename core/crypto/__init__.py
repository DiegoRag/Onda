"""Primitivas criptográficas: cifra AES-256-CTR e layout binário do frame.

A separação entre `cipher` (cripto pura) e `framer` (serialização de metadados) é
proposital: a cifra só sabe de bytes-entram, bytes-saem, enquanto o framer carrega os
metadados de áudio (taxa de amostragem, contagem de amostras, nonce) que o receptor
precisa para reconstruir o áudio original.
"""

from core.crypto.cipher import AESCipher
from core.crypto.framer import FrameHeader, Framer

__all__ = ["AESCipher", "FrameHeader", "Framer"]
