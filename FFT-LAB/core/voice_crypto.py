import hashlib
import os

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import global_configs


class VoiceCrypto:
    """Real cryptography pipeline for voice: record -> AES-256-CTR -> recover.

    IMPORTANT: this is REAL encryption (AES, key-based). It is conceptually
    different from the FFT/IFFT used in the other tabs, which is a reversible
    transform that anyone can undo WITHOUT a key. Confidentiality here comes
    from AES; without the correct password the audio cannot be recovered.

    The Fourier/OFDM modulation step (turning the ciphertext into an ultrasonic
    signal via IFFT) is intentionally left for a later stage.
    """

    def __init__(self, sample_rate: int = global_configs.AUDIO_RECORD_SAMPLE_RATE,
                 channels: int = global_configs.AUDIO_CHANNELS,
                 dtype: str = global_configs.AUDIO_DATA_TYPE):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Recording / playback
    # ------------------------------------------------------------------
    def record(self, duration_s: float = global_configs.AUDIO_RECORD_SAMPLE_DURATION) -> np.ndarray:
        """Record mono audio (blocking) and return a 1D int16 array."""
        samples = sd.rec(
            int(duration_s * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
        )
        sd.wait()
        return samples.flatten()

    def play(self, samples: np.ndarray) -> None:
        sd.play(samples, self.sample_rate)

    def stop(self) -> None:
        sd.stop()

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------
    @staticmethod
    def derive_key(password: str) -> bytes:
        """SHA-256(password) -> 32 bytes (AES-256 key).

        Academic-grade: a production system must use PBKDF2 or Argon2 with a
        random salt and many iterations.
        """
        return hashlib.sha256(password.encode("utf-8")).digest()

    # ------------------------------------------------------------------
    # AES-256-CTR  (pure crypto: bytes in, bytes out)
    # ------------------------------------------------------------------
    @staticmethod
    def encrypt(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
        """Return (nonce, ciphertext). A fresh random nonce is used per call,
        so encrypting the same audio twice yields different ciphertext."""
        nonce = os.urandom(global_configs.AES_NONCE_SIZE_BYTES)
        encryptor = Cipher(algorithms.AES(key), modes.CTR(nonce)).encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return nonce, ciphertext

    @staticmethod
    def decrypt(nonce: bytes, ciphertext: bytes, key: bytes) -> bytes:
        """Return plaintext. CTR has no integrity check: a wrong key produces
        garbage silently instead of raising."""
        decryptor = Cipher(algorithms.AES(key), modes.CTR(nonce)).decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    # ------------------------------------------------------------------
    # High-level helpers: audio array <-> encrypted bytes
    # ------------------------------------------------------------------
    def encrypt_audio(self, samples: np.ndarray, password: str) -> tuple[bytes, bytes]:
        """Serialize int16 samples to bytes and AES-encrypt them."""
        key = self.derive_key(password)
        plaintext = samples.astype(np.int16).tobytes()
        return self.encrypt(plaintext, key)

    def decrypt_audio(self, nonce: bytes, ciphertext: bytes, password: str) -> np.ndarray:
        """AES-decrypt and rebuild the int16 sample array."""
        key = self.derive_key(password)
        plaintext = self.decrypt(nonce, ciphertext, key)
        # frombuffer is read-only; copy so the result can be played/edited.
        return np.frombuffer(plaintext, dtype=np.int16).copy()

    # ------------------------------------------------------------------
    # WAV I/O
    # ------------------------------------------------------------------
    def save_wav(self, path: str, samples: np.ndarray) -> None:
        wavfile.write(path, self.sample_rate, samples.astype(np.int16))

    @staticmethod
    def ciphertext_to_samples(ciphertext: bytes) -> np.ndarray:
        """Reinterpret ciphertext bytes as int16 samples.

        The result sounds like pure noise: a concrete demonstration of what
        AES output 'sounds' like before the OFDM modulation stage exists.
        """
        return np.frombuffer(ciphertext, dtype=np.int16).copy()

    def save_encrypted_wav(self, path: str, ciphertext: bytes) -> None:
        """Write the ciphertext (as int16 noise samples) to a WAV file."""
        wavfile.write(path, self.sample_rate, self.ciphertext_to_samples(ciphertext))
