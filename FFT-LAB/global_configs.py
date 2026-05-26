"""
Central configuration for the FFT-LAB project.

Single source of truth for audio/crypto/modulation parameters. Importable as
`import global_configs` because the app is launched from the FFT-LAB directory
(main.py), which puts this folder on sys.path.

This module is intentionally const-only (no logic) so the parameters can be
inspected and discussed in isolation. Any change here ripples through the
entire pipeline — read the comments before tweaking.

Project context (spec_v3, adapted for over-air PC-to-PC transmission):
    A pipeline that records voice from a microphone, encrypts it with
    AES-256-CTR, modulates the encrypted bytes via OFDM (using IFFT to
    synthesize a multi-carrier signal), optionally scrambles the subcarrier
    assignment with a key, and transmits the result through a normal computer
    speaker. A second machine captures the audio with a microphone, runs the
    inverse pipeline (FFT -> unscramble -> decrypt) and plays back the voice.

    The transmission band is **6-10 kHz (audible)** rather than ultrasonic
    (18-22 kHz). Consumer-grade microphones and speakers (electret cells,
    laptop loudspeakers) have poor response above ~15 kHz; the audible band
    gives a much more reliable link at the cost of an audible "modem-like"
    chirp during transmission.

Where Fourier appears in this pipeline (for the report):
    1. IFFT in OFDM modulation     -> synthesizes the multi-carrier signal.
    2. FFT in OFDM demodulation    -> recovers data from the captured signal.
    3. FFT/IFFT in spectral denoise -> noise spectrum subtraction (STFT).
    4. Subcarrier permutation       -> key-controlled frequency-domain shuffle.
"""

from __future__ import annotations

# =============================================================================
# Voice recording (microphone capture for the source audio)
# =============================================================================
# Voice does not need CD quality: 16 kHz is intelligible and halves the data
# volume vs 44.1 kHz, which keeps OFDM transmission times reasonable.
AUDIO_RECORD_SAMPLE_DURATION: float = 3.0
AUDIO_RECORD_SAMPLE_RATE: int = 16_000
AUDIO_CHANNELS: int = 1
AUDIO_DATA_TYPE: str = "int16"

# Legacy aliases (kept so the existing voice_crypto.py UI tab keeps working).
RECORD_SAMPLE_RATE: int = AUDIO_RECORD_SAMPLE_RATE
RECORD_CHANNELS: int = AUDIO_CHANNELS
RECORD_DTYPE: str = AUDIO_DATA_TYPE
DEFAULT_RECORD_DURATION_S: float = AUDIO_RECORD_SAMPLE_DURATION

# =============================================================================
# Signal synthesizer (the "Sintetizador" tab — unrelated to OFDM)
# =============================================================================
# Kept at CD quality on purpose: that tab targets audible music, unlike voice
# capture which only needs 16 kHz.
SYNTH_SAMPLE_RATE: int = 44_100

# =============================================================================
# Transmission signal (OFDM carrier rate)
# =============================================================================
# 48 kHz is the universal "high-quality audio" sample rate that every consumer
# soundcard supports cleanly. Nyquist is 24 kHz, well above our 10 kHz band.
FS: int = 48_000

# =============================================================================
# Transmission band (where the OFDM subcarriers live)
# =============================================================================
# Original spec used 18-22 kHz (ultrasonic). We use 6-10 kHz because consumer
# microphones/speakers have far better response there. Audible trade-off:
# transmission sounds like a high-pitched "modem".
F_MIN: int = 6_000
F_MAX: int = 10_000

# =============================================================================
# OFDM core parameters
# =============================================================================
# N_FFT: size of the IFFT/FFT used per OFDM symbol. 256 is the sweet spot:
#   - bin spacing = FS / N_FFT = 48000 / 256 = 187.5 Hz (fine enough to fit 22
#     subcarriers inside a 4 kHz band)
#   - small enough that a Python FFT is essentially free
# N_CP: cyclic prefix length (samples). 64 = N_FFT/4, the classical choice.
#   The CP absorbs multi-path / timing slop without eating throughput.
N_FFT: int = 256
N_CP: int = 64

# Subcarrier indices used within the band. BIN = freq * N_FFT / FS.
# For F_MIN=6000, FS=48000, N_FFT=256: BIN_MIN = 32 (exactly).
# BIN_MAX = BIN_MIN + 21 -> 22 bins total = 1 pilot + 21 data carriers.
BIN_MIN: int = int(F_MIN * N_FFT / FS)            # 32
BIN_MAX: int = BIN_MIN + 21                       # 53
PILOT_BIN: int = BIN_MIN                          # bin 32 (lowest = pilot)
DATA_BINS: list[int] = list(range(BIN_MIN + 1, BIN_MAX + 1))  # 33..53 inclusive

# QPSK: 2 bits per complex symbol. Each OFDM frame carries 21 symbols.
BITS_PER_SYMBOL: int = 2
BITS_PER_OFDM_FRAME: int = len(DATA_BINS) * BITS_PER_SYMBOL   # 42 bits

# Pilot symbol value. Real-valued, unit amplitude — keeps channel estimation
# numerically simple: H = X[PILOT_BIN] / PILOT_VALUE.
PILOT_VALUE: complex = complex(1.0, 0.0)

# =============================================================================
# Preamble (chirp used for synchronization between sender and receiver)
# =============================================================================
# Linear chirp spanning the data band. 50 ms is long enough for a sharp
# correlation peak; 5 ms ramps at each end avoid spectral splatter.
PREAMBLE_DURATION_S: float = 0.050
PREAMBLE_F_START: int = F_MIN
PREAMBLE_F_END: int = F_MAX
PREAMBLE_RAMP_S: float = 0.005

# =============================================================================
# AES-256-CTR (the actual cryptographic layer — provides confidentiality)
# =============================================================================
AES_KEY_SIZE_BYTES: int = 32       # AES-256
AES_NONCE_SIZE_BYTES: int = 16     # CTR convention

# =============================================================================
# Frame header (binary layout of metadata that precedes the ciphertext)
# =============================================================================
# Layout: [nonce 16B][sample_rate 4B][num_samples 4B][ciphertext_length 4B]
# struct format: '<16sIII'  -> little-endian, 16 bytes + 3 uint32
HEADER_SIZE_BYTES: int = AES_NONCE_SIZE_BYTES + 4 + 4 + 4  # 28 bytes

# =============================================================================
# Spectral scrambling (obfuscation, NOT cryptography)
# =============================================================================
# A key-controlled permutation of the OFDM subcarrier assignments. Real
# confidentiality is provided by AES; this layer only demonstrates that we can
# manipulate the frequency domain with a key. It is vulnerable to known-
# plaintext and statistical attacks — see core/modulation/scrambler.py.
SCRAMBLE_SEED_TAG: bytes = b"scramble-v1"

# =============================================================================
# Spectral denoising (single-channel noise reduction via spectral subtraction)
# =============================================================================
# Assumes the first DENOISE_NOISE_FLOOR_FRAMES of input represent silence
# (used to characterize the noise spectrum). Recommend a brief pause at the
# start of any recording before speaking.
DENOISE_FRAME_SIZE: int = 1024           # samples per STFT window
DENOISE_HOP_SIZE: int = 256              # 75% overlap (1024 - 256*3 = 256)
DENOISE_NOISE_FLOOR_FRAMES: int = 5      # frames used to estimate noise floor
DENOISE_OVERSUBTRACTION_FACTOR: float = 2.0   # alpha — aggressiveness
DENOISE_SPECTRAL_FLOOR: float = 0.05     # beta — residual to avoid musical noise

# =============================================================================
# WAV I/O
# =============================================================================
# Silence padding at the start/end of a transmission WAV. Helps the receiver
# settle before the chirp arrives and avoids clipping the tail.
PRE_SILENCE_S: float = 0.050
POST_SILENCE_S: float = 0.050

# Int16 conventions. Multiplying by INT16_MAX (32767), not 32768, avoids
# producing +32768 which would wrap to -32768 in two's-complement int16.
INT16_MAX: int = 32_767
INT16_HEADROOM: float = 0.9   # peak-normalize to 90% of int16 range

# =============================================================================
# Sanity assertions (catch typos that would silently corrupt the pipeline)
# =============================================================================
assert BIN_MAX < N_FFT // 2, "Data bins must stay below Nyquist (N_FFT/2)."
assert F_MAX <= FS // 2, "F_MAX exceeds Nyquist (FS/2)."
assert BIN_MIN > 0, "PILOT_BIN must be strictly positive (bin 0 = DC)."
assert len(DATA_BINS) * BITS_PER_SYMBOL == BITS_PER_OFDM_FRAME
assert HEADER_SIZE_BYTES == AES_NONCE_SIZE_BYTES + 12
