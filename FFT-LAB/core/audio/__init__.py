"""Audio I/O: WAV file read/write, microphone capture, speaker playback,
and spectral-subtraction denoising.
"""

from core.audio.denoiser import SpectralDenoiser
from core.audio.player import AudioPlayer
from core.audio.recorder import AudioRecorder
from core.audio.wav_io import WavIO

__all__ = ["AudioPlayer", "AudioRecorder", "SpectralDenoiser", "WavIO"]
