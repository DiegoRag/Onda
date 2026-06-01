import numpy as np
import sounddevice as sd
from scipy import signal
import scipy.io.wavfile as wavfile

import global_configs

class AudioGenerator:
    def __init__(self, sample_rate=global_configs.SYNTH_SAMPLE_RATE):
        # 44100 Hz é o padrão de qualidade de CD de áudio.
        # Significa que o som tem 44.100 "pontos" por segundo.
        self.sample_rate = sample_rate

    def create_time_array(self, duration_seconds=2.0):
        """
        Cria a linha do tempo (eixo X).
        Gera um array de números que vai de 0 até o tempo desejado.
        """
        # np.linspace cria pontos espaçados igualmente.
        total_samples = int(self.sample_rate * duration_seconds)
        t = np.linspace(0, duration_seconds, total_samples, endpoint=False)
        return t

    def generate_single_wave(self, wave_type, freq, amp, phase_degrees, t):
        """
        Gera a matemática de uma única onda.
        """
        # 1. Ajuste de Escalas
        # Amplitude na interface vai de 0 a 100%, mas a matemática exige de 0.0 a 1.0
        amplitude_real = amp / 100.0 
        
        # Fase na interface é em graus (0 a 360), a matemática exige radianos
        phase_rad = phase_degrees * (np.pi / 180.0)

        # 2. A Fórmula Base (2 * pi * f * t + fase)
        onda_base = 2 * np.pi * freq * t + phase_rad

        # 3. Escolhendo a forma da onda
        if wave_type == "Senoidal (sin)":
            wave = amplitude_real * np.sin(onda_base)
            
        elif wave_type == "Quadrada (square)":
            wave = amplitude_real * signal.square(onda_base)
            
        elif wave_type == "Dente de Serra (saw)":
            wave = amplitude_real * signal.sawtooth(onda_base)
            
        elif wave_type == "Triangular (tri)":
            # Triangular é um dente de serra com largura de 0.5
            wave = amplitude_real * signal.sawtooth(onda_base, width=0.5)
            
        else:
            wave = np.zeros_like(t) # Silêncio se der erro

        return wave

    def play_audio(self, final_wave_array):
        """
        Toca a onda no alto-falante.
        """
        # sd.play toca a matriz de números usando a placa de som
        sd.play(final_wave_array, self.sample_rate)

    def stop_audio(self):
        """
        Para o áudio imediatamente.
        """
        sd.stop()

    def export_wav(self, filename, final_wave_array):
        """
        Salva a matriz de números como um arquivo de som .wav
        """
        # Para salvar em WAV padrão, o áudio precisa ser convertido 
        # de Float (0.0 a 1.0) para Inteiros de 16 bits (-32768 a 32767)
        audio_int16 = np.int16(final_wave_array * global_configs.INT16_MAX)
        wavfile.write(filename, self.sample_rate, audio_int16)
        print(f"Áudio salvo com sucesso em: {filename}")