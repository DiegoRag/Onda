import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift

class ImageFFTCrypto:
    def __init__(self):
        self.original_image = None
        self.fft_complex_data = None  # Guarda a matemática pura para conseguir reverter

    def load_image(self, filepath):
        """
        Carrega a imagem e converte para tons de cinza (matriz 2D).
        Cores RGB (3D) complicam a visualização do espectro sem ganho didático.
        """
        img = Image.open(filepath).convert('L')
        self.original_image = np.array(img)
        return self.original_image

    def apply_fft(self):
        """
        Aplica a FFT 2D e retorna o espectro visual (criptografado).
        """
        if self.original_image is None:
            return None

        # 1. Aplica a FFT Bidimensional (Gera números complexos)
        self.fft_complex_data = fft2(self.original_image)
        
        # 2. Centraliza as baixas frequências no meio da imagem (Padrão visual científico)
        shifted_fft = fftshift(self.fft_complex_data)
        
        # 3. Calcula a Magnitude (O valor absoluto do número complexo)
        magnitude = np.abs(shifted_fft)
        
        # 4. Escala Logarítmica: Como o centro é muito brilhante, aplicamos log
        # para conseguirmos enxergar os detalhes do espectro.
        magnitude_spectrum = 20 * np.log(magnitude + 1)
        
        return magnitude_spectrum

    def apply_ifft(self):
        """
        Pega os dados complexos guardados, reverte a matemática e recupera a imagem.
        """
        if self.fft_complex_data is None:
            return None

        # Aplica a Transformada Inversa diretamente nos dados originais
        restored = ifft2(self.fft_complex_data)
        
        # Pega a parte real (descartando ruídos matemáticos imaginários minúsculos)
        restored_image = np.abs(restored)
        
        return restored_image