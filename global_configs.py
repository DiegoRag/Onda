"""
Configuração central do projeto FFT-LAB.

Fonte única de verdade para os parâmetros de áudio/cripto/modulação. Importável como
`import global_configs` porque o app é iniciado a partir do diretório FFT-LAB
(main.py), que coloca esta pasta no sys.path.

Este módulo é propositalmente só de constantes (sem lógica), para os parâmetros poderem
ser inspecionados e discutidos isoladamente. Qualquer mudança aqui se propaga por toda a
pipeline — leia os comentários antes de mexer.

Contexto do projeto (spec_v3, adaptado para transmissão PC-a-PC pelo ar):
    Uma pipeline que grava voz de um microfone, encripta com AES-256-CTR, modula os
    bytes encriptados via OFDM (usando a IFFT para sintetizar um sinal multi-portadora),
    opcionalmente embaralha a atribuição das subportadoras com uma chave e transmite o
    resultado por um alto-falante comum de computador. Uma segunda máquina captura o
    áudio com um microfone, roda a pipeline inversa (FFT -> desembaralha -> decripta) e
    toca a voz de volta.

    A banda de transmissão é **6-10 kHz (audível)** em vez de ultrassônica (18-22 kHz).
    Microfones e alto-falantes de consumo (cápsulas de eletreto, alto-falantes de
    notebook) têm resposta ruim acima de ~15 kHz; a banda audível dá um enlace muito
    mais confiável, ao custo de um chirp audível "tipo modem" durante a transmissão.

Onde Fourier aparece nesta pipeline (para o relatório):
    1. IFFT na modulação OFDM      -> sintetiza o sinal multi-portadora.
    2. FFT na demodulação OFDM     -> recupera os dados do sinal capturado.
    3. Permutação de subportadoras -> embaralhamento na frequência controlado por chave.
"""

from __future__ import annotations

# =============================================================================
# Gravação de voz (captura de microfone do áudio de origem)
# =============================================================================
# Voz não precisa de qualidade de CD: 16 kHz é inteligível e reduz à metade o volume
# de dados vs 44.1 kHz, o que mantém os tempos de transmissão OFDM razoáveis.
AUDIO_RECORD_SAMPLE_DURATION: float = 3.0   # duração padrão de gravação (segundos)
AUDIO_RECORD_SAMPLE_RATE: int = 16_000      # taxa de amostragem da voz (Hz)
AUDIO_CHANNELS: int = 1                      # mono
AUDIO_DATA_TYPE: str = "int16"              # 16 bits por amostra

# Apelidos legados (mantidos para a aba de UI voice_crypto.py continuar funcionando).
RECORD_SAMPLE_RATE: int = AUDIO_RECORD_SAMPLE_RATE
RECORD_CHANNELS: int = AUDIO_CHANNELS
RECORD_DTYPE: str = AUDIO_DATA_TYPE
DEFAULT_RECORD_DURATION_S: float = AUDIO_RECORD_SAMPLE_DURATION

# =============================================================================
# Sintetizador de sinais (a aba "Sintetizador" — sem relação com o OFDM)
# =============================================================================
# Mantido em qualidade de CD de propósito: essa aba mira música audível, diferente da
# captura de voz que só precisa de 16 kHz.
SYNTH_SAMPLE_RATE: int = 44_100

# =============================================================================
# Sinal de transmissão (taxa da portadora OFDM)
# =============================================================================
# 48 kHz é a taxa universal de "áudio de alta qualidade" que toda placa de som de
# consumo suporta sem problema. Nyquist é 24 kHz, bem acima da nossa banda de 10 kHz.
FS: int = 48_000

# =============================================================================
# Banda de transmissão (onde ficam as subportadoras OFDM)
# =============================================================================
# A spec original usava 18-22 kHz (ultrassônico). Usamos 6-10 kHz porque microfones/
# alto-falantes de consumo têm resposta muito melhor ali. Trade-off audível: a
# transmissão soa como um "modem" agudo.
F_MIN: int = 6_000
F_MAX: int = 10_000

# =============================================================================
# Parâmetros centrais do OFDM
# =============================================================================
# N_FFT: tamanho da IFFT/FFT usada por símbolo OFDM. 256 é o ponto ideal:
#   - espaçamento de bin = FS / N_FFT = 48000 / 256 = 187.5 Hz (fino o bastante para
#     caber 22 subportadoras numa banda de 4 kHz)
#   - pequeno o bastante para uma FFT em Python ser essencialmente gratuita
# N_CP: comprimento do prefixo cíclico (amostras). 64 = N_FFT/4, a escolha clássica.
#   O CP absorve multipercurso / folga de temporização sem comer vazão.
N_FFT: int = 256
N_CP: int = 64

# Índices das subportadoras usadas dentro da banda. BIN = freq * N_FFT / FS.
# Para F_MIN=6000, FS=48000, N_FFT=256: BIN_MIN = 32 (exato).
# BIN_MAX = BIN_MIN + 21 -> 22 bins no total = 1 piloto + 21 portadoras de dados.
BIN_MIN: int = int(F_MIN * N_FFT / FS)            # 32
BIN_MAX: int = BIN_MIN + 21                       # 53
PILOT_BIN: int = BIN_MIN                          # bin 32 (o mais baixo = piloto)
DATA_BINS: list[int] = list(range(BIN_MIN + 1, BIN_MAX + 1))  # 33..53 inclusive

# QPSK: 2 bits por símbolo complexo. Cada frame OFDM carrega 21 símbolos.
BITS_PER_SYMBOL: int = 2
BITS_PER_OFDM_FRAME: int = len(DATA_BINS) * BITS_PER_SYMBOL   # 42 bits

# Valor do símbolo piloto. Real, amplitude unitária — mantém a estimativa de canal
# numericamente simples: H = X[PILOT_BIN] / PILOT_VALUE.
PILOT_VALUE: complex = complex(1.0, 0.0)

# =============================================================================
# Preâmbulo (chirp usado para sincronização entre transmissor e receptor)
# =============================================================================
# Chirp linear cobrindo a banda de dados. 50 ms é longo o bastante para um pico de
# correlação nítido; rampas de 5 ms em cada ponta evitam espalhamento espectral.
PREAMBLE_DURATION_S: float = 0.050
PREAMBLE_F_START: int = F_MIN
PREAMBLE_F_END: int = F_MAX
PREAMBLE_RAMP_S: float = 0.005

# =============================================================================
# AES-256-CTR (a camada criptográfica de verdade — fornece confidencialidade)
# =============================================================================
AES_KEY_SIZE_BYTES: int = 32       # AES-256
AES_NONCE_SIZE_BYTES: int = 16     # convenção do CTR

# =============================================================================
# Cabeçalho do frame (layout binário dos metadados que precedem o ciphertext)
# =============================================================================
# Layout: [nonce 16B][sample_rate 4B][num_samples 4B][ciphertext_length 4B]
# formato struct: '<16sIII'  -> little-endian, 16 bytes + 3 uint32
HEADER_SIZE_BYTES: int = AES_NONCE_SIZE_BYTES + 4 + 4 + 4  # 28 bytes

# =============================================================================
# Embaralhamento espectral (ofuscação, NÃO criptografia)
# =============================================================================
# Uma permutação das atribuições de subportadora OFDM controlada por chave. A
# confidencialidade real vem do AES; esta camada só demonstra que dá para manipular o
# domínio da frequência com uma chave. É vulnerável a ataques de known-plaintext e
# estatísticos — ver core/modulation/scrambler.py.
SCRAMBLE_SEED_TAG: bytes = b"scramble-v1"

# =============================================================================
# I/O de WAV
# =============================================================================
# Padding de silêncio no início/fim de um WAV de transmissão. Ajuda o receptor a se
# estabilizar antes do chirp chegar e evita clipar o final.
PRE_SILENCE_S: float = 0.050
POST_SILENCE_S: float = 0.050

# Convenções de int16. Multiplicar por INT16_MAX (32767), e não 32768, evita produzir
# +32768, que daria a volta para -32768 no int16 (complemento de dois).
INT16_MAX: int = 32_767
INT16_HEADROOM: float = 0.9   # normaliza o pico para 90% da faixa do int16

# =============================================================================
# Asserções de sanidade (pegam erros de digitação que corromperiam a pipeline)
# =============================================================================
assert BIN_MAX < N_FFT // 2, "Data bins must stay below Nyquist (N_FFT/2)."
assert F_MAX <= FS // 2, "F_MAX exceeds Nyquist (FS/2)."
assert BIN_MIN > 0, "PILOT_BIN must be strictly positive (bin 0 = DC)."
assert len(DATA_BINS) * BITS_PER_SYMBOL == BITS_PER_OFDM_FRAME
assert HEADER_SIZE_BYTES == AES_NONCE_SIZE_BYTES + 12
