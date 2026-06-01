# FFT-LAB

Laboratorio didatico de Transformada de Fourier - exploracao visual + criptografia de voz com OFDM acustico.

O app e dividido em abas que demonstram a Transformada de Fourier em varios papeis:

| Aba | Papel do Fourier |
|---|---|
| FFT Lab | Analise (FFT de sinais arbitrarios) |
| Criptografia 2D | Analise + manipulacao de imagens no dominio da frequencia |
| Voz Criptografada | AES-256-CTR puro (nao e Fourier, mas serve de contraste com criptografia real) |
| **OFDM Transmissao** | **Geracao (IFFT) + analise (FFT) + manipulacao (scrambling) - o coracao do projeto** |
| Sintetizador | Sintese (Fourier inverso aplicado para geracao) |
| Teoria / Referencias | Fundamento matematico |

## Pipeline OFDM (aba "OFDM Transmissao")

```
                 +-----------------+
   voz (16 kHz)  | SpectralDenoiser| (opcional - STFT/ISTFT)
        --------->                 |
                 +--------+--------+
                          | int16
                 +--------v--------+
                 |   WavIO (int16  |
                 |   -> bytes LE)  |
                 +--------+--------+
                          | plaintext bytes
                 +--------v--------+
                 |    AESCipher    |  (AES-256-CTR)
                 |  encrypt(...)   |
                 +--------+--------+
                          | (nonce, ciphertext)
                 +--------v--------+
                 |     Framer      |  header 28 B
                 |     .build      |  + ciphertext
                 +--------+--------+
                          | payload bytes
                 +--------v--------+
                 |   Scrambler     |  permutacao 21
                 |   (a partir da  |  a partir da senha
                 |   senha)        |  (SHA-256 + tag)
                 +--------+--------+
                          | permutation
                 +--------v--------+
                 |   OFDMModem     |  IFFT N=256
                 | bytes_to_signal |  + pilot @ bin 32
                 +--------+--------+  + cyclic prefix 64
                          | OFDM signal @ 48 kHz
                 +--------v--------+
                 |   PreambleSync  |  chirp 6->10 kHz, 50 ms
                 +--------+--------+
                          | chirp + OFDM signal
                 +--------v--------+
                 |      Sink:      |
                 |  WavIO.write    |  (loopback testing) -> .wav
                 |       OR        |
                 |  AudioPlayer.   |  (over-air)         -> alto-falante
                 |     play        |
                 +-----------------+

                 [ ar / cabo digital ]

                 +-----------------+
                 |   Source:       |
                 |  WavIO.read     |  <- .wav
                 |       OR        |
                 |  AudioRecorder. |  <- microfone
                 |     record      |
                 +--------+--------+
                          | captured signal @ 48 kHz
                 +--------v--------+
                 |  PreambleSync   |  cross-corr -> indice OFDM
                 |     .detect     |
                 +--------+--------+
                          | signal[ofdm_start:]
                 +--------v--------+
                 |   OFDMModem     |  FFT N=256
                 | signal_to_bytes |  + H = X[pilot]/PILOT
                 +--------+--------+  + bins/H -> unscramble
                          | payload bytes
                 +--------v--------+
                 |     Framer      |  parse header
                 |  parse_header   |  -> ciphertext_length, nonce, ...
                 +--------+--------+
                          | ciphertext + header
                 +--------v--------+
                 |   AESCipher     |
                 |   .decrypt      |  AES-256-CTR
                 +--------+--------+
                          | plaintext bytes
                 +--------v--------+
                 |   WavIO.bytes_  |  -> int16
                 |   to_int16      |
                 +-----------------+
                          |
                          v
                  voz recuperada
```

## Onde mora o Fourier neste projeto

1. **IFFT em `OFDMModem.modulate_frame`** - sintetiza o sinal de audio multiportadora a partir de bins de frequencia. Aqui o Fourier e **gerador de sinal**.
2. **FFT em `OFDMModem.demodulate_frame`** - extrai os simbolos dos bins de frequencia. Aqui o Fourier e **analisador**.
3. **Permutacao em `Scrambler`** - manipulacao direta de bins no dominio da frequencia controlada por chave. Aqui o Fourier e **substrato manipulavel**.
4. **STFT + spectral subtraction em `SpectralDenoiser`** - analise por janelas, modificacao seletiva de magnitudes, reconstrucao por ISTFT. Aqui o Fourier e **filtro**.
5. **Cross-correlation no `PreambleSync`** - implementada via FFT/IFFT (`scipy.signal.correlate method="fft"`). Aqui o Fourier e **acelerador de convolucao**.

## Implementacao da Transformada de Fourier do zero

Alem de usar `numpy.fft` no pipeline de producao (otimizado em C/SIMD), implementamos a Transformada de Fourier **do zero** em `core/modulation/fft_from_scratch.py`, com APENAS a biblioteca padrao do Python (`cmath`, `math`). Tres algoritmos sao fornecidos, em ordem crescente de sofisticacao:

| Funcao | Algoritmo | Complexidade |
|---|---|---|
| `dft_naive(x)` / `idft_naive(X)` | Formula direta: `X[k] = SUM_n x[n] * exp(-2*pi*j*k*n/N)` | O(N^2) |
| `fft_recursive(x)` / `ifft_recursive(X)` | Cooley-Tukey radix-2 recursivo (divide-and-conquer + butterfly) | O(N log N) |
| `fft_iterative(x)` / `ifft_iterative(X)` | Mesma FFT, in-place com permutacao bit-reverse explicita | O(N log N) |

### Verificacao matematica

O modulo inclui um teste automatico que compara as seis implementacoes com `numpy.fft` em entradas aleatorias complexas:

```
python -m core.modulation.fft_from_scratch
```

Saida esperada (N=256, tol=1e-9):

```
dft_naive (forward)         diff=5.3e-12   [PASS]
fft_recursive (forward)     diff=2.2e-14   [PASS]
fft_iterative (forward)     diff=1.2e-13   [PASS]
idft_naive (inverse)        diff=1.9e-13   [PASS]
ifft_recursive (inverse)    diff=1.9e-15   [PASS]
ifft_iterative (inverse)    diff=1.1e-14   [PASS]
roundtrip recursive         diff=9.9e-16   [PASS]
roundtrip iterative         diff=1.6e-14   [PASS]
```

As diferencas na ordem de `1e-12` a `1e-16` sao puramente ruido de arredondamento de ponto flutuante - **a matematica e identica** a do `numpy.fft`.

### Comparacao de performance (N=256, media sobre 100 chamadas)

| Implementacao | Tempo / chamada | Vs numpy |
|---|---|---|
| `dft_naive` (O(N^2)) | 41.19 ms | 1583x mais lento |
| `fft_recursive` (O(N log N)) | 1.97 ms | 76x |
| `fft_iterative` (O(N log N), in-place) | 1.09 ms | 42x |
| `numpy.fft.fft` (C/SIMD/BLAS) | 0.03 ms | 1x (baseline) |

A diferenca de 41 ms -> 2 ms entre `dft_naive` e `fft_iterative` valida o ganho **algoritmico** do Cooley-Tukey: `O(N log N)` vs `O(N^2)` corresponde a ~32x menos multiplicacoes para N=256 (256x8 = 2048 vs 256^2 = 65 536). O salto de 2 ms -> 0.03 ms entre nossa FFT em Python e a do numpy e **puramente custo de interpretador vs codigo nativo** - o algoritmo e o mesmo.

### Por que o pipeline usa `numpy.fft`

Cada transmissao OFDM produz milhares de IFFTs (uma por frame). Substituir `numpy.fft` pela nossa implementacao em Python multiplicaria o tempo de transmissao por ~40x. A escolha de usar `numpy.fft` no pipeline e puramente de engenharia - a **compreensao matematica completa** esta documentada e verificada em `fft_from_scratch.py`.

## Parametros principais (banda 6-10 kHz)

| Parametro | Valor | Comentario |
|---|---|---|
| `FS` | 48 000 Hz | Taxa do sinal de transmissao (placa de som padrao). |
| `F_MIN`, `F_MAX` | 6 000, 10 000 Hz | Banda OFDM. Audivel, mas amigavel a mic/speaker baratos. |
| `N_FFT` | 256 | Tamanho de cada bloco IFFT/FFT. |
| `N_CP` | 64 | Prefixo ciclico (~ N_FFT/4). |
| `BIN_MIN`, `BIN_MAX` | 32, 53 | 22 bins: 1 piloto + 21 dados. |
| `BITS_PER_OFDM_FRAME` | 42 | 21 portadoras x 2 bits (QPSK). |
| `AUDIO_RECORD_SAMPLE_RATE` | 16 000 Hz | Taxa da voz original. |
| `AES_KEY_SIZE_BYTES` | 32 | AES-256. |
| `PREAMBLE_DURATION_S` | 0.050 | Chirp linear 6->10 kHz. |

A versao original da spec usava **18-22 kHz (ultrassonico)**; reduzimos para **6-10 kHz (audivel)** porque microfones de eletreto e alto-falantes baratos tem resposta ruim acima de ~15 kHz. A consequencia: a transmissao sai como um chiado tipo modem - barulhento mas confiavel.

## Setup

```
# 1) venv (recomendado)
python -m venv .venv
.venv\Scripts\activate

# 2) dependencias
pip install -r requirements.txt

# 3) executar
python main.py
```

Requisitos do sistema:

- **Windows / Linux / macOS** com Python 3.10+
- `sounddevice` precisa de PortAudio (ja bundled no Windows; Linux: `sudo apt install libportaudio2`)
- Microfone (integrado do notebook serve) e alto-falante para o modo over-air

## Uso tipico - over-air entre dois PCs

1. Em **ambas** as maquinas: abra o app, va na aba **OFDM Transmissao**, digite a **mesma senha**.
2. No PC **transmissor**:
   - Grave uma voz curta (3-5 s).
   - Clique em **Transmitir via alto-falante**.
3. No PC **receptor** (apontando o microfone para o alto-falante do transmissor):
   - Antes do TX comecar, ajuste **Duracao de captura** para algo como o dobro do TX.
   - Clique em **Receber via microfone** imediatamente antes do transmissor disparar.
   - Quando terminar: **Tocar voz recuperada**.

Para um teste de sanidade sem hardware (loopback puro): use **Salvar WAV de transmissao** em um PC e **Receber de WAV** no mesmo.

## Estrutura do codigo

```
FFT-LAB/
+-- global_configs.py            # constantes unicas (banda, OFDM, AES, denoise...)
+-- main.py                      # entry point CTk
+-- core/
|   +-- voice_crypto.py          # AES legacy (aba "Voz Criptografada")
|   +-- audio_generator.py       # aba "Sintetizador"
|   +-- image_crypto.py          # aba "Criptografia 2D"
|   +-- crypto/
|   |   +-- cipher.py            # class AESCipher
|   |   +-- framer.py            # class Framer + FrameHeader
|   +-- modulation/
|   |   +-- qpsk.py              # class QPSKModem
|   |   +-- scrambler.py         # class Scrambler
|   |   +-- ofdm.py              # class OFDMModem  <- coracao do Fourier
|   |   +-- sync.py              # class PreambleSync (chirp)
|   |   +-- fft_from_scratch.py  # DFT/FFT do zero, sem numpy.fft
|   +-- channel/
|   |   +-- awgn.py              # class AWGNChannel (so p/ testes BER vs SNR)
|   +-- audio/
|   |   +-- wav_io.py            # class WavIO
|   |   +-- recorder.py          # class AudioRecorder
|   |   +-- player.py            # class AudioPlayer
|   |   +-- denoiser.py          # class SpectralDenoiser
|   +-- pipeline/
|       +-- transmitter.py       # class Transmitter (orquestra TX)
|       +-- receiver.py          # class Receiver    (orquestra RX)
+-- ui/
    +-- main_window.py           # navbar + roteamento de abas
    +-- views/
        +-- voice_crypto_view.py        # AES (preservado)
        +-- ofdm_transmission_view.py   # NOVA: TX/RX over-air + spectrograma
        +-- ... (outras abas)
```

## Principios de design

- **Uma classe por responsabilidade.** `AESCipher` so cifra, `OFDMModem` so modula, `Framer` so serializa. A composicao mora em `Transmitter` e `Receiver`.
- **Sem estado global escondido.** Cada classe recebe suas dependencias via construtor; o `global_configs` e puramente declarativo (constantes), nunca mutado.
- **Type hints em todas as APIs publicas.** Python 3.10+ syntax (`X | None`).
- **Docstrings explicam o porque, nao so o que.** O *que* fica claro no codigo; o *porque* e o que merece comentario.
- **Threads para operacoes longas na UI.** Gravacao, transmissao e recepcao rodam em background; o main loop do Tk nao trava.

## Limitacoes (importantes para o relatorio)

1. **Scrambling NAO e criptografia.** E obfuscacao no dominio da frequencia. A confidencialidade real vem do AES-256-CTR.
2. **CTR nao tem verificacao de integridade.** Senha errada produz ruido silenciosamente - sem excecao. Producao exigiria AES-GCM ou HMAC.
3. **Denoise assume silencio nos primeiros ~80 ms** da gravacao para estimar o piso de ruido. Comece a falar depois de uma breve pausa.
4. **Banda audivel.** A transmissao e ouvida como um chiado por todos no ambiente. Nao e stealth.
5. **Expansion ratio alto.** Uma voz de 2 s vira ~80 s de WAV transmitido (overhead de framing + OFDM + chirp).
6. **Multi-path real degrada a recepcao.** Reverberacao em salas vivas causa erros de bit. Salas mais mortas funcionam melhor. O cyclic prefix absorve parte disso.

## Principio de Kerckhoffs

O sistema e seguro mesmo que o atacante saiba **tudo** sobre o algoritmo - exceto a senha. A localizacao do chirp, a estrutura OFDM, os indices de bins, a derivacao SHA-256 - tudo isso e publico. A confidencialidade depende **so** do AES-256.

---

## Creditos

Projeto academico de Fisica - demonstracao da Transformada de Fourier em um sistema de comunicacao real.
