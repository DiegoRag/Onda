# Laboratório de Processamento Digital de Sinais (FFT-LAB)
## Documentação Técnica Avançada e Fundamentos Matemáticos

Este documento constitui o compêndio teórico e a documentação técnica integral do projeto **FFT-LAB**. O objetivo é detalhar de forma analítica e rigorosa as equações, os teoremas, as propriedades de transformadas e os pipelines algorítmicos implementados no núcleo matemático (`core/`) do sistema.

---

## Sumário
1. [Fundamentos da Transformada Discreta de Fourier (DFT)](#1-fundamentos-da-transformada-discreta-de-fourier-dft)
2. [Janelamento Temporal e Remoção Espectral de Ruído (Aba FFT Lab)](#2-janelamento-temporal-e-remoção-espectral-de-ruído-aba-fft-lab)
3. [Processamento de Sinais Bidimensionais (Aba Criptografia 2D)](#3-processamento-de-sinais-bidimensionais-aba-criptografia-2d)
4. [Criptografia de Fluxo Baseada em Segurança Simétrica (Voz Criptografada)](#4-criptografia-de-fluxo-baseada-em-segurança-simétrica-voz-criptografada)
5. [Transmissão Acústica Digital Multi-Portadora (Aba OFDM)](#5-transmissão-acústica-digital-multi-portadora-aba-ofdm)
6. [Modelagem Teórica do Sintetizador (Aba Sintetizador)](#6-modelagem-teórica-do-sintetizador-aba-sintetizador)

---

## 1. Fundamentos da Transformada Discreta de Fourier (DFT)

A Transformada Discreta de Fourier (DFT) mapeia uma sequência finita de amostras de um sinal discretizado no domínio do tempo para os seus componentes harmônicos equivalentes no domínio da frequência.

### 1.1 Formulação Analítica
Para um sinal discreto $x[n]$ contendo $N$ amostras uniformemente espaçadas no tempo, a Transformada Direta (DFT) é definida matematicamente por:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} k n}$$

Onde:
* $n$ é o índice temporal discreto ($0 \le n < N$).
* $k$ é o índice de frequência discreto ou *bin* espectral ($0 \le k < N$).
* $j = \sqrt{-1}$ representa a unidade imaginária.
* $e^{-j \frac{2\pi}{N} k n} = \cos\left(\frac{2\pi kn}{N}\right) - j \sin\left(\frac{2\pi kn}{N}\right)$ (Identidade de Euler).

Para reverter o espectro complexo $X[k]$ de volta ao domínio do tempo, emprega-se a Transformada Inversa Discreta de Fourier (IDFT):

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{+j \frac{2\pi}{N} k n}$$

### 1.2 Propriedades Teóricas Aplicadas
O comportamento do ecossistema algorítmico do **FFT-LAB** ancora-se em três propriedades fundamentais de DSP:

1. **Linearidade:** Essencial para a mistura de áudio e superposição de sinais:
   $$\text{DFT}(\alpha \cdot x[n] + \beta \cdot y[n]) = \alpha \cdot X[k] + \beta \cdot Y[k]$$

2. **Conservação de Energia (Teorema de Parseval):** Garante a invariância da energia total do sinal em ambos os domínios:
   $$\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2$$

3. **Simetria Conjugada:** Para qualquer sinal real ($x[n] \in \mathbb{R}$), o espectro de frequências exibe simetria em relação ao ponto médio (frequência de Nyquist $N/2$):
   $$X[N - k] = X^*[k]$$
   Onde $*$ denota o operador complexo conjugado ($a + bj \rightarrow a - bj$).

### 1.3 Otimização Computacional: Algoritmo FFT Radix-2
A computação direta da fórmula da DFT via laços iterativos convencionais exige uma complexidade assintótica de $\mathcal{O}(N^2)$ multiplicações complexas, tornando inviável o processamento de sinais em tempo real.

O algoritmo *Fast Fourier Transform* (FFT) de Cooley-Tukey (implementado em `core/modulation/fft_from_scratch.py`) adota a estratégia de divisão e conquista por decimação no tempo. Supondo que $N$ seja uma potência de 2, a sequência $x[n]$ é separada em seus índices pares ($n=2m$) e ímpares ($n=2m+1$):

$$X[k] = \sum_{m=0}^{N/2-1} x[2m] \cdot e^{-j \frac{2\pi}{N/2} k m} + e^{-j \frac{2\pi}{N} k} \sum_{m=0}^{N/2-1} x[2m+1] \cdot e^{-j \frac{2\pi}{N/2} k m}$$

$$X[k] = E[k] + W_N^k \cdot O[k]$$

Aproveitando-se da periodicidade periódica do fator de rotação (*Twiddle Factor*) $W_N^k = e^{-j \frac{2\pi}{N} k}$, onde $W_N^{k + N/2} = -W_N^k$, o cálculo das borboletas (*butterflies*) estruturais reduz a complexidade para:

$$\mathcal{O}(N \log_2 N)$$

| Tamanho do Bloco ($N$) | Operações Diretas ($\mathcal{O}(N^2)$) | Operações Otimizadas ($\mathcal{O}(N \log_2 N)$) | Ganho de Eficiência |
| :--- | :--- | :--- | :--- |
| 256 | 65.536 | ~2.048 | ~32 vezes mais rápido |
| 1024 | 1.048.576 | ~10.240 | ~102 vezes mais rápido |
| 4096 | 16.777.216 | ~49.152 | ~341 vezes mais rápido |

---

## 2. Janelamento Temporal e Remoção Espectral de Ruído (Aba FFT Lab)

Sinais reais de voz humana são intrinsecamente não-estacionários, o que significa que suas características estatísticas e harmônicas variam continuamente ao longo do tempo. Analisar um bloco longo de áudio por meio de uma única FFT causaria a perda da resolução temporal.

### 2.1 Short-Time Fourier Transform (STFT)
Para capturar a evolução dinâmica temporal do espectro, aplica-se a Transformada de Fourier de Tempo Curto (STFT), segmentando o sinal através de uma função de janela móvel $w[n]$ que desliza com um salto fixo (*Hop Size*) $H$:

$$X(m, k) = \sum_{n=0}^{N-1} x[n + m \cdot H] \cdot w[n] \cdot e^{-j \frac{2\pi}{N} k n}$$

Onde $m$ indexa a janela temporal corrente e $k$ mapeia as subportadoras harmônicas daquele trecho. No projeto, o pipeline de áudio utiliza tamanho de janela $N = 1024$ e salto $H = 256$, estabelecendo uma sobreposição (*overlap*) de 75% para mitigar distorções de descontinuidade de bordas.

### 2.2 Janela de Hann
A truncagem abrupta de um sinal gera descontinuidades artificiais em suas extremidades, provocando o fenômeno de espalhamento de energia conhecido como **Vazamento Espectral** (*Spectral Leakage*). Para suprimir esse efeito, multiplica-se o bloco pela função de janela de Hann:

$$w[n] = 0.5 \cdot \left( 1 - \cos\left( \frac{2\pi n}{N-1} \right) \right), \quad 0 \le n \le N-1$$

A janela de Hann atenua suavemente as amostras próximas às bordas em direção a zero, restringindo os lobos secundários do espectro e blindando a precisão do diagnóstico de bins adjacentes.

### 2.3 Algoritmo de Subtração Espectral de Boll (1979)
A remoção de ruídos estacionários de fundo baseia-se no princípio clássico de Boll. Assumindo que o ruído ambiental é aditivo e sua magnitude estatística altera-se lentamente, estima-se a magnitude média do ruído $|N(f)|$ calculando a média espectral dos primeiros 5 blocos capturados (zona de silêncio obrigatório do laboratório):

$$|N(f)| = \frac{1}{5} \sum_{m=0}^{4} |Y(m, f)|$$

Para os quadros subsequentes contendo o sinal corrompido $Y(m, f)$, modifica-se exclusivamente a magnitude matemática, mantendo a fase original intacta para a reconstrução:

$$|S(m, f)| = \max\left( |Y(m, f)| - \alpha \cdot |N(f)| \ , \ \beta \cdot |Y(m, f)| \right)$$

Onde os parâmetros operacionais são definidos no arquivo `global_configs.py`:
* $\alpha = 2.0$ (**Fator de Sobresubtração**): Expande a penalização matemática do ruído para eliminar resíduos agressivos.
* $\beta = 0.05$ (**Piso Espectral / Spectral Floor**): Impede que a magnitude caia para zero absoluto. Sem este piso, bins isolados flutuantes sobreviveriam aleatoriamente à subtração, gerando tons espúrios agudos conhecidos como **Ruído Musical** (*musical noise*).

Finalmente, recombina-se a fase original de curto termo $\angle Y(m, f)$ para ressintetizar o sinal limpo através da Transformada Inversa (ISTFT):

$$S(m, f) = |S(m, f)| \cdot e^{j \cdot \angle Y(m, f)}$$

---

## 3. Processamento de Sinais Bidimensionais (Aba Criptografia 2D)

Imagens digitais monocromáticas representam sinais discretos mapeados em um domínio espacial bidimensional discreto $f(x,y)$, onde cada coordenada armazena a intensidade de luminância de um pixel.

### 3.1 Formulação Analítica da DFT 2D
A transformada bidimensional estende os conceitos harmônicos para extrair frequências espaciais senoidais. Para uma imagem com dimensões $M \times N$:

$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}$$

Devido à propriedade de **Separabilidade** da transformada de Fourier, o cálculo 2D pode ser decomposto aplicando-se a FFT 1D clássica ao longo de cada linha, e subsequentemente nas colunas da matriz resultante.

A Transformada Inversa (IDFT 2D) recupera a imagem espacial sem perdas matemáticas:

$$f(x,y) = \frac{1}{M \cdot N} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u,v) \cdot e^{+j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}$$

### 3.2 Deslocamento DC (fftshift) e Faixa Dinâmica Logarítmica
A operação `fftshift` translada o componente DC (frequência zero) posicionado no índice (0,0) para o centro geométrico da tela: $(u_{center}, v_{center}) = (M/2, N/2)$, facilitando a interpretação radial do espectro.

A faixa de magnitude das frequências de uma imagem é excessivamente ampla. Para permitir a exibição sem saturação visual, comprime-se a faixa dinâmica logaritmicamente:

$$\text{Magnitude (dB)} = 20 \cdot \log_{10}\left( |F(u,v)| + 1.0 \right)$$

O fator $+1.0$ atua como uma barreira de segurança matemática, impedindo a ocorrência de $\log_{10}(0) = -\infty$.

---

## 4. Criptografia de Fluxo Baseada em Segurança Simétrica (Voz Criptografada)

A segurança real dos dados não depende do embaralhamento espectral, mas do padrão **AES-256** (*Advanced Encryption Standard*). O projeto adota o modo **CTR (Counter Mode)**, transformando o cifrador de bloco em uma **Cifra de Fluxo** (*Stream Cipher*).

A geração do fluxo pseudoaleatório de mascaramento (*Keystream*) é dada por:

$$\text{Keystream}_i = \text{AES}_K(\text{Nonce} \parallel \text{Contador}_i)$$

Onde $\parallel$ denota a concatenação. A encriptação ocorre via OU Exclusivo ($XOR$):

$$\text{Ciphertext}_i = \text{Plaintext}_i \oplus \text{Keystream}_i$$

A vantagem acústica do modo CTR sobre o modo CBC é a **não-propagação de erros**. Se o ruído acústico corromper um bit no ar, o receptor decifrará apenas um bit incorreto, manifestando-se como um micro-estalo acústico, mas preservando totalmente a fonética da voz humana.

---

## 5. Transmissão Acústica Digital Multi-Portadora (Aba OFDM)

A modulação OFDM (*Orthogonal Frequency Division Multiplexing*) é o cerne da transmissão de dados via ondas sonoras.

### 5.1 Codificação QPSK
Os dados binários são convertidos em símbolos complexos QPSK com mapeamento de Gray:

$$s = \frac{(1 - 2b_0) + j(1 - 2b_1)}{\sqrt{2}}$$

### 5.2 Síntese do Sinal no Domínio do Tempo (IFFT)
A IFFT atua como o próprio modulador, alocando os símbolos nas subportadoras simultaneamente:

$$x[n] = \frac{1}{N_{\text{FFT}}} \sum_{k=0}^{N_{\text{FFT}}-1} X[k] \cdot e^{+j \frac{2\pi}{N_{\text{FFT}}} k n}$$

Para forçar $x[n]$ a ser puramente real para envio ao alto-falante, impõe-se simetria Hermitiana:

$$X[N_{\text{FFT}} - k] = X^*[k]$$

### 5.3 Prefixo Cíclico e Equalização
Insere-se um Prefixo Cíclico de 64 amostras copiando a cauda do pacote para absorver eco e reverbação. No receptor, o canal é equalizado estimando a distorção através de uma subportadora Piloto conhecida:

$$H_{est} = \frac{Y[BIN_{\text{PILOTO}}]}{S_{pilot}} = Y[BIN_{\text{PILOTO}}]$$

$$\hat{S}_i = \frac{Y[BIN_{\text{DADOS}}[i]]}{H_{est}}$$

### 5.4 Sincronização por Correlação Cruzada de Chirp
A detecção do início exato do pacote no microfone é feita buscando o pico de correlação cruzada com um *Chirp* linear (varredura contínua de frequência):

$$\phi(t) = 2\pi \cdot \left( f_{start} \cdot t + \frac{f_{end} - f_{start}}{2T} \cdot t^2 \right)$$

$$c(t) = \cos(\phi(t))$$

---

## 6. Modelagem Teórica do Sintetizador (Aba Sintetizador)

O sintetizador gera sinais analíticos puramente matemáticos no tempo contínuo $t$, comprovando empiricamente as formulações teóricas da Síntese Aditiva.

$$y[n] = A \cdot \text{forma\_onda}\left( 2\pi \cdot f \cdot \frac{n}{Fs} + \varphi \right)$$

O software também permite modelagem em **Modulação de Amplitude (AM)**, onde a superposição das demais ondas atua como o envelope de uma onda portadora principal $c(t)$:

$$y_{\text{AM}}[n] = \left[ 1.0 + m[n] \right] \cdot c[n]$$

---