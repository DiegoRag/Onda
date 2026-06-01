import customtkinter as ctk

class TheoryView(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        # Cabeçalho Principal da Documentação
        self.title_label = ctk.CTkLabel(
            self,
            text="Documentação Técnica e Fundamentos Matemáticos",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(anchor="w", padx=20, pady=(20, 10))

        # Container Centralizado (Garante legibilidade travando a largura máxima)
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=(60, 60), pady=10)

        # Configurações de Tipografia e Layout de Leitura
        self.font_section = ctk.CTkFont(size=18, weight="bold")
        self.font_subsection = ctk.CTkFont(size=14, weight="bold")
        self.font_body = ctk.CTkFont(size=13, weight="normal")
        self.font_math = ctk.CTkFont(family="Courier", size=13, weight="bold")

        self.text_wrap = 780
        self.color_sec1 = ("#006B8F", "#06b6d4") # Ciano adaptativo
        self.color_sec2 = ("#86198F", "#c026d3") # Magenta adaptativo
        self.color_math_bg = ("#E5E7EB", "#1A1A1B")
        self.color_math_text = ("#111827", "#10b981") # Verde terminal no escuro

        # Cores de texto corrigidas
        self.text_color = ("#333333", "#CCCCCC")
        self.caption_color = ("#555555", "#888888")

        # =========================================================================
        # SEÇÃO 1: FUNDAMENTOS DA TRANSFORMADA DE FOURIER DISCRETA (DFT)
        # =========================================================================
        math_dft = (
            "  Direta:\n"
            "  X[k] = ∑_{n=0}^{N-1} x[n] · e^(-j · 2π · k · n / N)\n\n"
            "  Inversa (IDFT):\n"
            "  x[n] = (1 / N) · ∑_{k=0}^{N-1} X[k] · e^(+j · 2π · k · n / N)"
        )
        math_properties = (
            "  1. Linearidade: DFT(a · x[n] + b · y[n]) = a · X[k] + b · Y[k]\n"
            "  2. Conservação de Energia (Teorema de Parseval):\n"
            "     ∑_{n=0}^{N-1} |x[n]|² = (1 / N) · ∑_{k=0}^{N-1} |X[k]|²\n"
            "  3. Simetria Conjugada (para x[n] real):\n"
            "     X[N - k] = X*[k]  (onde * denota o complexo conjugado)"
        )
        math_cooley = (
            "  Separação em índices Pares (Even) e Ímpares (Odd):\n"
            "  X[k] = E[k] + w^k · O[k]\n"
            "  X[k + N/2] = E[k] - w^k · O[k]\n\n"
            "  Onde o fator de rotação (Twiddle Factor) é:\n"
            "  w = e^(-j · 2π / N)"
        )

        self._add_heading("1. Fundamentos da Transformada Discreta de Fourier (DFT)", self.color_sec1)
        self._add_body(
            "A Transformada Discreta de Fourier (DFT) mapeia um sinal do domínio discreto do tempo para "
            "o domínio discreto da frequência, permitindo a análise espectral de sequências finitas. "
            "Abaixo são descritas as formulações analíticas fundamentais utilizadas nesta aplicação:"
        )
        self._add_math_block(math_dft)

        self._add_subheading("Propriedades Matemáticas Aplicadas")
        self._add_body(
            "O comportamento algorítmico do software apoia-se em teoremas fundamentais do processamento digital de sinais (DSP):"
        )
        self._add_math_block(math_properties)

        self._add_subheading("Otimização Computacional: Algoritmo FFT Radix-2")
        self._add_body(
            "A computação direta da DFT exige uma complexidade de O(N²) operações. O algoritmo Fast Fourier Transform (FFT) "
            "de Cooley-Tukey (radix-2) aproveita-se da periodicidade e simetria dos fatores de rotação, dividindo a sequência "
            "recursivamente e reduzindo o custo computacional para O(N log N):"
        )
        self._add_math_block(math_cooley)

        # =========================================================================
        # SEÇÃO 2: ABA FFT LAB & FILTRAGEM ESPETRAL (DENOISING BOLL 1979)
        # =========================================================================
        math_stft = (
            "  Short-Time Fourier Transform (STFT):\n"
            "  X(m, k) = ∑_{n=0}^{N-1} x[n + m · H] · w[n] · e^(-j · 2π · k · n / N)\n\n"
            "  Onde 'w[n]' é a Janela de Hann e 'H' representa o tamanho do salto (Hop Size)."
        )
        math_boll = (
            "  Subtração Espectral com Piso de Ruído (Boll, 1979):\n"
            "  |S(f)| = max( |Y(f)| - α · |N(f)| ,  β · |Y(f)| )\n\n"
            "  Recombinação da Fase Original para Síntese via ISTFT:\n"
            "  S(f) = |S(f)| · e^(j · ∠Y(f))\n\n"
            "  Parâmetros do Sistema:\n"
            "  - α = 2.0 (Fator de Sobresubtração / Oversubtraction)\n"
            "  - β = 0.05 (Piso Espectral contra Ruído Musical / Spectral Floor)"
        )

        self._add_heading("2. Janelamento Temporal e Remoção Espectral de Ruído (Aba FFT Lab)", self.color_sec2)
        self._add_body(
            "Sinais reais de áudio e voz são não-estacionários (suas propriedades espectrais variam no tempo). Para contornar isso, "
            "o laboratório segmenta o sinal em blocos curtos e sobrepostos (STFT) utilizando uma Janela de Hann para atenuar o vazamento espectral:"
        )
        self._add_math_block(math_stft)

        self._add_subheading("Algoritmo de Subtração Espectral de Boll")
        self._add_body(
            "O software implementa a filtragem de ruído em ambiente ruidoso estimando a magnitude média do ruído |N(f)| "
            "nos primeiros 5 frames de silêncio do buffer capturado, aplicando a filtragem direta na magnitude complexa:"
        )
        self._add_math_block(math_boll)

        # =========================================================================
        # SEÇÃO 3: ABA CRIPTOGRAFIA 2D (PROCESSAMENTO DE IMAGEM)
        # =========================================================================
        math_2d = (
            "  DFT 2D Direta (Imagem M x N):\n"
            "  F(u,v) = ∑_{x=0}^{M-1} ∑_{y=0}^{N-1} f(x,y) · e^(-j · 2π · ( (u·x/M) + (v·y/N) ))\n\n"
            "  Compressão Espacial de Faixa Dinâmica (Escala Log):\n"
            "  Magnitude_dB = 20 · log_10( |F(u,v)| + 1.0 )"
        )

        self._add_heading("3. Processamento de Sinais Bidimensionais (Aba Criptografia 2D)", self.color_sec1)
        self._add_body(
            "Imagens digitais representam sinais discretos no domínio espacial 2D f(x,y). A Transformada de Fourier de duas "
            "dimensões decompõe a imagem em componentes senoidais espaciais. Devido à separabilidade da equação, aplica-se a "
            "FFT 1D unidimensional nas linhas e, subsequentemente, nas colunas do resultado:"
        )
        self._add_math_block(math_2d)
        self._add_body(
            "O componente DC (frequência zero) é transladado para as coordenadas centrais (M/2, N/2) através da função "
            "'fftshift', alinhando a visualização conforme os padrões científicos. A escala logarítmica é mandatória porque "
            "as baixas frequências concentram magnitudes ordens de grandeza superiores às altas frequências (detalhes/bordas)."
        )

        # =========================================================================
        # SEÇÃO 4: VOZ CRIPTOGRAFADA
        # =========================================================================
        self._add_heading("4. Criptografia de Voz (AES-256-CTR)", self.color_sec2)
        self._add_body(
            "A segurança real do projeto não vem da manipulação de frequências, mas do padrão AES de 256 bits. "
            "Utilizamos o Counter Mode (CTR) por atuar como uma cifra de fluxo (stream cipher).\n\n"
            "Sua principal vantagem acústica é a não-propagação de erros: se o ruído do ambiente corromper "
            "1 bit durante a transmissão, apenas 1 bit será corrompido no áudio decifrado, permitindo que a "
            "voz humana continue inteligível do outro lado, diferente de modos em bloco como CBC."
        )

        # =========================================================================
        # SEÇÃO 5: ABA TRANSMISSÃO ACÚSTICA MULTIPORTADORA (OFDM)
        # =========================================================================
        math_qpsk = (
            "  Mapeamento de Símbolos QPSK (Constelação de Gray):\n"
            "  s = [ (1 - 2·b_0) + j·(1 - 2·b_1) ] / √2\n\n"
            "  Equação de Conversão de Banda para Bin de Frequência k:\n"
            "  k = ⌊ (f · N_FFT) / Fs ⌋"
        )
        math_ofdm = (
            "  Sinal OFDM no Domínio do Tempo (Saída da IFFT):\n"
            "  x[n] = (1 / N_FFT) · ∑_{k=0}^{N_FFT-1} X[k] · e^(+j · 2π · k · n / N_FFT)\n\n"
            "  Equalização de Canal baseada em Subportadora Piloto:\n"
            "  H_est = Y[BIN_PILOTO] / S_conhecido\n"
            "  S_equalizado[i] = Y[BIN_DADOS[i]] / H_est"
        )
        math_chirp = (
            "  Fase Instantânea do Chirp Linear:\n"
            "  φ(t) = 2π · [ f_start · t + ( (f_end - f_start) / 2T ) · t² ]\n\n"
            "  Sinalizador de Sincronização:\n"
            "  c(t) = cos( φ(t) )"
        )

        self._add_heading("5. Transmissão Acústica Digital Multi-Portadora (Aba OFDM)", self.color_sec1)
        self._add_body(
            "A transmissão de dados pelo ar converte pacotes binários criptografados em símbolos complexos utilizando a "
            "tecnologia OFDM. Nesse cenário, a Transformada Inversa (IFFT) passa a atuar como o próprio gerador/modulador "
            "do sinal físico de transmissão."
        )
        self._add_math_block(math_qpsk)

        self._add_subheading("Síntese OFDM e Equalização Coerente")
        self._add_body(
            "Injeta-se simetria conjugada artificial no espectro para garantir que o sinal seja real. O Prefixo Cíclico (CP) "
            "de 64 amostras absorve reflexos (multipath) no ar:"
        )
        self._add_math_block(math_ofdm)

        self._add_subheading("Sincronização Temporal por Correlação de Chirp")
        self._add_body(
            "O receptor encontra o pacote no áudio contínuo usando correlação cruzada com um Chirp linear (varredura de frequência):"
        )
        self._add_math_block(math_chirp)

        # =========================================================================
        # SEÇÃO 6: ABA SINTETIZADOR DE SINAIS (O "PARQUINHO" FINAL)
        # =========================================================================
        math_synth = (
            "  Equação Fundamental de Onda:\n"
            "  y(t) = A · forma_onda( 2π · f · t + φ )\n\n"
            "  Onde:\n"
            "  - A ∈ [0.0, 1.0] (Amplitude normalizada obtida via UI em %)\n"
            "  - φ = φ_graus · (π / 180) (Conversão de Fase para Radianos)"
        )
        math_am = (
            "  y_AM(t) = [ 1.0 + m(t) ] · c(t)\n\n"
            "  Onde:\n"
            "  - c(t) = A_c · sin(2π · f_c · t + φ_c)  (Onda Portadora / Carrier)\n"
            "  - m(t) = ∑ A_i · forma_i(2π · f_i · t + φ_i) (Sinal Modulador / Envelope)"
        )

        self._add_heading("6. Modelagem Teórica do Sintetizador (Aba Sintetizador)", self.color_sec2)
        self._add_body(
            "Ao final de todo o processo de análise, o sintetizador atua como ferramenta didática baseada na Síntese Aditiva, "
            "provando que qualquer sinal complexo pode ser gerado a partir da soma iterativa de harmônicos simples."
        )
        self._add_math_block(math_synth)

        self._add_subheading("Modulação em Amplitude (AM)")
        self._add_body(
            "O módulo permite ativar a Modulação AM, onde a onda primária atua como portadora, e a soma vetorial das "
            "subsequentes gera o envelope que transporta a informação de baixa frequência:"
        )
        self._add_math_block(math_am)

        # Adiciona uma margem inferior para dar respiro ao scroll
        ctk.CTkLabel(self.content_frame, text="").pack(pady=40)

    # ==========================================
    # COMPONENTES AUXILIARES DE FORMATAÇÃO (UI)
    # ==========================================
    def _add_heading(self, text, color):
        ctk.CTkLabel(
            self.content_frame,
            text=text,
            font=self.font_section,
            text_color=color
        ).pack(anchor="w", pady=(35, 10))

    def _add_subheading(self, text):
        ctk.CTkLabel(
            self.content_frame,
            text=text,
            font=self.font_subsection,
            text_color=("gray20", "gray80")
        ).pack(anchor="w", pady=(20, 5))

    def _add_body(self, text):
        ctk.CTkLabel(
            self.content_frame,
            text=text,
            font=self.font_body,
            justify="left",
            wraplength=self.text_wrap,
            text_color=self.text_color
        ).pack(anchor="w", pady=(0, 10))

    def _add_math_block(self, math_text):
        """Renderiza um bloco de código com fundo escuro simulando console/LaTeX técnico."""
        block = ctk.CTkFrame(
            self.content_frame,
            fg_color=self.color_math_bg,
            corner_radius=6
        )
        block.pack(fill="x", pady=(5, 15), ipadx=15, ipady=12)

        label = ctk.CTkLabel(
            block,
            text=math_text,
            font=self.font_math,
            justify="left",
            text_color=self.color_math_text
        )
        label.pack(anchor="w", padx=15)