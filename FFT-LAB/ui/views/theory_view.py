import customtkinter as ctk

class TheoryView(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        # Cabeçalho Principal (Mantido alinhado à esquerda como título da página)
        self.title_label = ctk.CTkLabel(
            self,
            text="Módulo Teórico",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        self.title_label.pack(anchor="w", padx=20, pady=(20, 10))

        # ==========================================
        # CONTEÚDO CENTRALIZADO (Estilo "Blog Moderno")
        # ==========================================
        # Usamos padx generoso para "espremer" o texto e evitar linhas muito longas
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=(80, 80), pady=10)

        # --- TIPOGRAFIA PADRÃO ---
        font_title = ctk.CTkFont(size=20, weight="bold")
        font_body = ctk.CTkFont(size=14, weight="normal")

        # O wraplength em 700px garante uma média de 60-80 caracteres por linha
        text_wrap = 700

        # --- SEÇÃO 1: Tempo vs Frequência ---
        # Contraste Ajustado: Azul escuro no Modo Claro, Ciano no Modo Escuro
        ctk.CTkLabel(
            self.content_frame, 
            text="O Domínio do Tempo vs. Frequência", 
            font=font_title,
            text_color=("#006B8F", "#06b6d4")
        ).pack(anchor="w", pady=(20, 10)) # Ritmo: (Espaço Maior antes do Título, Menor depois)
        
        texto1 = (
            "No processamento de sinais, visualizamos sons nativamente no domínio do tempo (onde o gráfico exibe "
            "a amplitude da onda variando ao longo dos milissegundos). Porém, para análises complexas, o domínio da "
            "frequência se torna essencial. Ele desmonta o sinal original, revelando de forma clara quais 'notas' "
            "ou frequências exatas estão compondo aquele som naquele exato momento."
        )
        ctk.CTkLabel(
            self.content_frame,
            text=texto1,
            font=font_body,
            justify="left",
            wraplength=text_wrap,
            text_color=("#333333", "#CCCCCC") # Cinza escuro no Claro, Cinza claro no Escuro
        ).pack(anchor="w", pady=(0, 20)) # Ritmo: Espaço médio entre o texto e a imagem

        # Mockup de Imagem 1
        self.img_placeholder1 = ctk.CTkFrame(self.content_frame, height=220, corner_radius=10, border_width=2, fg_color=("gray90", "gray15"))
        self.img_placeholder1.pack(fill="x", pady=(0, 50)) # Ritmo: Espaço GRANDE (50px) para separar da próxima seção
        self.img_placeholder1.pack_propagate(False)

        ctk.CTkLabel(
            self.img_placeholder1,
            text="[ 🖼️ Diagrama: Comparação Tempo x Frequência ]",
            text_color=("#555555", "#888888")
        ).place(relx=0.5, rely=0.5, anchor="center")

        # --- SEÇÃO 2: Por que usar a FFT? ---
        # Contraste Ajustado: Roxo escuro no Modo Claro, Magenta no Modo Escuro
        ctk.CTkLabel(
            self.content_frame, 
            text="Por que usar a Transformada Rápida (FFT)?",
            font=font_title,
            text_color=("#86198F", "#c026d3")
        ).pack(anchor="w", pady=(0, 10))
        
        texto2 = (
            "A Transformada Discreta de Fourier (DFT) convencional é matematicamente perfeita, mas computacionalmente "
            "lenta (sua complexidade é O(N²)). A Transformada Rápida de Fourier (FFT), popularizada por Cooley e Tukey "
            "em 1965, é um algoritmo que aproveita redundâncias matemáticas para resolver a mesma equação em um tempo "
            "drasticamente menor (O(N log N)). Isso é o que torna possível processar áudio em tempo real nos computadores modernos."
        )
        ctk.CTkLabel(
            self.content_frame,
            text=texto2,
            font=font_body,
            justify="left",
            wraplength=text_wrap,
            text_color=("#333333", "#CCCCCC")
        ).pack(anchor="w", pady=(0, 20))

        # Mockup de Imagem 2
        self.img_placeholder2 = ctk.CTkFrame(self.content_frame, height=220, corner_radius=10, border_width=2, fg_color=("gray90", "gray15"))
        self.img_placeholder2.pack(fill="x", pady=(0, 50))
        self.img_placeholder2.pack_propagate(False)

        ctk.CTkLabel(
            self.img_placeholder2,
            text="[ 🖼️ Infográfico: Complexidade O(N²) vs O(N log N) ]",
            text_color=("#555555", "#888888")
        ).place(relx=0.5, rely=0.5, anchor="center")