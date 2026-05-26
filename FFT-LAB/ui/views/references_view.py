import customtkinter as ctk
import webbrowser

class ReferencesView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        # ==========================================
        # CABEÇALHO E SUBTÍTULO
        # ==========================================
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(40, 20))

        # Título Centralizado
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Referências e Documentação",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        self.title_label.pack(anchor="center")

        # Subtítulo (Contextualização com contraste seguro)
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text="Explore os fundamentos matemáticos da FFT e conheça as ferramentas open-source que tornaram este laboratório possível.",
            font=ctk.CTkFont(size=14),
            text_color=("#555555", "#AAAAAA")
        )
        self.subtitle_label.pack(anchor="center", pady=(5, 0))


        # ==========================================
        # CONTAINER CENTRAL DOS CARDS
        # ==========================================
        # Usamos um frame expansível para empurrar os cards para o meio da tela, sem esticá-los
        self.center_wrapper = ctk.CTkFrame(self, fg_color="transparent")
        self.center_wrapper.pack(expand=True, fill="both")

        # Grid interno que segura os dois cards centralizados
        self.grid_frame = ctk.CTkFrame(self.center_wrapper, fg_color="transparent")
        self.grid_frame.place(relx=0.5, rely=0.45, anchor="center") # Centraliza perfeitamente no eixo X e um pouco acima no eixo Y

        # --- CARD 1: BIBLIOTECAS ---
        # Definimos uma largura fixa para o card não esticar infinitamente
        self.card_libs = ctk.CTkFrame(self.grid_frame, corner_radius=10, width=350, height=220)
        self.card_libs.grid(row=0, column=0, padx=15, pady=15)
        self.card_libs.pack_propagate(False) # Impede que o frame encolha/estique com o texto interno

        ctk.CTkLabel(self.card_libs, text="Bibliotecas Utilizadas", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=(20, 15))

        self.create_link_button(self.card_libs, "NumPy / SciPy", "https://scipy.org/")
        self.create_link_button(self.card_libs, "CustomTkinter", "https://customtkinter.tomschimansky.com/")
        self.create_link_button(self.card_libs, "Matplotlib", "https://matplotlib.org/")

        # --- CARD 2: MATERIAL DE ESTUDO ---
        self.card_study = ctk.CTkFrame(self.grid_frame, corner_radius=10, width=350, height=220)
        self.card_study.grid(row=0, column=1, padx=15, pady=15)
        self.card_study.pack_propagate(False)

        ctk.CTkLabel(self.card_study, text="Material de Apoio", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=(20, 15))
        
        self.create_link_button(self.card_study, "3Blue1Brown: A Transformada de Fourier", "https://www.youtube.com/watch?v=spUNpyF58BY")
        self.create_link_button(self.card_study, "Documentação FFT (SciPy)", "https://docs.scipy.org/doc/scipy/reference/fft.html")


        # ==========================================
        # RODAPÉ (FOOTER COM CRÉDITOS)
        # ==========================================
        self.footer_label = ctk.CTkLabel(
            self,
            text="Desenvolvido por Diego Ragozini, Gustavo Germiniani, Henrique Macedo, Henrique Prado, João Henrique Ferreira e Nathaly Vieira • 2026 • Versão 1.0.1",
            font=ctk.CTkFont(size=11),
            text_color=("#888888", "#666666") # Um cinza bem discreto para não brigar com a interface
        )
        self.footer_label.pack(side="bottom", pady=20)


    # ==========================================
    # FUNÇÃO DE CRIAÇÃO DE LINKS
    # ==========================================
    def create_link_button(self, parent, text, url):
        """Cria um botão de link com Feedback Visual (Hover e Sublinhado)."""

        # Declaramos duas fontes: uma normal e uma com sublinhado para o Hover
        font_normal = ctk.CTkFont(size=14)
        font_hover = ctk.CTkFont(size=14, underline=True)

        btn = ctk.CTkButton(
            parent, 
            text=f"🔗 {text}", 
            fg_color="transparent", 
            text_color=("#006B8F", "#06b6d4"), # Azul escuro (Claro) e Ciano (Escuro)
            hover_color=("gray85", "gray20"),
            anchor="w",
            font=font_normal,
            command=lambda: webbrowser.open(url)
        )
        btn.pack(fill="x", padx=15, pady=2)

        # Eventos para o Affordance (Hover)
        btn.bind("<Enter>", lambda e: btn.configure(font=font_hover))
        btn.bind("<Leave>", lambda e: btn.configure(font=font_normal))