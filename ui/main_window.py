import customtkinter as ctk
from ui.views.fft_lab_view import FFTLabView
from ui.views.generator_view import GeneratorView
from ui.views.theory_view import TheoryView
from ui.views.references_view import ReferencesView
from ui.views.image_fft_view import ImageFFTView


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("FFT Lab")
        self.geometry("1200x750")
        self.minsize(900, 600)

        # Novo Grid: Linha 0 para a Navbar, Linha 1 para o Conteúdo
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ==========================================
        # NAVBAR SUPERIOR
        # ==========================================
        self.navbar = ctk.CTkFrame(self, height=60, corner_radius=0)
        self.navbar.grid(row=0, column=0, sticky="ew")

        # Logo
        self.logo_label = ctk.CTkLabel(self.navbar, text="FFT LAB", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(side="left", padx=(20, 40), pady=15)

        # Botões de Navegação (Dicionário para facilitar a troca de cores)
        self.nav_buttons = {}

        self.nav_buttons["fft"] = ctk.CTkButton(self.navbar, text="FFT Lab", fg_color="transparent", text_color=("gray10", "gray90"), command=lambda: self.select_frame("fft"))
        self.nav_buttons["fft"].pack(side="left", padx=5)

        # Linha corrigida (indentada corretamente para dentro do __init__)
        self.nav_buttons["image_fft"] = ctk.CTkButton(self.navbar, text="Criptografia 2D", fg_color="transparent", text_color=("gray10", "gray90"), command=lambda: self.select_frame("image_fft"))
        self.nav_buttons["image_fft"].pack(side="left", padx=5)

        self.nav_buttons["generator"] = ctk.CTkButton(self.navbar, text="Sintetizador", fg_color="transparent", text_color=("gray10", "gray90"), command=lambda: self.select_frame("generator"))
        self.nav_buttons["generator"].pack(side="left", padx=5)

        self.nav_buttons["theory"] = ctk.CTkButton(self.navbar, text="Teoria", fg_color="transparent", text_color=("gray10", "gray90"), command=lambda: self.select_frame("theory"))
        self.nav_buttons["theory"].pack(side="left", padx=5)

        self.nav_buttons["references"] = ctk.CTkButton(self.navbar, text="Referências", fg_color="transparent", text_color=("gray10", "gray90"), command=lambda: self.select_frame("references"))
        self.nav_buttons["references"].pack(side="left", padx=5)

        # Botão de Tema (Modo Claro/Escuro) alinhado à direita com text_color corrigido
        self.appearance_mode_btn = ctk.CTkButton(
            self.navbar,
            text="Modo Claro",
            width=100,
            fg_color="transparent",
            text_color=("black", "white"),
            border_width=1,
            command=self.toggle_appearance_mode
        )
        self.appearance_mode_btn.pack(side="right", padx=20)

        # ==========================================
        # ÁREA PRINCIPAL
        # ==========================================
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        # Instanciando Telas
        self.frames = {}
        self.frames["fft"] = FFTLabView(self.main_container)
        self.frames["image_fft"] = ImageFFTView(self.main_container)
        self.frames["generator"] = GeneratorView(self.main_container)
        self.frames["theory"] = TheoryView(self.main_container)
        self.frames["references"] = ReferencesView(self.main_container)

        self.select_frame("fft")

    def select_frame(self, name):
        """Alterna a tela e atualiza o estado ativo da Navbar"""
        for btn_name, btn in self.nav_buttons.items():
            if btn_name == name:
                btn.configure(fg_color=("#06b6d4", "#06b6d4"), text_color="white")
            else:
                btn.configure(fg_color="transparent", text_color=("gray10", "gray90"))

        for frame in self.frames.values():
            frame.grid_forget()
        self.frames[name].grid(row=0, column=0, sticky="nsew")

    def toggle_appearance_mode(self):
        current_mode = ctk.get_appearance_mode()
        if current_mode == "Dark":
            ctk.set_appearance_mode("Light")
            self.appearance_mode_btn.configure(text="Modo Escuro")
        else:
            ctk.set_appearance_mode("Dark")
            self.appearance_mode_btn.configure(text="Modo Claro")