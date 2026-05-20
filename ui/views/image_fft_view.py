import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.image_crypto import ImageFFTCrypto

# --- UTILITÁRIO DE TOOLTIP ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 30
        y += self.widget.winfo_rooty() + 20
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left', background="#1A1A1B", foreground="white", relief='solid', borderwidth=1, font=("Arial", 10, "normal"), padx=10, pady=5)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class ImageFFTView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.crypto_engine = ImageFFTCrypto()
        
        self.img_original = None
        self.img_fft = None
        self.img_restored = None

        self.grid_columnconfigure(0, weight=1, minsize=320)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # COLUNA ESQUERDA: CONTROLES E STATUS
        # ==========================================
        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        self.controls = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")
        self.controls.pack(fill="both", expand=True)

        ctk.CTkLabel(self.controls, text="Criptografia 2D (Imagens)", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 15))

        # --- CARD 1: ORIGINAL ---
        self.card_orig = ctk.CTkFrame(self.controls, corner_radius=10)
        self.card_orig.pack(fill="x", pady=(0, 15), ipadx=10, ipady=15)
        
        lbl_orig = ctk.CTkLabel(self.card_orig, text="1. Imagem Base", font=ctk.CTkFont(weight="bold"))
        lbl_orig.pack(anchor="w", padx=15, pady=(5, 10))
        ToolTip(lbl_orig, "Domínio Espacial.\nA imagem vista por nós, composta por pixels de claro e escuro.")

        self.draw_mini_graph(self.card_orig, "photo", "#06b6d4")

        self.btn_load = ctk.CTkButton(self.card_orig, text="Carregar Imagem", fg_color="#06b6d4", hover_color="#0891b2", text_color="white", command=self.load_image_file)
        self.btn_load.pack(fill="x", padx=15, pady=(15, 5))

        # --- CARD 2: FFT ---
        self.card_fft = ctk.CTkFrame(self.controls, corner_radius=10)
        self.card_fft.pack(fill="x", pady=(0, 15), ipadx=10, ipady=15)
        
        lbl_fft = ctk.CTkLabel(self.card_fft, text="2. Criptografia (Espectro)", font=ctk.CTkFont(weight="bold"))
        lbl_fft.pack(anchor="w", padx=15, pady=(5, 5))

        # Mini texto explicativo adicionado
        ctk.CTkLabel(self.card_fft, text="O centro brilha pois contém as formas principais.\nAs bordas escuras contêm os detalhes finos.", font=ctk.CTkFont(size=10), text_color="gray50", justify="left").pack(anchor="w", padx=15, pady=(0, 10))

        self.draw_mini_graph(self.card_fft, "spectrum", "#c026d3")

        self.btn_fft = ctk.CTkButton(self.card_fft, text="Aplicar FFT 2D", fg_color="#c026d3", hover_color="#a21caf", text_color="white", state="disabled", command=self.process_fft)
        self.btn_fft.pack(fill="x", padx=15, pady=(15, 5))

        # --- CARD 3: IFFT ---
        self.card_ifft = ctk.CTkFrame(self.controls, corner_radius=10)
        self.card_ifft.pack(fill="x", pady=(0, 15), ipadx=10, ipady=15)
        
        lbl_ifft = ctk.CTkLabel(self.card_ifft, text="3. Restauração (IFFT)", font=ctk.CTkFont(weight="bold"))
        lbl_ifft.pack(anchor="w", padx=15, pady=(5, 10))

        self.draw_mini_graph(self.card_ifft, "photo_dashed", "#eab308")

        self.btn_ifft = ctk.CTkButton(self.card_ifft, text="Descriptografar (IFFT)", fg_color="#eab308", hover_color="#ca8a04", text_color="black", state="disabled", command=self.process_ifft)
        self.btn_ifft.pack(fill="x", padx=15, pady=(15, 5))

        # --- BARRA DE STATUS ---
        self.status_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.status_frame.pack(fill="x", pady=(10, 0))

        self.lbl_status = ctk.CTkLabel(self.status_frame, text="Status: Aguardando imagem...", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray50")
        self.lbl_status.pack(anchor="w")

        self.progress_bar = ctk.CTkProgressBar(self.status_frame, progress_color="#06b6d4", height=8)
        self.progress_bar.pack(fill="x", pady=(5, 0))
        self.progress_bar.set(0)

        # ==========================================
        # COLUNA DIREITA: DISPLAY MATPLOTLIB
        # ==========================================
        self.view_panel = ctk.CTkFrame(self, corner_radius=10)
        self.view_panel.grid(row=0, column=1, sticky="nsew")
        self.view_panel.grid_rowconfigure(1, weight=1)
        self.view_panel.grid_columnconfigure(0, weight=1)

        # Toolbar superior
        self.toolbar = ctk.CTkFrame(self.view_panel, fg_color="transparent")
        self.toolbar.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        self.toolbar.grid_columnconfigure(2, weight=1) # Espaçador
        
        ctk.CTkLabel(self.toolbar, text="Modo de Exibição:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=(0, 15), sticky="w")
        
        # Segmented Button Consistente
        self.seg_view = ctk.CTkSegmentedButton(
            self.toolbar, 
            values=["Original", "Espectro (Criptografado)", "Restaurado"],
            selected_color="#06b6d4",
            selected_hover_color="#0891b2",
            command=self.update_display
        )
        self.seg_view.grid(row=0, column=1, sticky="w")
        self.seg_view.set("Original")

        # Controles de Zoom/Pan da Imagem
        self.zoom_ctrl = ctk.CTkFrame(self.toolbar, fg_color=("gray85", "gray20"), corner_radius=6)
        self.zoom_ctrl.grid(row=0, column=3, sticky="e")
        btn_opts = {"width": 30, "height": 30, "fg_color": "transparent", "text_color": ("black", "white"), "hover_color": ("gray75", "gray30")}

        ctk.CTkButton(self.zoom_ctrl, text="◀", command=lambda: self.adjust_camera('left'), **btn_opts).pack(side="left", padx=1, pady=1)
        ctk.CTkButton(self.zoom_ctrl, text="▶", command=lambda: self.adjust_camera('right'), **btn_opts).pack(side="left", padx=1, pady=1)
        ctk.CTkButton(self.zoom_ctrl, text="➕", command=lambda: self.adjust_camera('in'), **btn_opts).pack(side="left", padx=1, pady=1)
        ctk.CTkButton(self.zoom_ctrl, text="➖", command=lambda: self.adjust_camera('out'), **btn_opts).pack(side="left", padx=1, pady=1)

        # Matplotlib Canvas
        self.graph_container = ctk.CTkFrame(self.view_panel, corner_radius=5)
        self.graph_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.graph_container.pack_propagate(False)

        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05) 
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.ax.axis('off')
        self.fig.patch.set_facecolor("#151515" if ctk.get_appearance_mode() == "Dark" else "#F9FAFB")

    # ==========================================
    # DESENHO DE MINI-GRÁFICOS 2D
    # ==========================================
    def draw_mini_graph(self, parent, g_type, color_hex):
        bg_color = self._apply_appearance_mode(["#EAEAEA", "#2D2D2E"])
        wrapper = ctk.CTkFrame(parent, fg_color=bg_color, height=60, corner_radius=5)
        wrapper.pack(fill="x", padx=15, pady=5)
        wrapper.pack_propagate(False)

        canvas = tk.Canvas(wrapper, bg=bg_color, highlightthickness=0, height=60)
        canvas.pack(fill="both", expand=True, padx=5, pady=5)

        w = 250

        if g_type == "photo":
            # Desenha um ícone genérico de "Foto" (Montanhas e Sol)
            canvas.create_rectangle(w//2 - 30, 10, w//2 + 30, 40, outline=color_hex, width=2)
            canvas.create_polygon(w//2 - 30, 40, w//2 - 10, 20, w//2 + 10, 40, fill=color_hex)
            canvas.create_polygon(w//2 - 5, 40, w//2 + 15, 25, w//2 + 30, 40, fill=color_hex)
            canvas.create_oval(w//2 + 10, 15, w//2 + 20, 25, fill=color_hex)

        elif g_type == "spectrum":
            # Desenha um espectro FFT: Fundo escuro, estrela/brilho no centro
            canvas.create_rectangle(w//2 - 30, 10, w//2 + 30, 40, fill="#1A1A1B", outline=color_hex, width=2)
            canvas.create_oval(w//2 - 5, 20, w//2 + 5, 30, fill="white", outline="white") # Baixas Frequências
            canvas.create_oval(w//2 - 15, 10, w//2 + 15, 40, fill="", outline="white", dash=(1, 2)) # Halo

        elif g_type == "photo_dashed":
            # Imagem restaurada (Foto com contorno tracejado)
            canvas.create_rectangle(w//2 - 30, 10, w//2 + 30, 40, outline=color_hex, width=2, dash=(2, 2))
            canvas.create_polygon(w//2 - 30, 40, w//2 - 10, 20, w//2 + 10, 40, fill=color_hex)

    # ==========================================
    # LÓGICA DE AÇÕES E CÂMERA
    # ==========================================
    def adjust_camera(self, action):
        """Aplica Zoom e Navegação (Pan) limitando as coordenadas (x, y) da imagem."""
        if not self.ax.get_images(): return # Se não tiver imagem plotada, não faz nada

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim() # No Matplotlib de imagens, o Topo é 0 e a Base é a altura

        width = xlim[1] - xlim[0]
        height = ylim[0] - ylim[1]

        pan_step = width * 0.1

        if action == 'in':
            self.ax.set_xlim(xlim[0] + width*0.1, xlim[1] - width*0.1)
            self.ax.set_ylim(ylim[0] - height*0.1, ylim[1] + height*0.1)
        elif action == 'out':
            self.ax.set_xlim(xlim[0] - width*0.1, xlim[1] + width*0.1)
            self.ax.set_ylim(ylim[0] + height*0.1, ylim[1] - height*0.1)
        elif action == 'left':
            self.ax.set_xlim(xlim[0] - pan_step, xlim[1] - pan_step)
        elif action == 'right':
            self.ax.set_xlim(xlim[0] + pan_step, xlim[1] + pan_step)

        self.canvas.draw()

    def load_image_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")])
        if filepath:
            self.lbl_status.configure(text="Status: Carregando imagem...")
            self.progress_bar.set(0.3)

            # Carrega a matriz da imagem
            self.img_original = self.crypto_engine.load_image(filepath)

            self.btn_fft.configure(state="normal")
            self.seg_view.set("Original")
            self.update_display("Original")

            self.lbl_status.configure(text="Status: Imagem base pronta.")
            self.progress_bar.set(0)

    def process_fft(self):
        self.lbl_status.configure(text="Status: Calculando Transformada Rápida (FFT)...")
        self.progress_bar.set(0.6)

        # Como a FFT pode demorar 1 segundo em imagens HD, forçamos o Tkinter a atualizar a tela
        self.update_idletasks()

        self.img_fft = self.crypto_engine.apply_fft()

        self.btn_ifft.configure(state="normal")
        self.seg_view.set("Espectro (Criptografado)")
        self.update_display("Espectro (Criptografado)")

        self.lbl_status.configure(text="Status: Imagem transformada (Espectro).")
        self.progress_bar.set(1.0)

    def process_ifft(self):
        self.lbl_status.configure(text="Status: Restaurando domínio espacial (IFFT)...")
        self.progress_bar.set(0.6)
        self.update_idletasks()

        self.img_restored = self.crypto_engine.apply_ifft()

        self.seg_view.set("Restaurado")
        self.update_display("Restaurado")

        self.lbl_status.configure(text="Status: Restauração concluída com sucesso.")
        self.progress_bar.set(1.0)

    def update_display(self, view_mode):
        self.ax.clear()
        self.ax.axis('off')
        
        img_to_show = None
        title = ""

        if view_mode == "Original" and self.img_original is not None:
            img_to_show = self.img_original
            title = "Sinal Original (Domínio Espacial)"
        elif view_mode == "Espectro (Criptografado)" and self.img_fft is not None:
            img_to_show = self.img_fft
            title = "Espectro de Frequências (Domínio da Frequência)"
        elif view_mode == "Restaurado" and self.img_restored is not None:
            img_to_show = self.img_restored
            title = "Imagem Restaurada via IFFT"

        if img_to_show is not None:
            self.ax.imshow(img_to_show, cmap='gray')
            fg_color = "white" if ctk.get_appearance_mode() == "Dark" else "black"
            self.ax.set_title(title, color=fg_color, fontsize=12, pad=10)
        else:
            self.ax.text(0.5, 0.5, "Nenhuma imagem processada neste modo.", color="gray", ha="center", va="center")

        self.canvas.draw()