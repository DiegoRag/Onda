import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.audio_generator import AudioGenerator

class GeneratorView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.audio_gen = AudioGenerator()

        self.grid_columnconfigure(0, weight=1, minsize=350) # Aumentei um pouco para caber os inputs
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        self.wave_counter = 1
        self.active_waves = []

        self.plot_duration = 0.02
        self.plot_offset = 0.0

        # ==========================================
        # COLUNA ESQUERDA: MIXER
        # ==========================================
        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        self.mixer_panel = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")
        self.mixer_panel.pack(fill="both", expand=True)

        self.title_label = ctk.CTkLabel(self.mixer_panel, text="Sintetizador de Sinais", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(anchor="w", pady=(0, 5))
        
        self.math_label = ctk.CTkLabel(self.mixer_panel, text="y(t) = A * sin(2πft + φ)", text_color=("#555555", "#AAAAAA"))
        self.math_label.pack(anchor="w", pady=(0, 20))

        self.btn_add_wave = ctk.CTkButton(
            self.mixer_panel, text="➕ Adicionar Nova Onda", font=ctk.CTkFont(weight="bold"),
            fg_color="#06b6d4", text_color="white", hover_color="#0891b2",
            command=self.add_wave_card
        )
        self.btn_add_wave.pack(fill="x", pady=(0, 20), ipady=8)

        self.waves_container = ctk.CTkFrame(self.mixer_panel, fg_color="transparent")
        self.waves_container.pack(fill="both", expand=True)

        # ==========================================
        # COLUNA DIREITA: OSCILOSCÓPIO E EXPORTAÇÃO
        # ==========================================
        self.right_panel = ctk.CTkFrame(self, corner_radius=10)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)

        # --- TOOLBAR (NOVOS CONTROLES DE MODO) ---
        self.toolbar_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.toolbar_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        self.toolbar_frame.grid_columnconfigure(2, weight=1) # Espaçador dinâmico

        # 1. Switch de Inspeção Visual
        self.switch_inspect = ctk.CTkSwitch(
            self.toolbar_frame, text="Inspecionar Ondas", progress_color="#06b6d4",
            text_color=("black", "white"), command=self.update_graph
        )
        self.switch_inspect.grid(row=0, column=0, sticky="w", padx=(0, 20))

        # 2. Modo de Síntese (Aditiva vs Portadora)
        self.synth_mode = ctk.CTkSegmentedButton(
            self.toolbar_frame,
            values=["Síntese Aditiva (+)", "Modulação AM (Portadora)"],
            selected_color="#c026d3", selected_hover_color="#a21caf",
            command=self.update_graph
        )
        self.synth_mode.grid(row=0, column=1, sticky="w")
        self.synth_mode.set("Síntese Aditiva (+)") # Padrão

        # 3. Controles de Zoom
        self.zoom_ctrl = ctk.CTkFrame(self.toolbar_frame, fg_color=("gray85", "gray20"), corner_radius=6)
        self.zoom_ctrl.grid(row=0, column=3, sticky="e")
        btn_opts = {"width": 30, "height": 30, "fg_color": "transparent", "text_color": ("black", "white"), "hover_color": ("gray75", "gray30")}

        ctk.CTkButton(self.zoom_ctrl, text="◀", command=lambda: self.adjust_camera('left'), **btn_opts).pack(side="left", padx=1, pady=1)
        ctk.CTkButton(self.zoom_ctrl, text="▶", command=lambda: self.adjust_camera('right'), **btn_opts).pack(side="left", padx=1, pady=1)
        ctk.CTkButton(self.zoom_ctrl, text="➕", command=lambda: self.adjust_camera('in'), **btn_opts).pack(side="left", padx=1, pady=1)
        ctk.CTkButton(self.zoom_ctrl, text="➖", command=lambda: self.adjust_camera('out'), **btn_opts).pack(side="left", padx=1, pady=1)

        # --- ÁREA DO GRÁFICO (MATPLOTLIB) ---
        self.graph_container = ctk.CTkFrame(self.right_panel, corner_radius=5)
        self.graph_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.graph_container.pack_propagate(False)

        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- MÓDULO DE ÁUDIO ---
        self.audio_panel = ctk.CTkFrame(self.right_panel, fg_color=("gray90", "gray15"), corner_radius=10)
        self.audio_panel.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20), ipadx=10, ipady=10)
        self.audio_panel.grid_columnconfigure(1, weight=1)

        self.player_frame = ctk.CTkFrame(self.audio_panel, fg_color="transparent")
        self.player_frame.grid(row=0, column=0, sticky="w", padx=10)

        self.btn_play = ctk.CTkButton(self.player_frame, text="▶ PLAY", width=100, fg_color="#10b981", hover_color="#059669", text_color="white", font=ctk.CTkFont(weight="bold"), command=self.action_play)
        self.btn_play.pack(side="left", padx=(0, 10))
        self.btn_stop = ctk.CTkButton(self.player_frame, text="■ STOP", width=100, fg_color="#ef4444", hover_color="#dc2626", text_color="white", font=ctk.CTkFont(weight="bold"), command=self.action_stop)
        self.btn_stop.pack(side="left")

        self.export_frame = ctk.CTkFrame(self.audio_panel, fg_color="transparent")
        self.export_frame.grid(row=0, column=2, sticky="e", padx=10)

        self.btn_wav = ctk.CTkButton(self.export_frame, text="💾 Salvar WAV", width=110, fg_color="transparent", border_width=1, text_color=("black", "white"), command=self.action_export_wav)
        self.btn_wav.pack(side="left", padx=(0, 10))

        self.add_wave_card()

    # ==========================================
    # LÓGICA DE CÂMERA E GRÁFICO (COM ONDA PORTADORA)
    # ==========================================
    def adjust_camera(self, action):
        if action == 'in': self.plot_duration /= 1.5
        elif action == 'out': self.plot_duration *= 1.5
        elif action == 'left': self.plot_offset = max(0, self.plot_offset - (self.plot_duration * 0.2))
        elif action == 'right': self.plot_offset += (self.plot_duration * 0.2)
        self.update_graph()

    def update_graph(self, *args):
        self.ax.clear()
        is_dark = ctk.get_appearance_mode() == "Dark"
        bg_color = "#151515" if is_dark else "#F9FAFB"
        fg_color = "white" if is_dark else "black"

        self.fig.patch.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)
        self.ax.tick_params(colors=fg_color)
        for spine in self.ax.spines.values(): spine.set_color("#4B5563")

        if not self.active_waves:
            self.ax.text(0.5, 0.5, "Nenhuma onda ativa.", color="gray", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw()
            return

        t_plot = self.audio_gen.create_time_array(self.plot_duration) + self.plot_offset
        inspect_mode = self.switch_inspect.get() == 1
        current_mode = self.synth_mode.get()
        colors = ["#c026d3", "#eab308", "#ec4899", "#8b5cf6"]

        # --- LÓGICA DE SÍNTESE ---
        if current_mode == "Síntese Aditiva (+)":
            final_wave = np.zeros_like(t_plot)
            for idx, wave_data in enumerate(self.active_waves):
                single_wave = self.audio_gen.generate_single_wave(wave_data["type_var"].get(), wave_data["freq_var"].get(), wave_data["amp_var"].get(), wave_data["phase_var"].get(), t_plot)
                final_wave += single_wave
                if inspect_mode:
                    self.ax.plot(t_plot, single_wave, color=colors[idx % len(colors)], linestyle="--", alpha=0.6, linewidth=1.5)

            self.ax.plot(t_plot, final_wave, color="#06b6d4", linewidth=2.5, label="Soma Final")

        elif current_mode == "Modulação AM (Portadora)":
            # Onda 1 é a PORTADORA (Carrier)
            c_data = self.active_waves[0]
            carrier = self.audio_gen.generate_single_wave(c_data["type_var"].get(), c_data["freq_var"].get(), c_data["amp_var"].get(), c_data["phase_var"].get(), t_plot)

            # As outras ondas formam a MODULADORA (Envelope)
            modulator = np.zeros_like(t_plot)
            for idx, wave_data in enumerate(self.active_waves[1:]):
                single_mod = self.audio_gen.generate_single_wave(wave_data["type_var"].get(), wave_data["freq_var"].get(), wave_data["amp_var"].get(), wave_data["phase_var"].get(), t_plot)
                modulator += single_mod

            # Matemática da Modulação AM: Sinal = (1 + Envelope) * Portadora
            final_wave = (1.0 + modulator) * carrier

            if inspect_mode:
                self.ax.plot(t_plot, carrier, color=colors[0], alpha=0.5, linewidth=1.0, label="Onda Portadora (Onda 1)")
                if len(self.active_waves) > 1:
                    # Desenha o Envelope Superior e Inferior para estudo visual
                    self.ax.plot(t_plot, 1.0 + modulator, color="#eab308", linestyle="--", linewidth=1.5, label="Envelope")
                    self.ax.plot(t_plot, -1.0 - modulator, color="#eab308", linestyle="--", linewidth=1.5, alpha=0.5)

            self.ax.plot(t_plot, final_wave, color="#c026d3", linewidth=2.0, label="Sinal Modulado")
            self.ax.legend(loc="upper right", fontsize=8, facecolor=bg_color, edgecolor="#4B5563", labelcolor=fg_color)

        self.ax.set_title(f"Osciloscópio - {current_mode}", color=fg_color, fontsize=10, pad=10)
        self.ax.set_xlabel("Tempo (segundos)", color=fg_color, fontsize=9)
        self.ax.set_ylabel("Amplitude", color=fg_color, fontsize=9)
        self.ax.grid(True, linestyle=":", alpha=0.3, color=fg_color)
        self.canvas.draw()

    # ==========================================
    # LÓGICA DE CRIAÇÃO DINÂMICA (COM INPUTS DE TEXTO)
    # ==========================================
    def add_wave_card(self):
        if len(self.active_waves) >= 4: return

        colors = ["#c026d3", "#eab308", "#ec4899", "#8b5cf6"]
        color = colors[(self.wave_counter - 1) % len(colors)]

        card = ctk.CTkFrame(self.waves_container, corner_radius=8)
        card.pack(fill="x", pady=(0, 15), ipadx=5, ipady=10)

        wave_data = {
            "frame": card,
            "type_var": ctk.StringVar(value="Senoidal (sin)"),
            "freq_var": ctk.DoubleVar(value=440),
            "amp_var": ctk.DoubleVar(value=50),
            "phase_var": ctk.DoubleVar(value=0)
        }
        self.active_waves.append(wave_data)

        border = ctk.CTkFrame(card, width=5, fg_color=color, corner_radius=0)
        border.place(relx=0, rely=0, relheight=1)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=(15, 10), pady=(0, 10))

        # O título se adapta dinamicamente caso seja modo AM
        title_text = "Onda 1 (Portadora)" if self.wave_counter == 1 else f"Onda {self.wave_counter}"
        title_label = ctk.CTkLabel(header, text=title_text, font=ctk.CTkFont(weight="bold"))
        title_label.pack(side="left")
        wave_data["title_label"] = title_label # Guarda referência para atualizar depois

        btn_close = ctk.CTkButton(header, text="🗑️", width=30, height=30, fg_color="transparent", text_color=("#4B5563", "#9CA3AF"), hover_color="#ef4444")
        btn_close.bind("<Enter>", lambda e: btn_close.configure(text_color="white"))
        btn_close.bind("<Leave>", lambda e: btn_close.configure(text_color=("#4B5563", "#9CA3AF")))

        def delete_wave():
            self.active_waves.remove(wave_data)
            card.destroy()
            self.update_graph()

        btn_close.configure(command=delete_wave)
        btn_close.pack(side="right")

        self.wave_counter += 1

        ctk.CTkLabel(card, text="Forma de Onda:", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=15)
        ctk.CTkOptionMenu(
            card, values=["Senoidal (sin)", "Quadrada (square)", "Dente de Serra (saw)", "Triangular (tri)"],
            variable=wave_data["type_var"], fg_color=("gray85", "gray25"), text_color=("black", "white"), command=self.update_graph
        ).pack(fill="x", padx=15, pady=(0, 15))

        # Agora usando a nova função com Inputs Digitáveis
        self.create_interactive_slider(card, "Frequência", 20, 5000, 440, "Hz", color, wave_data["freq_var"])
        self.create_interactive_slider(card, "Amplitude", 0, 100, 50, "%", color, wave_data["amp_var"])
        self.create_interactive_slider(card, "Fase", 0, 360, 0, "°", color, wave_data["phase_var"])

        self.update_graph()

    def create_interactive_slider(self, parent, name, min_val, max_val, default_val, unit, hex_color, tk_var):
        """Cria o controle deslizante acompanhado de um campo de texto (Entry) editável."""
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.pack(fill="x", padx=15, pady=(0, 10))

        labels_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
        labels_frame.pack(fill="x", pady=(0, 2))

        ctk.CTkLabel(labels_frame, text=name, font=ctk.CTkFont(size=12)).pack(side="left")

        # Container do Input + Unidade
        input_container = ctk.CTkFrame(labels_frame, fg_color="transparent")
        input_container.pack(side="right")

        # Campo de Texto Interativo (Entry)
        val_entry = ctk.CTkEntry(input_container, width=50, height=24, fg_color=("gray85", "gray20"), text_color=hex_color, font=ctk.CTkFont(weight="bold"), border_width=1)
        val_entry.insert(0, str(default_val))
        val_entry.pack(side="left", padx=(0, 5))

        ctk.CTkLabel(input_container, text=unit, font=ctk.CTkFont(size=12, weight="bold"), text_color=hex_color).pack(side="left")

        slider = ctk.CTkSlider(row_frame, from_=min_val, to=max_val, progress_color=hex_color, variable=tk_var)
        slider.pack(fill="x")

        # Lógica 1: Arrastar o slider atualiza o campo de texto
        def sync_entry_from_slider(value):
            val_entry.delete(0, 'end')
            val_entry.insert(0, str(int(value)))
            self.update_graph()

        slider.configure(command=sync_entry_from_slider)

        # Lógica 2: Digitar no campo de texto atualiza o slider (ao apertar Enter ou clicar fora)
        def sync_slider_from_entry(event=None):
            try:
                # Tenta converter o texto digitado em número
                typed_val = float(val_entry.get())
                # Garante que não vai estourar os limites do slider
                typed_val = max(min_val, min(max_val, typed_val))

                tk_var.set(typed_val)
                slider.set(typed_val)

                val_entry.delete(0, 'end')
                val_entry.insert(0, str(int(typed_val)))
                self.update_graph()
            except ValueError:
                # Se o usuário digitar "abc", ignora e volta ao valor que estava no slider
                val_entry.delete(0, 'end')
                val_entry.insert(0, str(int(tk_var.get())))

        val_entry.bind("<Return>", sync_slider_from_entry) # Atualiza ao dar Enter
        val_entry.bind("<FocusOut>", sync_slider_from_entry) # Atualiza ao clicar fora da caixa

    # ==========================================
    # LÓGICA DE ÁUDIO COM SUPORTE AM
    # ==========================================
    def build_final_audio_array(self, duration=2.0):
        if not self.active_waves: return None
        t = self.audio_gen.create_time_array(duration)
        current_mode = self.synth_mode.get()

        if current_mode == "Síntese Aditiva (+)":
            final_wave = np.zeros_like(t)
            for wave_data in self.active_waves:
                final_wave += self.audio_gen.generate_single_wave(wave_data["type_var"].get(), wave_data["freq_var"].get(), wave_data["amp_var"].get(), wave_data["phase_var"].get(), t)

        elif current_mode == "Modulação AM (Portadora)":
            c_data = self.active_waves[0]
            carrier = self.audio_gen.generate_single_wave(c_data["type_var"].get(), c_data["freq_var"].get(), c_data["amp_var"].get(), c_data["phase_var"].get(), t)

            modulator = np.zeros_like(t)
            for wave_data in self.active_waves[1:]:
                modulator += self.audio_gen.generate_single_wave(wave_data["type_var"].get(), wave_data["freq_var"].get(), wave_data["amp_var"].get(), wave_data["phase_var"].get(), t)

            final_wave = (1.0 + modulator) * carrier

        max_amplitude = np.max(np.abs(final_wave))
        if max_amplitude > 1.0: final_wave = final_wave / max_amplitude
        return final_wave

    def action_play(self):
        final_wave = self.build_final_audio_array(duration=2.0)
        if final_wave is not None: self.audio_gen.play_audio(final_wave)

    def action_stop(self):
        self.audio_gen.stop_audio()

    def action_export_wav(self):
        final_wave = self.build_final_audio_array(duration=5.0)
        if final_wave is not None:
            filepath = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Arquivos de Áudio WAV", "*.wav")], title="Salvar Onda Resultante")
            if filepath: self.audio_gen.export_wav(filepath, final_wave)