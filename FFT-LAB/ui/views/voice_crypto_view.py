import customtkinter as ctk
from tkinter import filedialog

from core.voice_crypto import VoiceCrypto


class VoiceCryptoView(ctk.CTkFrame):
    """Tab that runs the real voice-encryption pipeline (record -> AES -> recover).

    Visualization (spectrum / OFDM) is intentionally left as a placeholder for a
    later stage; for now this wires the working pipeline to the UI.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.engine = VoiceCrypto()

        # Pipeline state
        self.audio = None          # recorded int16 samples
        self.nonce = None          # AES nonce
        self.ciphertext = None     # AES ciphertext bytes
        self.recovered = None      # decrypted int16 samples

        self.grid_columnconfigure(0, weight=1, minsize=340)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # LEFT PANEL: CONTROLS
        # ==========================================
        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        self.controls = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")
        self.controls.pack(fill="both", expand=True)

        ctk.CTkLabel(
            self.controls, text="Voz Criptografada (AES)",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(
            self.controls,
            text="Criptografia REAL com chave (AES-256-CTR).\n"
                 "Diferente da FFT das outras abas, que é reversível sem chave.",
            font=ctk.CTkFont(size=11), text_color="gray50", justify="left"
        ).pack(anchor="w", pady=(0, 15))

        # --- CARD: PASSWORD ---
        self.card_pwd = ctk.CTkFrame(self.controls, corner_radius=10)
        self.card_pwd.pack(fill="x", pady=(0, 15), ipadx=10, ipady=15)
        ctk.CTkLabel(self.card_pwd, text="Senha", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(5, 10))
        self.entry_pwd = ctk.CTkEntry(self.card_pwd, placeholder_text="Digite a senha...", show="*")
        self.entry_pwd.pack(fill="x", padx=15, pady=(0, 5))

        # --- CARD 1: RECORD ---
        self.card_rec = ctk.CTkFrame(self.controls, corner_radius=10)
        self.card_rec.pack(fill="x", pady=(0, 15), ipadx=10, ipady=15)
        ctk.CTkLabel(self.card_rec, text="1. Gravar Voz", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(5, 10))
        self.btn_record = ctk.CTkButton(self.card_rec, text="🔴 Gravar Áudio", fg_color="#06b6d4", hover_color="#0891b2", text_color="white", command=self.action_record)
        self.btn_record.pack(fill="x", padx=15, pady=(0, 8))
        self.btn_play_orig = ctk.CTkButton(self.card_rec, text="▶ Tocar Original", fg_color=("gray85", "gray25"), text_color=("black", "white"), hover_color=("gray75", "gray35"), state="disabled", command=self.action_play_original)
        self.btn_play_orig.pack(fill="x", padx=15, pady=(0, 5))

        # --- CARD 2: ENCRYPT ---
        self.card_enc = ctk.CTkFrame(self.controls, corner_radius=10)
        self.card_enc.pack(fill="x", pady=(0, 15), ipadx=10, ipady=15)
        ctk.CTkLabel(self.card_enc, text="2. Criptografar (AES)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(5, 10))
        self.btn_encrypt = ctk.CTkButton(self.card_enc, text="🔒 Criptografar", fg_color="#c026d3", hover_color="#a21caf", text_color="white", state="disabled", command=self.action_encrypt)
        self.btn_encrypt.pack(fill="x", padx=15, pady=(0, 8))
        self.btn_play_enc = ctk.CTkButton(self.card_enc, text="▶ Tocar Criptografado", fg_color=("gray85", "gray25"), text_color=("black", "white"), hover_color=("gray75", "gray35"), state="disabled", command=self.action_play_encrypted)
        self.btn_play_enc.pack(fill="x", padx=15, pady=(0, 8))
        self.btn_save_enc = ctk.CTkButton(self.card_enc, text="💾 Salvar WAV Criptografado", fg_color="transparent", border_width=1, text_color=("black", "white"), state="disabled", command=self.action_save_encrypted)
        self.btn_save_enc.pack(fill="x", padx=15, pady=(0, 5))

        # --- CARD 3: DECRYPT ---
        self.card_dec = ctk.CTkFrame(self.controls, corner_radius=10)
        self.card_dec.pack(fill="x", pady=(0, 15), ipadx=10, ipady=15)
        ctk.CTkLabel(self.card_dec, text="3. Descriptografar", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(5, 10))
        self.btn_decrypt = ctk.CTkButton(self.card_dec, text="🔓 Descriptografar", fg_color="#eab308", hover_color="#ca8a04", text_color="black", state="disabled", command=self.action_decrypt)
        self.btn_decrypt.pack(fill="x", padx=15, pady=(0, 8))
        self.btn_play_rec = ctk.CTkButton(self.card_dec, text="▶ Tocar Recuperado", fg_color=("gray85", "gray25"), text_color=("black", "white"), hover_color=("gray75", "gray35"), state="disabled", command=self.action_play_recovered)
        self.btn_play_rec.pack(fill="x", padx=15, pady=(0, 5))

        # --- STATUS ---
        self.status_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.status_frame.pack(fill="x", pady=(10, 0))
        self.lbl_status = ctk.CTkLabel(self.status_frame, text="Status: aguardando gravação...", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray50")
        self.lbl_status.pack(anchor="w")

        # ==========================================
        # RIGHT PANEL: VISUALIZATION (placeholder for later)
        # ==========================================
        self.view_panel = ctk.CTkFrame(self, corner_radius=10, fg_color=("gray85", "gray10"))
        self.view_panel.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(
            self.view_panel,
            text="[ Visualização do espectro / OFDM ]\n\nReservado para a próxima etapa.",
            text_color="gray50", justify="center"
        ).place(relx=0.5, rely=0.5, anchor="center")

    # ==========================================
    # ACTIONS
    # ==========================================
    def _set_status(self, text):
        self.lbl_status.configure(text=f"Status: {text}")
        self.update_idletasks()

    def action_record(self):
        self._set_status("gravando...")
        self.audio = self.engine.record()
        self.btn_play_orig.configure(state="normal")
        self.btn_encrypt.configure(state="normal")
        self._set_status(f"gravado ({len(self.audio)} amostras). Pronto para criptografar.")

    def action_play_original(self):
        if self.audio is not None:
            self.engine.play(self.audio)

    def action_encrypt(self):
        password = self.entry_pwd.get()
        if not password:
            self._set_status("digite uma senha antes de criptografar.")
            return
        if self.audio is None:
            self._set_status("grave um áudio primeiro.")
            return
        self.nonce, self.ciphertext = self.engine.encrypt_audio(self.audio, password)
        self.btn_play_enc.configure(state="normal")
        self.btn_save_enc.configure(state="normal")
        self.btn_decrypt.configure(state="normal")
        self._set_status(f"criptografado ({len(self.ciphertext)} bytes). O áudio agora é ruído.")

    def action_play_encrypted(self):
        if self.ciphertext is not None:
            self.engine.play(self.engine.ciphertext_to_samples(self.ciphertext))

    def action_save_encrypted(self):
        if self.ciphertext is None:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("Arquivos de Áudio WAV", "*.wav")],
            title="Salvar áudio criptografado",
        )
        if filepath:
            self.engine.save_encrypted_wav(filepath, self.ciphertext)
            self._set_status("WAV criptografado salvo (soa como ruído).")

    def action_decrypt(self):
        password = self.entry_pwd.get()
        if not password:
            self._set_status("digite a senha usada na criptografia.")
            return
        if self.ciphertext is None:
            self._set_status("criptografe um áudio primeiro.")
            return
        self.recovered = self.engine.decrypt_audio(self.nonce, self.ciphertext, password)
        self.btn_play_rec.configure(state="normal")
        self._set_status("descriptografado. Senha errada = ruído (CTR não verifica integridade).")

    def action_play_recovered(self):
        if self.recovered is not None:
            self.engine.play(self.recovered)
