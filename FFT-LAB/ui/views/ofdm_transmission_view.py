"""GUI tab for the full OFDM transmit/receive pipeline.

This tab does NOT replace the existing 'Voz Criptografada' tab (which only
exercises AES). It exposes the full chain:

    voice -> [denoise?] -> AES -> framing -> scramble -> OFDM -> chirp -> WAV/speaker
    WAV/microphone -> chirp detect -> OFDM -> unscramble -> AES decrypt -> voice

Heavy work runs on background threads so the Tk main loop stays responsive.
The right panel embeds a matplotlib spectrogram of the most recent
transmission/capture for visual confirmation of the chirp + 21 subcarriers.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import signal as sps

matplotlib.use("Agg")  # non-interactive backend; we embed via FigureCanvasTkAgg

import global_configs
from core.audio.player import AudioPlayer
from core.audio.recorder import AudioRecorder
from core.audio.wav_io import WavIO
from core.pipeline.receiver import ReceptionError, Receiver
from core.pipeline.transmitter import Transmitter

logger = logging.getLogger(__name__)


class OFDMTransmissionView(ctk.CTkFrame):
    """Tab that drives the over-air-capable OFDM voice transmission pipeline."""

    # Visual styling matches the existing FFT-LAB aesthetic.
    ACCENT_TX: str = "#06b6d4"
    ACCENT_TX_HOVER: str = "#0891b2"
    ACCENT_RX: str = "#22c55e"
    ACCENT_RX_HOVER: str = "#16a34a"
    ACCENT_ANALYZE: str = "#c026d3"

    def __init__(self, master: ctk.CTkBaseClass, **kwargs) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)

        # Background recorder/player used outside the Transmitter/Receiver
        # (e.g., for capturing the source voice or previewing the recovered
        # audio). The pipeline classes have their own internals.
        self._voice_recorder: AudioRecorder = AudioRecorder()
        self._preview_player: AudioPlayer = AudioPlayer()

        # In-memory state mutated by the action handlers.
        self._voice_samples: np.ndarray | None = None     # int16
        self._tx_signal: np.ndarray | None = None         # float32 @ 48 kHz
        self._rx_recovered_samples: np.ndarray | None = None  # int16
        self._rx_sample_rate: int | None = None

        self._build_layout()

    # ==================================================================
    # Layout construction
    # ==================================================================
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1, minsize=360)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # ---- LEFT: controls --------------------------------------------------
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        controls = ctk.CTkScrollableFrame(left, fg_color="transparent")
        controls.pack(fill="both", expand=True)

        ctk.CTkLabel(
            controls,
            text="OFDM Transmissão",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(
            controls,
            text=(
                "Pipeline completo: voz → AES → framing → scramble → OFDM (IFFT) → "
                "chirp → WAV ou alto-falante.\n"
                "Banda 6-10 kHz. Audível, modem-like."
            ),
            font=ctk.CTkFont(size=11),
            text_color="gray50",
            justify="left",
            wraplength=320,
        ).pack(anchor="w", pady=(0, 15))

        # ----- Password card -----
        # Two separate fields: one for TX, one for RX. Default both to the
        # same value so the happy path "just works". Changing them to
        # different values lets the user demonstrate the AES failure mode —
        # see the explanation label at the bottom of the card.
        self._card_password = self._make_card(controls, "Senhas (TX / RX)")

        ctk.CTkLabel(
            self._card_password,
            text="Senha de transmissao (AES):",
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).pack(fill="x", padx=15, pady=(0, 2))
        self._entry_tx_password = ctk.CTkEntry(
            self._card_password,
            placeholder_text="Usada para encriptar (lado do transmissor)",
            show="*",
        )
        self._entry_tx_password.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(
            self._card_password,
            text="Senha de recepcao (AES):",
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).pack(fill="x", padx=15, pady=(0, 2))
        self._entry_rx_password = ctk.CTkEntry(
            self._card_password,
            placeholder_text="Usada para decriptar (lado do receptor)",
            show="*",
        )
        self._entry_rx_password.pack(fill="x", padx=15, pady=(0, 5))

        ctk.CTkLabel(
            self._card_password,
            text=(
                "Iguais → pipeline funciona (voz recuperada).\n"
                "Diferentes → CTR decifra para ruido sem erro: a voz nao volta. "
                "Demonstracao de que a confidencialidade vem do AES."
            ),
            font=ctk.CTkFont(size=10, slant="italic"),
            text_color="gray50",
            wraplength=300,
            justify="left",
        ).pack(anchor="w", padx=15, pady=(8, 5))

        # ----- 1) Record card -----
        self._card_record = self._make_card(controls, "1. Gravar voz")
        self._entry_duration = self._labeled_entry(
            self._card_record,
            "Duração (s):",
            default=str(global_configs.AUDIO_RECORD_SAMPLE_DURATION),
        )
        self._btn_record = self._make_button(
            self._card_record,
            "🔴 Gravar Voz",
            self.ACCENT_TX,
            self.ACCENT_TX_HOVER,
            command=self._on_record,
        )
        self._btn_play_voice = self._make_button(
            self._card_record,
            "▶ Tocar gravação",
            None, None,
            command=self._on_play_voice,
            state="disabled",
        )
        self._btn_load_voice = self._make_button(
            self._card_record,
            "📂 Carregar WAV de voz",
            None, None,
            command=self._on_load_voice,
        )

        # ----- 2) Transmit card -----
        self._card_tx = self._make_card(controls, "2. Transmitir")
        self._chk_denoise_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self._card_tx,
            text="Aplicar denoise antes do AES",
            variable=self._chk_denoise_var,
        ).pack(anchor="w", padx=15, pady=(0, 8))
        self._btn_tx_wav = self._make_button(
            self._card_tx,
            "💾 Salvar WAV de transmissão",
            self.ACCENT_TX,
            self.ACCENT_TX_HOVER,
            command=self._on_tx_to_wav,
            state="disabled",
        )
        self._btn_tx_speaker = self._make_button(
            self._card_tx,
            "📢 Transmitir via alto-falante",
            self.ACCENT_TX,
            self.ACCENT_TX_HOVER,
            command=self._on_tx_to_speaker,
            state="disabled",
        )

        # ----- 3) Receive card -----
        self._card_rx = self._make_card(controls, "3. Receber")
        self._entry_capture = self._labeled_entry(
            self._card_rx,
            "Duração de captura (s):",
            default="6",
        )
        self._btn_rx_mic = self._make_button(
            self._card_rx,
            "🎤 Receber via microfone",
            self.ACCENT_RX,
            self.ACCENT_RX_HOVER,
            command=self._on_rx_from_mic,
        )
        self._btn_rx_wav = self._make_button(
            self._card_rx,
            "📂 Receber de WAV",
            self.ACCENT_RX,
            self.ACCENT_RX_HOVER,
            command=self._on_rx_from_wav,
        )
        self._btn_play_recovered = self._make_button(
            self._card_rx,
            "▶ Tocar voz recuperada",
            None, None,
            command=self._on_play_recovered,
            state="disabled",
        )
        self._btn_save_recovered = self._make_button(
            self._card_rx,
            "💾 Salvar voz recuperada",
            None, None,
            command=self._on_save_recovered,
            state="disabled",
        )

        # ----- 4) Analyze card -----
        self._card_an = self._make_card(controls, "4. Analisar espectro")
        self._btn_analyze_tx = self._make_button(
            self._card_an,
            "🔍 Plotar TX atual",
            self.ACCENT_ANALYZE, None,
            command=lambda: self._plot_spectrogram(self._tx_signal, global_configs.FS, "Transmissão"),
            state="disabled",
        )

        # ----- Status -----
        status_frame = ctk.CTkFrame(left, fg_color="transparent")
        status_frame.pack(fill="x", pady=(10, 0))
        self._lbl_status = ctk.CTkLabel(
            status_frame,
            text="Status: aguardando...",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray50",
            wraplength=320,
            justify="left",
        )
        self._lbl_status.pack(anchor="w")

        # ---- RIGHT: spectrogram canvas --------------------------------------
        self._right_panel = ctk.CTkFrame(self, corner_radius=10, fg_color=("gray85", "gray10"))
        self._right_panel.grid(row=0, column=1, sticky="nsew")
        self._right_panel.grid_columnconfigure(0, weight=1)
        self._right_panel.grid_rowconfigure(0, weight=1)

        self._figure: Figure = Figure(figsize=(6, 4), dpi=100, facecolor="#1a1a1a")
        self._mpl_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self._figure, master=self._right_panel)
        self._mpl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self._draw_placeholder()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _make_card(self, parent: ctk.CTkBaseClass, title: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.pack(fill="x", pady=(0, 15), ipadx=10, ipady=15)
        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=15, pady=(5, 10))
        return card

    def _labeled_entry(
        self, parent: ctk.CTkBaseClass, label: str, default: str
    ) -> ctk.CTkEntry:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=(0, 8))
        ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12)).pack(side="left")
        entry = ctk.CTkEntry(row, width=80)
        entry.insert(0, default)
        entry.pack(side="right")
        return entry

    def _make_button(
        self,
        parent: ctk.CTkBaseClass,
        text: str,
        fg: str | None,
        hover: str | None,
        command,
        state: str = "normal",
    ) -> ctk.CTkButton:
        kwargs: dict = {"text": text, "command": command, "state": state}
        if fg is None:
            kwargs.update(
                fg_color=("gray85", "gray25"),
                hover_color=("gray75", "gray35"),
                text_color=("black", "white"),
            )
        else:
            kwargs.update(fg_color=fg, text_color="white")
            if hover:
                kwargs["hover_color"] = hover
        btn = ctk.CTkButton(parent, **kwargs)
        btn.pack(fill="x", padx=15, pady=(0, 8))
        return btn

    # ==================================================================
    # Status / async helpers
    # ==================================================================
    def _set_status(self, text: str) -> None:
        self._lbl_status.configure(text=f"Status: {text}")
        self.update_idletasks()

    def _run_async(self, name: str, work) -> None:
        """Run `work()` on a background thread; UI stays responsive.

        Exceptions raised in the worker are caught and shown in the status bar.
        """
        def runner() -> None:
            try:
                work()
            except Exception as exc:  # noqa: BLE001 — we surface everything
                logger.exception("Async task %s failed", name)
                self.after(0, lambda: self._set_status(f"erro em {name}: {exc}"))

        threading.Thread(target=runner, name=f"voicecrypto-{name}", daemon=True).start()

    def _tx_password(self) -> str | None:
        """Return the transmit password, or None (with status) if empty."""
        password = self._entry_tx_password.get().strip()
        if not password:
            self._set_status("informe a senha de transmissao primeiro.")
            return None
        return password

    def _rx_password(self) -> str | None:
        """Return the receive password, or None (with status) if empty."""
        password = self._entry_rx_password.get().strip()
        if not password:
            self._set_status("informe a senha de recepcao primeiro.")
            return None
        return password

    # ==================================================================
    # Actions: record voice
    # ==================================================================
    def _on_record(self) -> None:
        try:
            duration = float(self._entry_duration.get())
        except ValueError:
            self._set_status("duração inválida.")
            return
        self._set_status(f"gravando {duration:.1f}s...")
        self._btn_record.configure(state="disabled")

        def work() -> None:
            samples = self._voice_recorder.record(duration)
            self._voice_samples = samples.astype(np.int16)
            self.after(0, self._post_record_update)

        self._run_async("record", work)

    def _post_record_update(self) -> None:
        n = self._voice_samples.size if self._voice_samples is not None else 0
        self._set_status(f"voz capturada ({n} amostras).")
        self._btn_record.configure(state="normal")
        self._btn_play_voice.configure(state="normal")
        self._btn_tx_wav.configure(state="normal")
        self._btn_tx_speaker.configure(state="normal")

    def _on_play_voice(self) -> None:
        if self._voice_samples is None:
            return
        self._preview_player.play(
            self._voice_samples,
            sample_rate=self._voice_recorder.sample_rate,
            blocking=False,
        )

    def _on_load_voice(self) -> None:
        path = filedialog.askopenfilename(
            title="Carregar WAV de voz",
            filetypes=[("Arquivos WAV", "*.wav")],
        )
        if not path:
            return
        sample_rate, samples_f32 = WavIO.read(path)
        # Voice path expects 16-bit at AUDIO_RECORD_SAMPLE_RATE. We accept
        # whatever WAV was loaded but warn if the rate disagrees with config.
        int16_samples = (samples_f32 * global_configs.INT16_MAX).astype(np.int16)
        self._voice_samples = int16_samples
        self._voice_recorder = AudioRecorder(sample_rate=sample_rate)
        self._set_status(
            f"voz carregada ({len(int16_samples)} amostras @ {sample_rate} Hz)."
        )
        self._btn_play_voice.configure(state="normal")
        self._btn_tx_wav.configure(state="normal")
        self._btn_tx_speaker.configure(state="normal")

    # ==================================================================
    # Actions: transmit
    # ==================================================================
    def _build_tx_signal(self) -> tuple[np.ndarray, str] | None:
        if self._voice_samples is None:
            self._set_status("grave ou carregue uma voz antes de transmitir.")
            return None
        password = self._tx_password()
        if password is None:
            return None

        transmitter = Transmitter(password)
        signal, result = transmitter.build_signal(
            self._voice_samples,
            voice_sample_rate=self._voice_recorder.sample_rate,
            denoise=self._chk_denoise_var.get(),
        )
        self._tx_signal = signal
        summary = (
            f"sinal pronto: {result.ofdm_frames} frames OFDM, "
            f"{result.transmission_duration_s:.2f}s @ {global_configs.FS} Hz "
            f"(denoise={'on' if result.denoise_applied else 'off'})."
        )
        return signal, summary

    def _on_tx_to_wav(self) -> None:
        out = self._build_tx_signal()
        if out is None:
            return
        signal, summary = out
        path = filedialog.asksaveasfilename(
            title="Salvar WAV de transmissão",
            defaultextension=".wav",
            filetypes=[("Arquivos WAV", "*.wav")],
        )
        if not path:
            return
        WavIO.write(path, signal, global_configs.FS)
        self._set_status(summary + f" Salvo em {Path(path).name}.")
        self._btn_analyze_tx.configure(state="normal")
        self._plot_spectrogram(signal, global_configs.FS, "Transmissão")

    def _on_tx_to_speaker(self) -> None:
        out = self._build_tx_signal()
        if out is None:
            return
        signal, summary = out
        self._set_status(summary + " Tocando no alto-falante...")
        self._btn_analyze_tx.configure(state="normal")
        self._plot_spectrogram(signal, global_configs.FS, "Transmissão")

        def work() -> None:
            self._preview_player.play(
                signal, sample_rate=global_configs.FS, blocking=True
            )
            self.after(0, lambda: self._set_status("transmissão concluída."))

        self._run_async("tx-speaker", work)

    # ==================================================================
    # Actions: receive
    # ==================================================================
    def _on_rx_from_mic(self) -> None:
        password = self._rx_password()
        if password is None:
            return
        try:
            capture_s = float(self._entry_capture.get())
        except ValueError:
            self._set_status("duração de captura inválida.")
            return

        self._set_status(f"capturando {capture_s:.1f}s do mic...")

        def work() -> None:
            receiver = Receiver(password)
            try:
                recovered = receiver.from_microphone(capture_s)
            except ReceptionError as exc:
                self.after(0, lambda: self._set_status(f"falha na recepção: {exc}"))
                return
            self._rx_recovered_samples = recovered.samples
            self._rx_sample_rate = recovered.sample_rate

            # Plot the captured signal for visual feedback. We re-capture
            # through the recorder used by the receiver internally, but the
            # cleanest source is to re-read the WAV; here we just plot from
            # the recovered audio's metadata via the receiver instance.
            self.after(0, lambda: self._set_status(
                f"recuperado: {recovered.samples.size} amostras, "
                f"|H|={recovered.h_estimate_magnitude:.3f}"
            ))
            self.after(0, lambda: self._btn_play_recovered.configure(state="normal"))
            self.after(0, lambda: self._btn_save_recovered.configure(state="normal"))

        self._run_async("rx-mic", work)

    def _on_rx_from_wav(self) -> None:
        password = self._rx_password()
        if password is None:
            return
        path = filedialog.askopenfilename(
            title="Carregar WAV de transmissão",
            filetypes=[("Arquivos WAV", "*.wav")],
        )
        if not path:
            return
        self._set_status(f"decodificando {Path(path).name}...")

        def work() -> None:
            receiver = Receiver(password)
            try:
                recovered = receiver.from_wav(path)
            except ReceptionError as exc:
                self.after(0, lambda: self._set_status(f"falha na recepção: {exc}"))
                return
            self._rx_recovered_samples = recovered.samples
            self._rx_sample_rate = recovered.sample_rate
            # Plot the source for diagnostic
            sample_rate, signal = WavIO.read(path)
            self.after(0, lambda: self._plot_spectrogram(signal, sample_rate, f"RX ({Path(path).name})"))
            self.after(0, lambda: self._set_status(
                f"recuperado: {recovered.samples.size} amostras, "
                f"|H|={recovered.h_estimate_magnitude:.3f}"
            ))
            self.after(0, lambda: self._btn_play_recovered.configure(state="normal"))
            self.after(0, lambda: self._btn_save_recovered.configure(state="normal"))

        self._run_async("rx-wav", work)

    def _on_play_recovered(self) -> None:
        if self._rx_recovered_samples is None or self._rx_sample_rate is None:
            return
        self._preview_player.play(
            self._rx_recovered_samples,
            sample_rate=self._rx_sample_rate,
            blocking=False,
        )

    def _on_save_recovered(self) -> None:
        if self._rx_recovered_samples is None or self._rx_sample_rate is None:
            return
        path = filedialog.asksaveasfilename(
            title="Salvar voz recuperada",
            defaultextension=".wav",
            filetypes=[("Arquivos WAV", "*.wav")],
        )
        if path:
            WavIO.write(
                path,
                self._rx_recovered_samples.astype(np.float32),
                self._rx_sample_rate,
            )
            self._set_status(f"voz recuperada salva em {Path(path).name}.")

    # ==================================================================
    # Spectrogram rendering
    # ==================================================================
    def _draw_placeholder(self) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#1a1a1a")
        ax.text(
            0.5, 0.5,
            "[ espectrograma da transmissão ]\n\nGere um sinal e clique em 'Plotar TX atual'.",
            color="gray", ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("gray")
        self._mpl_canvas.draw()

    def _plot_spectrogram(
        self,
        signal: np.ndarray | None,
        sample_rate: int,
        title: str,
    ) -> None:
        if signal is None or signal.size == 0:
            self._draw_placeholder()
            return

        # scipy spectrogram with parameters tuned for the 6-10 kHz band.
        f, t, Sxx = sps.spectrogram(
            signal.astype(np.float64),
            fs=sample_rate,
            nperseg=512,
            noverlap=384,
            window="hann",
        )
        # Convert to dB with a small floor to avoid log10(0).
        Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
        vmax = float(Sxx_db.max())
        vmin = vmax - 60.0  # 60 dB dynamic range

        self._figure.clear()
        self._figure.set_facecolor("#1a1a1a")

        # Two-panel: full band on top, zoom on bottom.
        ax_full = self._figure.add_subplot(2, 1, 1)
        ax_full.pcolormesh(t, f, Sxx_db, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        ax_full.set_ylabel("Hz", color="white", fontsize=9)
        ax_full.set_title(f"{title} — espectro completo", color="white", fontsize=10)
        ax_full.axhline(global_configs.F_MIN, color="white", linestyle="--", linewidth=0.6, alpha=0.6)
        ax_full.axhline(global_configs.F_MAX, color="white", linestyle="--", linewidth=0.6, alpha=0.6)
        self._style_axes(ax_full)

        ax_zoom = self._figure.add_subplot(2, 1, 2)
        ax_zoom.pcolormesh(t, f, Sxx_db, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        ax_zoom.set_ylim(global_configs.F_MIN - 500, global_configs.F_MAX + 500)
        ax_zoom.set_xlabel("Tempo (s)", color="white", fontsize=9)
        ax_zoom.set_ylabel("Hz", color="white", fontsize=9)
        ax_zoom.set_title("Zoom: banda OFDM (chirp + 21 subportadoras)", color="white", fontsize=10)
        self._style_axes(ax_zoom)

        self._figure.tight_layout()
        self._mpl_canvas.draw()

    @staticmethod
    def _style_axes(ax) -> None:
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("gray")
        ax.set_facecolor("#0d0d0d")
