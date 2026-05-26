"""FFT Lab — visualize a WAV (or live-recorded) audio in time / frequency / IFFT.

This tab is a self-contained demonstration of the forward/inverse Fourier
relationship:

    time_signal  --FFT-->  freq_spectrum  --IFFT-->  recovered_signal

The three signals are toggled by the checkboxes at the top of the right
panel; matplotlib renders whatever combination is selected.

Two recording flows are supported:

    - Fixed-duration record (legacy): not used here anymore, but the
      underlying AudioRecorder method is preserved for other consumers.
    - Live (push-to-talk) record: click "Gravar" -> stream starts and the
      strip chart on the right scrolls in real time; click "Parar" to stop.

After a signal exists (loaded or recorded), a ▶ Tocar button on the right
panel plays it back. A yellow vertical cursor moves across the time-domain
plots in sync with playback.

OOP composition used here
-------------------------
    WavIO            -- core/audio/wav_io.py        (file loading)
    AudioRecorder    -- core/audio/recorder.py      (microphone capture + streaming)
    AudioPlayer      -- core/audio/player.py        (non-blocking playback)
    SignalState      -- ui/widgets/signal_state.py  (state container)
    TriplePlotCanvas -- ui/widgets/triple_plot_canvas.py (matplotlib widget)
"""

from __future__ import annotations

import logging
import math
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import numpy as np

from core.audio.player import AudioPlayer
from core.audio.recorder import AudioRecorder
from core.audio.wav_io import WavIO
from ui.widgets.signal_state import SignalState
from ui.widgets.triple_plot_canvas import TriplePlotCanvas

logger = logging.getLogger(__name__)


# ============================================================================
# Tooltip — small helper for hover-explanations on labels.
# ============================================================================
class ToolTip:
    """Hover tooltip on any tk widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tooltip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, event=None) -> None:
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 30
        y += self.widget.winfo_rooty() + 20
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#1A1A1B",
            foreground="white",
            relief="solid",
            borderwidth=1,
            font=("Arial", 10, "normal"),
            padx=10,
            pady=5,
        ).pack(ipadx=1)
        self.tooltip_window = tw

    def _hide(self, event=None) -> None:
        if self.tooltip_window is not None:
            self.tooltip_window.destroy()
            self.tooltip_window = None


# ============================================================================
# FFT Lab View
# ============================================================================
class FFTLabView(ctk.CTkFrame):
    """Time / FFT / IFFT visualization tab with live recording and playback."""

    # Color palette — shared between left-panel decoration and the canvas.
    COLOR_ORIGINAL: str = "#06b6d4"   # cyan
    COLOR_FFT: str = "#c026d3"        # purple
    COLOR_IFFT: str = "#f97316"       # orange
    COLOR_IFFT_HOVER: str = "#ea580c"
    COLOR_PLAY: str = "#22c55e"       # green
    COLOR_PLAY_HOVER: str = "#16a34a"
    COLOR_RECORD: str = "#ef4444"     # red, used when "Parar Gravacao"
    COLOR_RECORD_HOVER: str = "#dc2626"

    # Recording settings.
    RECORD_SAMPLE_RATE: int = 16_000
    LIVE_WINDOW_SECONDS: float = 3.0
    LIVE_FPS: int = 30                # canvas refresh rate during recording
    PLAYBACK_FPS: int = 30            # cursor refresh rate during playback

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)

        # ----- Composition: reusable building blocks -----
        # float32 makes plotting trivial (already in [-1, 1] range).
        self._recorder: AudioRecorder = AudioRecorder(
            sample_rate=self.RECORD_SAMPLE_RATE,
            dtype="float32",
        )
        self._player: AudioPlayer = AudioPlayer()
        self._state: SignalState = SignalState()

        # ----- Recording state -----
        self._is_recording: bool = False
        self._record_lock: threading.Lock = threading.Lock()
        self._record_chunks: list[np.ndarray] = []   # protected by _record_lock
        self._live_cursor: int = 0                   # index of next chunk to plot

        # ----- Playback state -----
        # Three independent play buttons, one per signal kind. Only one may be
        # active at a time; clicking a second kind stops the first.
        self._is_playing: bool = False
        self._playing_kind: str | None = None        # "orig" | "fft" | "ifft" | None
        self._play_start_perf: float = 0.0
        self._play_duration_s: float = 0.0

        self._build_layout()

    # ==================================================================
    # Layout
    # ==================================================================
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1, minsize=320)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # ---------- LEFT: controls ----------
        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        self.controls_panel = ctk.CTkScrollableFrame(
            self.left_panel, fg_color="transparent"
        )
        self.controls_panel.pack(fill="both", expand=True)

        ctk.CTkLabel(
            self.controls_panel,
            text="FFT Lab: Analise",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w", pady=(0, 15))

        # ----- Card 1: Original -----
        card_orig = ctk.CTkFrame(self.controls_panel, corner_radius=10)
        card_orig.pack(fill="x", pady=(0, 20), ipadx=10, ipady=15)

        lbl_orig = ctk.CTkLabel(
            card_orig, text="1. Sinal Original", font=ctk.CTkFont(weight="bold")
        )
        lbl_orig.pack(anchor="w", padx=15, pady=(5, 10))
        ToolTip(
            lbl_orig,
            "Onda de audio no Dominio do Tempo.\n"
            "Amplitude (volume) variando em segundos.",
        )
        self._draw_mini_graph(card_orig, "wave", self.COLOR_ORIGINAL)
        self._btn_load = ctk.CTkButton(
            card_orig,
            text="Carregar Arquivo",
            fg_color=self.COLOR_ORIGINAL,
            hover_color="#0891b2",
            text_color="white",
            command=self._on_load,
        )
        self._btn_load.pack(fill="x", padx=15, pady=(15, 10))
        # Live record toggle: starts as "Gravar", toggles to "Parar Gravacao".
        self._btn_record = ctk.CTkButton(
            card_orig,
            text="🔴 Gravar (clique pra iniciar)",
            fg_color=("gray85", "gray25"),
            text_color=("black", "white"),
            hover_color=("gray75", "gray35"),
            command=self._on_record_toggle,
        )
        self._btn_record.pack(fill="x", padx=15, pady=(0, 5))

        # ----- Card 2: FFT -----
        card_fft = ctk.CTkFrame(self.controls_panel, corner_radius=10)
        card_fft.pack(fill="x", pady=(0, 20), ipadx=10, ipady=15)
        lbl_fft = ctk.CTkLabel(
            card_fft, text="2. Criptografia (FFT)", font=ctk.CTkFont(weight="bold")
        )
        lbl_fft.pack(anchor="w", padx=15, pady=(5, 10))
        ToolTip(
            lbl_fft,
            "Transformada Rapida de Fourier.\n"
            "Converte tempo em frequencia (espectro).",
        )
        self._draw_mini_graph(card_fft, "bars", self.COLOR_FFT)
        self._btn_apply_fft = ctk.CTkButton(
            card_fft,
            text="Aplicar FFT",
            fg_color=self.COLOR_FFT,
            hover_color="#a21caf",
            text_color="white",
            state="disabled",
            command=self._on_apply_fft,
        )
        self._btn_apply_fft.pack(fill="x", padx=15, pady=(15, 5))

        # ----- Card 3: IFFT -----
        card_ifft = ctk.CTkFrame(self.controls_panel, corner_radius=10)
        card_ifft.pack(fill="x", pady=(0, 20), ipadx=10, ipady=15)
        lbl_ifft = ctk.CTkLabel(
            card_ifft, text="3. Restauracao (IFFT)", font=ctk.CTkFont(weight="bold")
        )
        lbl_ifft.pack(anchor="w", padx=15, pady=(5, 10))
        ToolTip(
            lbl_ifft,
            "Transformada Inversa.\n"
            "Reconstroi o sinal a partir do espectro.",
        )
        self._draw_mini_graph(card_ifft, "noisy_wave", self.COLOR_IFFT)
        self._btn_apply_ifft = ctk.CTkButton(
            card_ifft,
            text="Aplicar IFFT",
            fg_color=self.COLOR_IFFT,
            text_color="white",
            hover_color=self.COLOR_IFFT_HOVER,
            state="disabled",
            command=self._on_apply_ifft,
        )
        self._btn_apply_ifft.pack(fill="x", padx=15, pady=(15, 5))

        # ----- Status -----
        status_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        status_frame.pack(fill="x", pady=(10, 0))
        self._lbl_status = ctk.CTkLabel(
            status_frame,
            text="Status: aguardando audio...",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray50",
            wraplength=300,
            justify="left",
        )
        self._lbl_status.pack(anchor="w")
        self._progress = ctk.CTkProgressBar(
            status_frame, progress_color=self.COLOR_ORIGINAL, height=8
        )
        self._progress.pack(fill="x", pady=(5, 0))
        self._progress.set(0)

        # ---------- RIGHT: visualization ----------
        self.view_panel = ctk.CTkFrame(self, corner_radius=10)
        self.view_panel.grid(row=0, column=1, sticky="nsew")
        self.view_panel.grid_rowconfigure(0, weight=0)   # filters
        self.view_panel.grid_rowconfigure(1, weight=1)   # canvas
        self.view_panel.grid_rowconfigure(2, weight=0)   # player
        self.view_panel.grid_columnconfigure(0, weight=1)

        # ---- Filters row ----
        filters = ctk.CTkFrame(self.view_panel, fg_color="transparent")
        filters.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        filters.grid_columnconfigure(5, weight=1)
        ctk.CTkLabel(
            filters, text="Visualizacao Ativa:", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, padx=(0, 15), sticky="e")

        self._var_orig = ctk.BooleanVar(value=True)
        self._var_fft = ctk.BooleanVar(value=False)
        self._var_ifft = ctk.BooleanVar(value=False)
        # Overlay mode: all visible signals share a single axes, each
        # normalized to its own peak. Useful for comparing shapes (e.g.,
        # showing that Original and IFFT are practically identical).
        self._var_overlay = ctk.BooleanVar(value=False)

        self._chk_orig = ctk.CTkCheckBox(
            filters, text="Original",
            fg_color=self.COLOR_ORIGINAL, text_color=("black", "white"),
            variable=self._var_orig, command=self._on_visibility_changed,
        )
        self._chk_orig.grid(row=0, column=1, padx=10)

        self._chk_fft = ctk.CTkCheckBox(
            filters, text="FFT",
            fg_color=self.COLOR_FFT, text_color=("black", "white"),
            variable=self._var_fft, command=self._on_visibility_changed,
        )
        self._chk_fft.grid(row=0, column=2, padx=10)

        self._chk_ifft = ctk.CTkCheckBox(
            filters, text="IFFT",
            fg_color=self.COLOR_IFFT, text_color=("black", "white"),
            variable=self._var_ifft, command=self._on_visibility_changed,
        )
        self._chk_ifft.grid(row=0, column=3, padx=10)

        self._chk_overlay = ctk.CTkCheckBox(
            filters, text="Sobrepor",
            fg_color="#a3a3a3", text_color=("black", "white"),
            variable=self._var_overlay, command=self._on_visibility_changed,
        )
        self._chk_overlay.grid(row=0, column=4, padx=(20, 10))

        toolbar = ctk.CTkFrame(filters, fg_color="transparent")
        toolbar.grid(row=0, column=5, sticky="e")
        btn_args = {
            "width": 30,
            "fg_color": ("gray80", "gray20"),
            "text_color": ("black", "white"),
            "hover_color": ("gray70", "gray30"),
        }
        ctk.CTkButton(toolbar, text="◀", command=self._on_pan_left, **btn_args).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="▶", command=self._on_pan_right, **btn_args).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="➖", command=self._on_zoom_out, **btn_args).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="➕", command=self._on_zoom_in, **btn_args).pack(side="left", padx=2)

        # ---- Canvas row ----
        canvas_holder = ctk.CTkFrame(
            self.view_panel, corner_radius=5, fg_color=("gray85", "gray10")
        )
        canvas_holder.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
        canvas_holder.grid_rowconfigure(0, weight=1)
        canvas_holder.grid_columnconfigure(0, weight=1)

        self._plot: TriplePlotCanvas = TriplePlotCanvas(canvas_holder)
        self._plot.COLOR_ORIGINAL = self.COLOR_ORIGINAL
        self._plot.COLOR_FFT = self.COLOR_FFT
        self._plot.COLOR_IFFT = self.COLOR_IFFT
        self._plot.widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self._sync_visibility()

        # ---- Player row — one button per signal ----
        player = ctk.CTkFrame(self.view_panel, fg_color="transparent")
        player.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))

        self._btn_play_orig = ctk.CTkButton(
            player, text="▶ Original",
            fg_color=self.COLOR_ORIGINAL, hover_color="#0891b2",
            text_color="white", state="disabled", width=110,
            command=lambda: self._on_play_toggle("orig"),
        )
        self._btn_play_orig.pack(side="left", padx=(0, 5))

        self._btn_play_fft = ctk.CTkButton(
            player, text="▶ FFT",
            fg_color=self.COLOR_FFT, hover_color="#a21caf",
            text_color="white", state="disabled", width=110,
            command=lambda: self._on_play_toggle("fft"),
        )
        self._btn_play_fft.pack(side="left", padx=(0, 5))

        self._btn_play_ifft = ctk.CTkButton(
            player, text="▶ IFFT",
            fg_color=self.COLOR_IFFT, hover_color=self.COLOR_IFFT_HOVER,
            text_color="white", state="disabled", width=110,
            command=lambda: self._on_play_toggle("ifft"),
        )
        self._btn_play_ifft.pack(side="left", padx=(0, 10))

        self._lbl_player_status = ctk.CTkLabel(
            player,
            text="(grave ou carregue um sinal para tocar)",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray50",
        )
        self._lbl_player_status.pack(side="left")

        # Indexable by kind, for compact per-kind UI updates.
        self._play_buttons: dict[str, ctk.CTkButton] = {
            "orig": self._btn_play_orig,
            "fft": self._btn_play_fft,
            "ifft": self._btn_play_ifft,
        }

    # ==================================================================
    # Status helpers
    # ==================================================================
    def _set_status(self, text: str) -> None:
        self._lbl_status.configure(text=f"Status: {text}")
        self.update_idletasks()

    def _set_progress(self, value: float) -> None:
        self._progress.set(max(0.0, min(1.0, value)))

    # ==================================================================
    # Visibility -> canvas
    # ==================================================================
    def _sync_visibility(self) -> None:
        self._plot.set_visibility(
            orig=self._var_orig.get(),
            fft=self._var_fft.get(),
            ifft=self._var_ifft.get(),
        )
        self._plot.set_overlay(self._var_overlay.get())
        # During animation we rebuild the animation figure with the new
        # visibility instead of replacing it with the static plots — that way
        # playback keeps running smoothly. Overlay mode is ignored during
        # animation because the animated FFT panel needs its own axes.
        if self._plot.is_animating:
            self._plot.rebuild_animation_figure()
        else:
            self._plot.render(self._state)

    def _on_visibility_changed(self) -> None:
        self._sync_visibility()

    # ==================================================================
    # Load file
    # ==================================================================
    def _on_load(self) -> None:
        if self._is_recording or self._is_playing:
            return
        path = filedialog.askopenfilename(
            title="Carregar arquivo WAV",
            filetypes=[("Arquivos WAV", "*.wav")],
        )
        if not path:
            return
        try:
            sample_rate, samples_f32 = WavIO.read(path)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"erro ao ler WAV: {exc}")
            return
        self._set_signal(samples_f32, sample_rate, source=Path(path).name)

    # ==================================================================
    # Live recording — toggle button drives this flow
    # ==================================================================
    def _on_record_toggle(self) -> None:
        if self._is_playing:
            # Don't start a recording while audio is playing.
            return
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        # Reset bookkeeping.
        with self._record_lock:
            self._record_chunks = []
        self._live_cursor = 0
        self._is_recording = True

        # UI: morph the button into a stop button and disable other actions.
        self._btn_record.configure(
            text="⏹️ Parar Gravacao",
            fg_color=self.COLOR_RECORD,
            hover_color=self.COLOR_RECORD_HOVER,
            text_color="white",
        )
        self._btn_load.configure(state="disabled")
        self._btn_apply_fft.configure(state="disabled")
        self._btn_apply_ifft.configure(state="disabled")
        for btn in self._play_buttons.values():
            btn.configure(state="disabled")
        self._set_status("gravando ao vivo... clique de novo para parar.")
        self._set_progress(0.0)

        # Switch canvas to live strip-chart mode.
        self._plot.enter_live_mode(
            sample_rate=self._recorder.sample_rate,
            window_seconds=self.LIVE_WINDOW_SECONDS,
        )

        # Start the actual audio capture stream. The callback runs on the
        # audio thread, so it just stashes chunks into a thread-safe list.
        def on_chunk(samples: np.ndarray) -> None:
            with self._record_lock:
                self._record_chunks.append(samples)

        try:
            self._recorder.start_stream(on_chunk)
        except Exception as exc:  # noqa: BLE001
            self._is_recording = False
            self._set_status(f"erro ao iniciar microfone: {exc}")
            self._restore_idle_ui()
            return

        # Kick off the UI poll loop (~30 fps) that drains chunks and updates
        # the strip chart on the main thread.
        self._tick_live()

    def _tick_live(self) -> None:
        if not self._is_recording:
            return

        # Drain any chunks we have not plotted yet.
        with self._record_lock:
            new_chunks = self._record_chunks[self._live_cursor:]
            self._live_cursor = len(self._record_chunks)

        if new_chunks:
            new = np.concatenate(new_chunks)
            self._plot.update_live(new)
            total_samples = self._live_cursor and sum(c.size for c in self._record_chunks)
            duration = total_samples / self._recorder.sample_rate
            self._set_status(f"gravando... duracao atual {duration:.1f}s")

        # Schedule next frame.
        self.after(int(1000 / self.LIVE_FPS), self._tick_live)

    def _stop_recording(self) -> None:
        if not self._is_recording:
            return
        self._is_recording = False

        # Stop the audio stream first; after this call, no more chunks arrive.
        self._recorder.stop_stream()

        # Concatenate everything we received.
        with self._record_lock:
            all_chunks = list(self._record_chunks)
        if not all_chunks:
            self._set_status("nenhum audio capturado.")
            self._plot.exit_live_mode()
            self._restore_idle_ui()
            return

        full_signal = np.concatenate(all_chunks).astype(np.float32)

        # Leave live mode and show the static plots.
        self._plot.exit_live_mode()
        self._restore_idle_ui()
        self._set_signal(
            full_signal, self._recorder.sample_rate, source="microfone (live)"
        )

    def _restore_idle_ui(self) -> None:
        self._btn_record.configure(
            text="🔴 Gravar (clique pra iniciar)",
            fg_color=("gray85", "gray25"),
            hover_color=("gray75", "gray35"),
            text_color=("black", "white"),
        )
        self._btn_load.configure(state="normal")

    # ==================================================================
    # Common signal-loaded code path (used by both file load and live record)
    # ==================================================================
    def _set_signal(
        self,
        samples: np.ndarray,
        sample_rate: int,
        source: str,
    ) -> None:
        self._state.clear()
        self._state.time_signal = samples.astype(np.float32)
        self._state.sample_rate = sample_rate
        self._set_status(
            f"sinal carregado de '{source}': {samples.size} amostras "
            f"@ {sample_rate} Hz ({samples.size / sample_rate:.2f}s)"
        )
        self._set_progress(1.0)
        self._btn_apply_fft.configure(state="normal")
        self._btn_apply_ifft.configure(state="disabled")
        # Only the Original play button is meaningful right now; FFT/IFFT
        # become playable after the user applies the corresponding transform.
        self._btn_play_orig.configure(state="normal")
        self._btn_play_fft.configure(state="disabled")
        self._btn_play_ifft.configure(state="disabled")
        self._lbl_player_status.configure(
            text=f"pronto ({samples.size / sample_rate:.2f}s)"
        )
        self._var_orig.set(True)
        self._var_fft.set(False)
        self._var_ifft.set(False)
        self._sync_visibility()

    # ==================================================================
    # FFT / IFFT
    # ==================================================================
    def _on_apply_fft(self) -> None:
        if not self._state.has_time:
            self._set_status("carregue ou grave um sinal antes.")
            return
        self._state.freq_spectrum = np.fft.fft(self._state.time_signal)
        self._set_status(
            f"FFT aplicada — {self._state.freq_spectrum.size} bins, "
            f"resolucao {self._state.sample_rate / self._state.freq_spectrum.size:.2f} Hz/bin"
        )
        self._btn_apply_ifft.configure(state="normal")
        self._btn_play_fft.configure(state="normal")
        self._var_fft.set(True)
        self._sync_visibility()

    def _on_apply_ifft(self) -> None:
        if not self._state.has_freq:
            self._set_status("aplique a FFT antes da IFFT.")
            return
        recovered = np.fft.ifft(self._state.freq_spectrum).real
        self._state.recovered_signal = recovered.astype(np.float32)
        max_err = float(
            np.max(np.abs(self._state.recovered_signal - self._state.time_signal))
        )
        self._set_status(
            f"IFFT aplicada — erro maximo vs original: {max_err:.2e} "
            "(precisao do float64)"
        )
        self._btn_play_ifft.configure(state="normal")
        self._var_ifft.set(True)
        self._sync_visibility()

    # ==================================================================
    # FFT-as-audio sonification — pure |FFT|, no cryptography of any kind
    # ==================================================================
    def _build_fft_audio(self) -> np.ndarray:
        """Convert the FFT into playable audio by sonifying the magnitudes.

        This is the LITERAL interpretation of "play the purple wave":

            audio = |FFT(time_signal)|   <-- exactly what the FFT plot draws

        No phase manipulation. No encryption. No transformations beyond what
        is mathematically required to push positive-valued magnitudes through
        a soundcard:

            1. Center around zero (subtract the mean) — without this step the
               output is a constant DC level, which is inaudible.
            2. Normalize to [-0.5, 0.5] so the output never clips.

        Pedagogical point: the FFT is an array of N numbers. If you reinter-
        pret that array as audio samples and play it at the original sample
        rate, you hear the *shape* of the spectrum directly. The result is
        weird because the FFT is NOT an audio signal — it's frequency-domain
        data. This is exactly the lesson: time-domain and frequency-domain
        are two different views; mistaking one for the other gives nonsense.

        What you will perceive when playing voice
        -----------------------------------------
        The output sounds like a burst at the START and another burst at the
        END, with near-silence in between. This is a real mathematical
        property — conjugate symmetry — not a bug:

            For any real-valued input x[n], the FFT satisfies
                |X[N - k]| = |X[k]|
            So |FFT(x)| is mirror-symmetric around the middle of the array.
            Voice has energy concentrated in low frequencies (low bins ≈
            START of the array; their conjugate mirror is at HIGH bins ≈
            END of the array). The middle bins correspond to high
            frequencies where voice has little energy → silence in the
            middle of the playback.

        This effect is itself a nice teachable moment about the structure of
        the Fourier transform.
        """
        if not self._state.has_freq:
            raise RuntimeError("freq_spectrum not yet computed.")
        magnitudes = np.abs(self._state.freq_spectrum).astype(np.float32)

        # Step 1: subtract the mean. The magnitudes are all >= 0, so without
        # this the output is a positive DC level (no audible content).
        centered = magnitudes - float(magnitudes.mean())

        # Step 2: peak-normalize to a safe playback range.
        peak = float(np.max(np.abs(centered))) or 1.0
        return (centered / peak * 0.5).astype(np.float32)

    def _get_audio_for(self, kind: str) -> np.ndarray:
        """Return the audio buffer associated with one of the play buttons."""
        if kind == "orig":
            return self._state.time_signal
        if kind == "ifft":
            return self._state.recovered_signal
        if kind == "fft":
            return self._build_fft_audio()
        raise ValueError(f"unknown play kind: {kind!r}")

    # ==================================================================
    # Playback (one button per signal — orig / fft / ifft)
    # ==================================================================
    def _on_play_toggle(self, kind: str) -> None:
        """Click on any play button. Stops current playback or starts new one."""
        if self._is_recording:
            return

        if self._is_playing:
            # Same button as the one playing → user wants to stop.
            if self._playing_kind == kind:
                self._stop_playback()
                return
            # Different kind: stop current, then start the new one.
            self._stop_playback()

        self._start_playback(kind)

    def _start_playback(self, kind: str) -> None:
        """Two-phase start: precompute STFT in a thread, then animate + play."""
        if not self._state.has_time:
            return
        # Sanity check for FFT/IFFT — make sure the data exists.
        if kind == "fft" and not self._state.has_freq:
            return
        if kind == "ifft" and not self._state.has_recovered:
            return

        try:
            audio = self._get_audio_for(kind)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"erro ao montar audio ({kind}): {exc}")
            return

        self._playing_kind = kind

        # Phase 1: heavy STFT precompute on a worker thread.
        self._set_status(f"preparando animacao para '{kind}' (calculando STFT)...")
        self._lbl_player_status.configure(text="preparando...")
        for btn in self._play_buttons.values():
            btn.configure(state="disabled")
        self._btn_record.configure(state="disabled")
        self._btn_load.configure(state="disabled")
        self._btn_apply_fft.configure(state="disabled")
        self._btn_apply_ifft.configure(state="disabled")

        state = self._state

        def worker() -> None:
            try:
                anim_data = self._plot.compute_animation_data(
                    state,
                    playing_signal=audio,
                    window_seconds=2.0,
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._set_status(
                    f"erro ao preparar animacao: {exc}"
                ))
                self.after(0, self._reset_play_after_error)
                return
            self.after(0, lambda: self._begin_playback_after_prep(kind, audio, anim_data))

        threading.Thread(target=worker, name="fftlab-stft", daemon=True).start()

    def _reset_play_after_error(self) -> None:
        self._playing_kind = None
        # Restore play buttons to their natural availability based on state.
        self._btn_play_orig.configure(state=("normal" if self._state.has_time else "disabled"))
        self._btn_play_fft.configure(state=("normal" if self._state.has_freq else "disabled"))
        self._btn_play_ifft.configure(state=("normal" if self._state.has_recovered else "disabled"))
        self._reset_all_play_buttons_to_play()
        self._btn_record.configure(state="normal")
        self._btn_load.configure(state="normal")
        if self._state.has_time:
            self._btn_apply_fft.configure(state="normal")
        if self._state.has_freq:
            self._btn_apply_ifft.configure(state="normal")

    def _begin_playback_after_prep(
        self, kind: str, audio: np.ndarray, anim_data: dict,
    ) -> None:
        """Phase 2: install the precomputed data and start the audio + ticker."""
        sr = self._state.sample_rate
        self._play_duration_s = audio.size / sr
        self._is_playing = True

        # Switch canvas to animation mode (matplotlib lines built on UI thread).
        self._plot.setup_animation(anim_data)

        # UI: morph the *active* play button into a stop button, leave the
        # other two disabled so the user can't fire a second playback.
        for k, btn in self._play_buttons.items():
            if k == kind:
                btn.configure(
                    state="normal",
                    text="⏹ Parar",
                    fg_color="#ef4444",
                    hover_color="#dc2626",
                )
            else:
                btn.configure(state="disabled")

        label = {"orig": "Original", "fft": "FFT", "ifft": "IFFT"}[kind]
        self._lbl_player_status.configure(
            text=f"tocando {label} ({self._play_duration_s:.2f}s)..."
        )

        # Kick off playback. Use perf_counter for cursor timing (monotonic).
        try:
            self._player.play(audio, sample_rate=sr, blocking=False)
        except Exception as exc:  # noqa: BLE001
            self._is_playing = False
            self._set_status(f"erro ao tocar: {exc}")
            self._plot.exit_animation_mode()
            self._sync_visibility()
            self._reset_all_play_buttons_to_play()
            self._reset_play_after_error()
            return

        self._play_start_perf = time.perf_counter()
        self._tick_animation()

    def _tick_animation(self) -> None:
        if not self._is_playing:
            return
        elapsed = time.perf_counter() - self._play_start_perf
        if elapsed >= self._play_duration_s:
            self._stop_playback()
            return
        self._plot.update_animation_frame(elapsed)
        self.after(int(1000 / self.PLAYBACK_FPS), self._tick_animation)

    def _stop_playback(self) -> None:
        if not self._is_playing:
            return
        self._is_playing = False
        self._playing_kind = None
        try:
            self._player.stop()
        except Exception:  # noqa: BLE001
            pass
        self._plot.exit_animation_mode()
        self._sync_visibility()   # back to static plots
        self._reset_all_play_buttons_to_play()
        # Re-enable controls that were disabled during prepare/playback.
        self._btn_record.configure(state="normal")
        self._btn_load.configure(state="normal")
        if self._state.has_time:
            self._btn_apply_fft.configure(state="normal")
        if self._state.has_freq:
            self._btn_apply_ifft.configure(state="normal")
        self._lbl_player_status.configure(
            text=f"pronto ({self._play_duration_s:.2f}s)"
        )

    def _reset_all_play_buttons_to_play(self) -> None:
        """Put all three play buttons back to their idle (▶ ...) appearance."""
        self._btn_play_orig.configure(
            text="▶ Original", fg_color=self.COLOR_ORIGINAL, hover_color="#0891b2",
            state=("normal" if self._state.has_time else "disabled"),
        )
        self._btn_play_fft.configure(
            text="▶ FFT", fg_color=self.COLOR_FFT, hover_color="#a21caf",
            state=("normal" if self._state.has_freq else "disabled"),
        )
        self._btn_play_ifft.configure(
            text="▶ IFFT", fg_color=self.COLOR_IFFT, hover_color=self.COLOR_IFFT_HOVER,
            state=("normal" if self._state.has_recovered else "disabled"),
        )

    # ==================================================================
    # Toolbar — pan/zoom on the time axis
    # ==================================================================
    def _adjust_xlim(self, factor_zoom: float, shift_fraction: float) -> None:
        any_axes = False
        for ax in self._plot._figure.get_axes():
            any_axes = True
            x0, x1 = ax.get_xlim()
            mid = (x0 + x1) / 2.0
            half = (x1 - x0) / 2.0 * factor_zoom
            mid_shifted = mid + (x1 - x0) * shift_fraction
            ax.set_xlim(mid_shifted - half, mid_shifted + half)
        if any_axes:
            self._plot._mpl_canvas.draw_idle()

    def _on_zoom_in(self) -> None:
        self._adjust_xlim(factor_zoom=0.5, shift_fraction=0.0)

    def _on_zoom_out(self) -> None:
        self._adjust_xlim(factor_zoom=2.0, shift_fraction=0.0)

    def _on_pan_left(self) -> None:
        self._adjust_xlim(factor_zoom=1.0, shift_fraction=-0.2)

    def _on_pan_right(self) -> None:
        self._adjust_xlim(factor_zoom=1.0, shift_fraction=+0.2)

    # ==================================================================
    # Mini graph (kept for the left-panel cards; purely decorative)
    # ==================================================================
    def _draw_mini_graph(self, parent, graph_type: str, color_hex: str) -> None:
        bg_color = self._apply_appearance_mode(["#EAEAEA", "#2D2D2E"])
        wrapper = ctk.CTkFrame(parent, fg_color=bg_color, height=60, corner_radius=5)
        wrapper.pack(fill="x", padx=15, pady=5)
        wrapper.pack_propagate(False)
        canvas = tk.Canvas(wrapper, bg=bg_color, highlightthickness=0, height=60)
        canvas.pack(fill="both", expand=True, padx=5, pady=5)
        width = 250
        if graph_type == "wave":
            points = []
            for x in range(width):
                y = 25 + math.sin(x * 0.08) * 15
                points.extend([x, y])
            canvas.create_line(points, fill=color_hex, width=2, smooth=True)
        elif graph_type == "bars":
            for i in range(10, width - 10, 8):
                height = 5 if i % 40 == 0 else (25 if i % 25 == 0 else 12)
                canvas.create_rectangle(i, 50 - height, i + 4, 50, fill=color_hex, outline="")
        elif graph_type == "noisy_wave":
            points = []
            for x in range(width):
                y = 25 + math.sin(x * 0.08) * 15 + (math.sin(x * 0.9) * 3)
                points.extend([x, y])
            canvas.create_line(points, fill=color_hex, width=2, dash=(2, 2))
