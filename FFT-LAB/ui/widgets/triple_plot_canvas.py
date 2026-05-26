"""Matplotlib widget that plots the three signals of the FFT Lab.

Two display modes:

    1. STATIC mode (default) — three subplots showing time, FFT spectrum and
       IFFT reconstruction, with checkboxes controlling visibility. Use
       `set_visibility(...)` + `render(state)`.

    2. LIVE mode — a single time-domain strip chart that scrolls left as new
       samples arrive. Use `enter_live_mode(sr)`, `update_live(samples)`,
       `exit_live_mode()` to return to static mode.

A vertical PLAYBACK CURSOR can be drawn on the time-domain subplots while
audio plays back, via `set_cursor(time_s)` / `clear_cursor()`.

Naming caveat: we expose `mpl_canvas` instead of the more natural `canvas`
because `customtkinter.CTkFrame` has its own private `_canvas` attribute used
internally for rounded-corner rendering. Naming our matplotlib canvas the
same way would collide and crash on resize events.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import signal as sps

from ui.widgets.signal_state import SignalState

matplotlib.use("Agg")  # backend-agnostic; we embed via FigureCanvasTkAgg


class TriplePlotCanvas:
    """Matplotlib canvas showing time signal, FFT spectrum, and IFFT reconstruction.

    Static usage:
        canvas = TriplePlotCanvas(parent_frame)
        canvas.widget.grid(row=1, column=0, sticky="nsew")
        canvas.set_visibility(orig=True, fft=True, ifft=False)
        canvas.render(signal_state)

    Live usage (during a recording):
        canvas.enter_live_mode(sample_rate=16000, window_seconds=3.0)
        # inside a timer:
        canvas.update_live(new_samples)
        # when done:
        canvas.exit_live_mode()
        canvas.render(state)  # show the final result
    """

    COLOR_ORIGINAL: str = "#06b6d4"
    COLOR_FFT: str = "#c026d3"
    COLOR_IFFT: str = "#f97316"
    COLOR_CURSOR: str = "#facc15"   # yellow vertical line

    BG_COLOR: str = "#1a1a1a"
    AXES_BG: str = "#0d0d0d"
    TEXT_COLOR: str = "white"
    GRID_COLOR: str = "#333333"

    def __init__(
        self,
        parent,
        *,
        figsize: tuple[float, float] = (8, 5),
        dpi: int = 100,
    ) -> None:
        self._figure: Figure = Figure(figsize=figsize, dpi=dpi, facecolor=self.BG_COLOR)
        self._mpl_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(
            self._figure, master=parent
        )

        # Static-mode visibility flags (controlled by checkboxes).
        self._show_orig: bool = True
        self._show_fft: bool = False
        self._show_ifft: bool = False
        self._overlay_mode: bool = False  # if True, draw all visible kinds on one axes

        # Cursor state — one Line2D per time-domain subplot (or None for freq).
        self._cursor_lines: list = []

        # Live-mode state.
        self._live_mode: bool = False
        self._live_buffer: np.ndarray | None = None
        self._live_line = None  # type: ignore[assignment]
        self._live_sample_rate: int | None = None

        # Animation-mode state (playback with moving wave + STFT).
        self._anim_mode: bool = False
        self._anim_data: dict | None = None
        self._anim_lines: dict = {}     # kind -> Line2D
        self._anim_axes: dict = {}      # kind -> Axes

        self._draw_placeholder()

    # ==================================================================
    # Public surface
    # ==================================================================
    @property
    def widget(self):
        """Return the underlying tk widget for grid/pack placement."""
        return self._mpl_canvas.get_tk_widget()

    def set_visibility(self, *, orig: bool, fft: bool, ifft: bool) -> None:
        """Configure which signals will be drawn on the next static render."""
        self._show_orig = orig
        self._show_fft = fft
        self._show_ifft = ifft

    def set_overlay(self, overlay: bool) -> None:
        """Toggle overlay mode (all visible signals drawn on a single axes)."""
        self._overlay_mode = bool(overlay)

    def render(self, state: SignalState) -> None:
        """Redraw the canvas based on `state` and the current visibility flags.

        Has no effect while in live mode (strip chart wins) or animation mode
        (the playback animation owns the figure). Caller code should call
        `rebuild_animation_figure()` instead when in animation mode.
        """
        if self._live_mode or self._anim_mode:
            return

        plots: list[str] = []
        if self._show_orig and state.has_time:
            plots.append("orig")
        if self._show_fft and state.has_freq:
            plots.append("fft")
        if self._show_ifft and state.has_recovered:
            plots.append("ifft")

        self._figure.clear()
        self._figure.set_facecolor(self.BG_COLOR)
        self._cursor_lines = []  # cleared because all axes were cleared

        if not plots:
            self._draw_placeholder()
            self._mpl_canvas.draw_idle()
            return

        # Branch: overlay (single axes) vs stacked (one subplot per signal).
        if self._overlay_mode and len(plots) >= 2:
            self._render_overlay(state, plots)
            return

        n = len(plots)
        for i, kind in enumerate(plots, start=1):
            ax = self._figure.add_subplot(n, 1, i)
            self._style_axes(ax)
            if kind == "orig":
                self._plot_time(
                    ax, state.time_signal, state.sample_rate,
                    title="Sinal Original (tempo)", color=self.COLOR_ORIGINAL,
                )
                # Cursor will live on this time-domain plot.
                cursor = ax.axvline(
                    0, color=self.COLOR_CURSOR, linewidth=1.2, alpha=0.0
                )
                self._cursor_lines.append(cursor)
            elif kind == "fft":
                self._plot_freq(
                    ax, state.freq_spectrum, state.sample_rate,
                    title="FFT (espectro de frequencia)", color=self.COLOR_FFT,
                )
                self._cursor_lines.append(None)  # no cursor on freq axis
            elif kind == "ifft":
                self._plot_time(
                    ax, state.recovered_signal, state.sample_rate,
                    title="IFFT (sinal reconstruido)", color=self.COLOR_IFFT,
                )
                cursor = ax.axvline(
                    0, color=self.COLOR_CURSOR, linewidth=1.2, alpha=0.0
                )
                self._cursor_lines.append(cursor)

        self._figure.tight_layout()
        self._mpl_canvas.draw_idle()

    def _render_overlay(self, state: SignalState, plots: list[str]) -> None:
        """Draw all visible signals on a single axes, normalized for comparison.

        Each signal is normalized to its own peak (so it fits in [-1, 1]) and
        plotted against a unitless 'posicao normalizada' x-axis (0 to 1).

        Caveat (worth stating in the relatorio): the x-axis is NOT time for
        the FFT line — it is array index normalized. Comparing the shapes is
        legitimate (e.g., to see that Original and IFFT coincide), but the
        x-axis values are not physically meaningful when mixed-domain.
        """
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)

        for kind in plots:
            if kind == "orig":
                samples = state.time_signal
                color = self.COLOR_ORIGINAL
                label = "Original"
            elif kind == "fft":
                samples = np.abs(state.freq_spectrum)
                color = self.COLOR_FFT
                label = "|FFT|"
            elif kind == "ifft":
                samples = state.recovered_signal
                color = self.COLOR_IFFT
                label = "IFFT"
            else:
                continue

            samples = np.asarray(samples, dtype=np.float64)
            # Downsample to keep redraws snappy.
            n = samples.size
            n_plot = min(n, 4000)
            if n > n_plot:
                idx = np.linspace(0, n - 1, n_plot).astype(int)
                samples = samples[idx]

            peak = float(np.max(np.abs(samples))) or 1.0
            normalized = samples / peak
            x = np.linspace(0.0, 1.0, samples.size)
            ax.plot(x, normalized, color=color, linewidth=0.9, alpha=0.75, label=label)

        ax.set_xlim(0, 1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel(
            "posicao normalizada (cada onda mapeada para [0, 1])",
            color=self.TEXT_COLOR, fontsize=8,
        )
        ax.set_ylabel(
            "amplitude normalizada (cada onda escalada para [-1, 1])",
            color=self.TEXT_COLOR, fontsize=8,
        )
        ax.set_title(
            "Modo Sobreposto — todas as ondas no mesmo eixo",
            color=self.TEXT_COLOR, fontsize=10,
        )

        legend = ax.legend(
            loc="upper right",
            facecolor=self.AXES_BG,
            edgecolor="gray",
            labelcolor=self.TEXT_COLOR,
            fontsize=9,
        )
        legend.get_frame().set_alpha(0.85)

        self._figure.tight_layout()
        self._mpl_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Playback cursor
    # ------------------------------------------------------------------
    def set_cursor(self, time_s: float) -> None:
        """Move the playback cursor to `time_s` on every time-domain subplot."""
        if not self._cursor_lines:
            return
        for line in self._cursor_lines:
            if line is None:
                continue
            line.set_xdata([time_s, time_s])
            line.set_alpha(0.8)
        self._mpl_canvas.draw_idle()

    def clear_cursor(self) -> None:
        """Hide the playback cursor (does not remove the Line objects)."""
        if not self._cursor_lines:
            return
        for line in self._cursor_lines:
            if line is None:
                continue
            line.set_alpha(0.0)
        self._mpl_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Animation mode (during playback)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_animation_data(
        state: SignalState,
        *,
        playing_signal: np.ndarray | None = None,
        window_seconds: float = 2.0,
        stft_window_s: float = 0.05,
        stft_hop_s: float = 0.01,
    ) -> dict:
        """Precompute the STFT and helper arrays. SAFE TO CALL FROM A THREAD.

        Does NOT touch matplotlib — pure numpy/scipy. The result is a plain
        dict that can be passed to `setup_animation` on the UI thread.

        Parameters
        ----------
        state : SignalState
            Must have at least `time_signal` + `sample_rate`. If
            `recovered_signal` is None, the IFFT plot will be skipped.
        playing_signal : np.ndarray | None
            The signal whose STFT will drive the FFT plot animation and whose
            duration is used as the playback timeline. Defaults to
            `state.time_signal` (i.e., playing the original). Pass a different
            signal (e.g., the FFT-as-audio) to animate based on it.
        window_seconds : float
            Width of the scrolling time-domain window during animation.
        stft_window_s : float
            STFT analysis window length in seconds. 50 ms is a good default —
            short enough to "follow" voice transitions, long enough for ~20 Hz
            frequency resolution.
        stft_hop_s : float
            STFT hop in seconds. 10 ms → 100 frames per second of audio,
            far above any FPS we'd render at.
        """
        if state.time_signal is None or state.sample_rate is None:
            raise ValueError("state must have time_signal and sample_rate.")
        sr = state.sample_rate

        signal_for_stft = (
            playing_signal if playing_signal is not None else state.time_signal
        )

        nperseg = max(8, int(stft_window_s * sr))
        hop_samples = max(1, int(stft_hop_s * sr))
        noverlap = max(0, nperseg - hop_samples)

        f, t_stft, Zxx = sps.stft(
            signal_for_stft.astype(np.float64),
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )
        # scipy.signal.stft already normalizes by the window sum, so |Zxx|
        # is on the same scale as the input signal amplitude. Do NOT divide
        # by nperseg again — that would make magnitudes ~800x too small.
        magnitudes = np.abs(Zxx).astype(np.float32)

        return {
            "time_signal": state.time_signal.astype(np.float32),
            "recovered_signal": (
                state.recovered_signal.astype(np.float32)
                if state.recovered_signal is not None else None
            ),
            "sample_rate": sr,
            "duration_s": signal_for_stft.size / sr,
            "window_seconds": float(window_seconds),
            "stft_mag": magnitudes,
            "stft_freqs": f,
            "stft_times": t_stft,
            "stft_max": float(magnitudes.max()) if magnitudes.size else 0.0,
        }

    def setup_animation(self, anim_data: dict) -> None:
        """Build the matplotlib figure with pre-cached lines for fast updates.

        MUST run on the UI thread. After this returns, the figure is ready and
        you can call `update_animation_frame(t)` at high frequency.

        Respects the current visibility flags (`set_visibility(...)`).
        """
        self._anim_mode = True
        self._anim_data = anim_data
        self._anim_lines = {}
        self._anim_axes = {}

        plots: list[str] = []
        has_recovered = anim_data.get("recovered_signal") is not None
        if self._show_orig:
            plots.append("orig")
        if self._show_fft:
            plots.append("fft")
        if self._show_ifft and has_recovered:
            plots.append("ifft")

        self._figure.clear()
        self._figure.set_facecolor(self.BG_COLOR)

        if not plots:
            self._draw_placeholder()
            self._mpl_canvas.draw_idle()
            return

        sr = anim_data["sample_rate"]
        win = anim_data["window_seconds"]
        time_signal = anim_data["time_signal"]
        recovered = anim_data["recovered_signal"]

        n = len(plots)
        for i, kind in enumerate(plots, start=1):
            ax = self._figure.add_subplot(n, 1, i)
            self._style_axes(ax)
            if kind == "orig":
                self._draw_animated_time_axis(
                    ax, time_signal, sr,
                    color=self.COLOR_ORIGINAL,
                    title="Sinal Original (tempo, janela deslizante)",
                )
                self._anim_axes[kind] = ax
            elif kind == "ifft":
                self._draw_animated_time_axis(
                    ax, recovered, sr,
                    color=self.COLOR_IFFT,
                    title="IFFT (sinal reconstruido, janela deslizante)",
                )
                self._anim_axes[kind] = ax
            elif kind == "fft":
                # The FFT plot is a single line whose y values are swapped at
                # each frame to the appropriate STFT column. x stays fixed
                # (frequency bins).
                freqs = anim_data["stft_freqs"]
                initial = anim_data["stft_mag"][:, 0] if anim_data["stft_mag"].size else np.zeros_like(freqs)
                (line,) = ax.plot(
                    freqs, initial, color=self.COLOR_FFT, linewidth=0.9
                )
                ymax = max(anim_data["stft_max"] * 1.1, 1e-6)
                ax.set_ylim(0, ymax)
                ax.set_xlim(0, sr / 2)
                ax.set_xlabel("frequencia (Hz)", color=self.TEXT_COLOR, fontsize=8)
                ax.set_ylabel("|X(f, t)|", color=self.TEXT_COLOR, fontsize=8)
                ax.set_title(
                    "FFT instantanea (STFT) — espectro no momento atual",
                    color=self.TEXT_COLOR, fontsize=10,
                )
                self._anim_lines["fft"] = line
                self._anim_axes["fft"] = ax

        # Initial time window: [0, win] for all time-domain axes.
        for kind in ("orig", "ifft"):
            if kind in self._anim_axes:
                self._anim_axes[kind].set_xlim(0, win)

        self._figure.tight_layout()
        self._mpl_canvas.draw_idle()

    def _draw_animated_time_axis(
        self,
        ax,
        samples: np.ndarray,
        sample_rate: int,
        color: str,
        title: str,
    ) -> None:
        """Plot the FULL waveform once. Scrolling will be done by changing xlim."""
        n = samples.size
        n_plot = min(n, 6000)        # cap to keep redraws fast
        if n > n_plot:
            idx = np.linspace(0, n - 1, n_plot).astype(int)
            t = idx / sample_rate
            y = samples[idx]
        else:
            t = np.arange(n) / sample_rate
            y = samples
        ax.plot(t, y, color=color, linewidth=0.9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("tempo (s)", color=self.TEXT_COLOR, fontsize=8)
        ax.set_ylabel("amplitude", color=self.TEXT_COLOR, fontsize=8)
        ax.set_title(title, color=self.TEXT_COLOR, fontsize=10)

    def update_animation_frame(self, t: float) -> None:
        """Advance the animation to playback time `t` (seconds since start)."""
        if not self._anim_mode or self._anim_data is None:
            return

        win = self._anim_data["window_seconds"]
        duration = self._anim_data["duration_s"]
        half = win / 2.0

        # Centered scrolling window, but clamped to [0, duration] so the
        # axis never extends beyond the signal on either end.
        win_start = t - half
        win_end = t + half
        if win_start < 0:
            win_start, win_end = 0.0, min(duration, win)
        elif win_end > duration:
            win_start, win_end = max(0.0, duration - win), duration

        for kind in ("orig", "ifft"):
            if kind in self._anim_axes:
                self._anim_axes[kind].set_xlim(win_start, win_end)

        # FFT: find the STFT frame closest to `t` and swap y-data.
        if "fft" in self._anim_lines:
            stft_times = self._anim_data["stft_times"]
            mag = self._anim_data["stft_mag"]
            if mag.size:
                idx = int(np.searchsorted(stft_times, t))
                idx = max(0, min(idx, mag.shape[1] - 1))
                self._anim_lines["fft"].set_ydata(mag[:, idx])

        self._mpl_canvas.draw_idle()

    def exit_animation_mode(self) -> None:
        """Leave animation mode. Caller should follow with `render(state)`."""
        self._anim_mode = False
        self._anim_data = None
        self._anim_lines = {}
        self._anim_axes = {}

    def rebuild_animation_figure(self) -> None:
        """Re-run setup_animation with the cached data and current visibility.

        Useful when the user toggles a checkbox during playback — the figure
        needs to be rebuilt to honor the new visibility, but the heavy STFT
        computation can be reused.
        """
        if self._anim_data is None:
            return
        self.setup_animation(self._anim_data)

    @property
    def is_animating(self) -> bool:
        return self._anim_mode

    # ------------------------------------------------------------------
    # Live (strip chart) mode
    # ------------------------------------------------------------------
    def enter_live_mode(
        self,
        sample_rate: int,
        window_seconds: float = 3.0,
    ) -> None:
        """Switch to a single-plot strip chart that scrolls left as data arrives.

        The buffer is pre-allocated to `window_seconds * sample_rate` samples
        and only the most recent samples are shown. Designed for ~30 fps
        updates.
        """
        self._live_mode = True
        self._live_sample_rate = sample_rate
        n = int(window_seconds * sample_rate)
        self._live_buffer = np.zeros(n, dtype=np.float32)

        self._figure.clear()
        self._figure.set_facecolor(self.BG_COLOR)
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        ax.set_xlim(0, window_seconds)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            "Gravando ao vivo... (clique Parar para finalizar)",
            color=self.TEXT_COLOR,
            fontsize=11,
        )
        ax.set_xlabel("tempo (s) — janela deslizante", color=self.TEXT_COLOR, fontsize=9)
        ax.set_ylabel("amplitude", color=self.TEXT_COLOR, fontsize=9)

        t = np.arange(n) / sample_rate
        (self._live_line,) = ax.plot(
            t, self._live_buffer, color=self.COLOR_ORIGINAL, linewidth=0.9
        )
        self._figure.tight_layout()
        self._mpl_canvas.draw_idle()

    def update_live(self, new_samples: np.ndarray) -> None:
        """Push `new_samples` into the strip chart and redraw."""
        if not self._live_mode or self._live_buffer is None or new_samples.size == 0:
            return
        n = new_samples.size
        cap = self._live_buffer.size
        if n >= cap:
            # New batch is larger than the window — keep only the tail.
            self._live_buffer[:] = new_samples[-cap:].astype(np.float32)
        else:
            # Shift left by n samples, append new ones at the end.
            self._live_buffer[:-n] = self._live_buffer[n:]
            self._live_buffer[-n:] = new_samples.astype(np.float32)
        self._live_line.set_ydata(self._live_buffer)
        self._mpl_canvas.draw_idle()

    def exit_live_mode(self) -> None:
        """Leave live mode. Caller should follow up with `render(state)`."""
        self._live_mode = False
        self._live_buffer = None
        self._live_line = None
        self._live_sample_rate = None

    @property
    def is_live(self) -> bool:
        return self._live_mode

    # ==================================================================
    # Drawing helpers (private)
    # ==================================================================
    def _draw_placeholder(self) -> None:
        ax = self._figure.add_subplot(111)
        self._style_axes(ax)
        ax.text(
            0.5, 0.5,
            "Carregue um arquivo WAV ou grave audio\n"
            "depois marque Original / FFT / IFFT acima",
            color="gray",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_time(
        self,
        ax,
        samples: np.ndarray,
        sample_rate: int,
        title: str,
        color: str,
    ) -> None:
        # Downsample to ~4000 points for plotting if the signal is huge —
        # the human eye cannot perceive more, and matplotlib gets slow
        # otherwise. We pick uniform indices to keep the waveform shape.
        n_plot = min(samples.size, 4000)
        if samples.size > n_plot:
            idx = np.linspace(0, samples.size - 1, n_plot).astype(int)
            t = idx / sample_rate
            y = samples[idx]
        else:
            t = np.arange(samples.size) / sample_rate
            y = samples
        ax.plot(t, y, color=color, linewidth=0.9)
        ax.set_xlabel("tempo (s)", color=self.TEXT_COLOR, fontsize=8)
        ax.set_ylabel("amplitude", color=self.TEXT_COLOR, fontsize=8)
        ax.set_title(title, color=self.TEXT_COLOR, fontsize=10)
        ax.set_xlim(0, t[-1] if t.size else 1)

    def _plot_freq(
        self,
        ax,
        spectrum: np.ndarray,
        sample_rate: int,
        title: str,
        color: str,
    ) -> None:
        # Single-sided amplitude spectrum: only bins 0..N/2.
        n = spectrum.size
        half = n // 2 + 1
        magnitude = np.abs(spectrum[:half]) / n        # normalize
        freqs = np.linspace(0, sample_rate / 2, half)

        ax.plot(freqs, magnitude, color=color, linewidth=0.9)
        ax.set_xlabel("frequencia (Hz)", color=self.TEXT_COLOR, fontsize=8)
        ax.set_ylabel("|X(f)|", color=self.TEXT_COLOR, fontsize=8)
        ax.set_title(title, color=self.TEXT_COLOR, fontsize=10)
        ax.set_xlim(0, sample_rate / 2)

    def _style_axes(self, ax) -> None:
        ax.set_facecolor(self.AXES_BG)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("gray")
        ax.grid(True, color=self.GRID_COLOR, linewidth=0.3, alpha=0.5)
