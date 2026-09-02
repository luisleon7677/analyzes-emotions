"""Interfaz gráfica: análisis emocional de audio por fragmentos."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
import sounddevice as sd
import torchaudio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from analyzer import DEFAULT_CHUNK_SECONDS, EmotionAnalyzer

BG = "#1e1e1e"
PANEL = "#2a2a2a"
ACCENT = "#3d8bfd"
TEXT = "#f0f0f0"
MUTED = "#9a9a9a"
PLAYHEAD = "#ffffff"

# Colores por emoción del modelo español UMUTeam
EMOTION_COLORS = {
    "sadness": "#4da3ff",
    "fear": "#9b59b6",
    "disgust": "#1abc9c",
    "anger": "#e74c3c",
    "neutral": "#ffd43b",
    "joy": "#51cf66",
}

EMOTION_LEGEND = (
    ("Triste", "sadness"),
    ("Miedo", "fear"),
    ("Disgusto", "disgust"),
    ("Enojo", "anger"),
    ("Neutral", "neutral"),
    ("Alegre", "joy"),
)


def color_for_result(dominant: str, valence_pct: float) -> str:
    key = dominant.lower()
    if key in EMOTION_COLORS:
        return EMOTION_COLORS[key]
    if valence_pct < 45:
        return EMOTION_COLORS["sadness"]
    if valence_pct < 58:
        return EMOTION_COLORS["neutral"]
    return EMOTION_COLORS["joy"]


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


class EmotionApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Análisis emocional de audio")
        self.geometry("1000x780")
        self.minsize(860, 660)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=BG)

        self.audio_path: str | None = None
        self.analyzer: EmotionAnalyzer | None = None
        self._busy = False
        self._results: list = []
        self._duration = 0.0
        self._playing = False
        self._seek_pos = 0.0
        self._play_anchor_mono = 0.0
        self._play_anchor_pos = 0.0
        self._playhead_line: Line2D | None = None
        self._updating_slider = False
        self._tick_job: str | None = None
        self._audio_data: np.ndarray | None = None
        self._audio_sr = 44100

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._load_model_async)

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20, 8))

        ctk.CTkLabel(
            header,
            text="Emociones en el tiempo",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Fragmenta el audio, analiza cada tramo y muestra triste → alegre en la gráfica.",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=MUTED,
        ).pack(anchor="w", pady=(4, 0))

        controls = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=12)
        controls.pack(fill="x", padx=24, pady=12)

        row = ctk.CTkFrame(controls, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=16)

        self.file_label = ctk.CTkLabel(
            row,
            text="Ningún audio seleccionado",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=MUTED,
            anchor="w",
        )
        self.file_label.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            row,
            text="Agregar audio",
            width=140,
            command=self._pick_audio,
            fg_color=ACCENT,
            hover_color="#2f6fd6",
        ).pack(side="right", padx=(8, 0))

        opts = ctk.CTkFrame(controls, fg_color="transparent")
        opts.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkLabel(
            opts,
            text="Tamaño de fragmento (s):",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=TEXT,
        ).pack(side="left")

        self.chunk_var = ctk.StringVar(value=str(DEFAULT_CHUNK_SECONDS))
        self.chunk_entry = ctk.CTkEntry(
            opts,
            width=70,
            textvariable=self.chunk_var,
            justify="center",
        )
        self.chunk_entry.pack(side="left", padx=(8, 16))

        ctk.CTkLabel(
            opts,
            text="Recomendado: 3 s",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=MUTED,
        ).pack(side="left")

        self.process_btn = ctk.CTkButton(
            opts,
            text="Procesar",
            width=120,
            command=self._start_process,
            state="disabled",
        )
        self.process_btn.pack(side="right")

        self.status_label = ctk.CTkLabel(
            self,
            text="Cargando modelo de emociones…",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=MUTED,
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=28, pady=(0, 4))

        self.progress = ctk.CTkProgressBar(self, height=8)
        self.progress.pack(fill="x", padx=28, pady=(0, 8))
        self.progress.set(0)

        chart_frame = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=16)
        chart_frame.pack(fill="both", expand=True, padx=24, pady=(4, 8))

        self.fig = Figure(figsize=(9, 4.2), dpi=100, facecolor=PANEL)
        self.ax = self.fig.add_subplot(111)
        self._style_axes(empty=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)
        self.canvas.mpl_connect("button_press_event", self._on_chart_click)

        legend = ctk.CTkFrame(self, fg_color="transparent")
        legend.pack(fill="x", padx=28, pady=(0, 4))

        legend_items = ctk.CTkFrame(legend, fg_color="transparent")
        legend_items.pack(side="left", fill="x", expand=True)

        for label, key in EMOTION_LEGEND:
            color = EMOTION_COLORS[key]
            item = ctk.CTkFrame(legend_items, fg_color="transparent")
            item.pack(side="left", padx=(0, 14))
            swatch = tk.Canvas(item, width=12, height=12, bg=BG, highlightthickness=0)
            swatch.create_oval(1, 1, 11, 11, fill=color, outline=color)
            swatch.pack(side="left", padx=(0, 5))
            ctk.CTkLabel(
                item,
                text=label,
                font=ctk.CTkFont(family="Segoe UI", size=11),
                text_color=MUTED,
            ).pack(side="left")

        self.summary_label = ctk.CTkLabel(
            legend,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=TEXT,
        )
        self.summary_label.pack(side="right")

        self.emotion_now_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=MUTED,
            anchor="w",
        )
        self.emotion_now_label.pack(fill="x", padx=28, pady=(0, 4))

        player = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=12)
        player.pack(fill="x", padx=24, pady=(0, 18))

        player_row = ctk.CTkFrame(player, fg_color="transparent")
        player_row.pack(fill="x", padx=16, pady=14)

        self.play_btn = ctk.CTkButton(
            player_row,
            text="▶",
            width=48,
            height=36,
            command=self._toggle_play,
            state="disabled",
            fg_color=ACCENT,
            hover_color="#2f6fd6",
            font=ctk.CTkFont(size=16),
        )
        self.play_btn.pack(side="left")

        self.time_label = ctk.CTkLabel(
            player_row,
            text="0:00 / 0:00",
            width=90,
            font=ctk.CTkFont(family="Consolas", size=13),
            text_color=TEXT,
        )
        self.time_label.pack(side="left", padx=(12, 8))

        self.seek_slider = ctk.CTkSlider(
            player_row,
            from_=0,
            to=1,
            number_of_steps=1000,
            command=self._on_seek_drag,
            state="disabled",
        )
        self.seek_slider.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.seek_slider.set(0)
        self.seek_slider.bind("<ButtonRelease-1>", self._on_seek_release)

        ctk.CTkLabel(
            player,
            text="Línea blanca = posición actual · clic en la gráfica para saltar a ese momento.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=MUTED,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(0, 12))

    def _style_axes(self, empty: bool = False, xmax: float = 60) -> None:
        self.ax.clear()
        self._playhead_line = None
        self.ax.set_facecolor(PANEL)
        self.ax.set_ylabel("Emoción", color=TEXT, fontsize=11, labelpad=10)
        self.ax.set_xlabel("Tiempo", color=TEXT, fontsize=11, labelpad=8)
        self.ax.set_ylim(0, 105)
        self.ax.set_yticks([20, 40, 60, 80, 100])
        self.ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color=TEXT)
        self.ax.set_xlim(0, max(xmax, 10))
        self.ax.tick_params(axis="x", colors=TEXT)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_linestyle("--")
        self.ax.spines["left"].set_color("#ffffff")
        self.ax.spines["bottom"].set_color("#ffffff")
        self.ax.yaxis.grid(True, linestyle=":", alpha=0.25, color="#ffffff")
        self.ax.set_axisbelow(True)
        if empty:
            self.ax.text(
                0.5,
                0.5,
                "Agrega un audio y pulsa Procesar",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=12,
            )
        self.fig.tight_layout()

    def _ensure_playhead(self) -> None:
        if self._playhead_line is None:
            self._playhead_line = self.ax.axvline(
                x=0,
                color=PLAYHEAD,
                linewidth=1.8,
                linestyle="-",
                alpha=0.95,
                zorder=5,
            )

    def _set_playhead(self, t: float) -> None:
        if not self._results:
            return
        self._ensure_playhead()
        assert self._playhead_line is not None
        self._playhead_line.set_xdata([t, t])
        self.canvas.draw_idle()
        self._update_emotion_at(t)

    def _update_emotion_at(self, t: float) -> None:
        if not self._results:
            self.emotion_now_label.configure(text="")
            return
        nearest = min(self._results, key=lambda r: abs(r.time_sec - t))
        self.emotion_now_label.configure(
            text=(
                f"En {format_time(t)} → {nearest.dominant_es} "
                f"({nearest.confidence:.0f}% confianza) · "
                f"valencia {nearest.valence_pct:.0f}%"
            )
        )

    def _current_position(self) -> float:
        if self._playing:
            elapsed = time.monotonic() - self._play_anchor_mono
            return min(self._duration, self._play_anchor_pos + elapsed)
        return self._seek_pos

    def _load_model_async(self) -> None:
        def work() -> None:
            try:
                analyzer = EmotionAnalyzer()
                self.after(0, lambda: self._on_model_ready(analyzer))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_model_error(str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _on_model_ready(self, analyzer: EmotionAnalyzer) -> None:
        self.analyzer = analyzer
        self.status_label.configure(text="Modelo listo. Selecciona un audio.")
        if self.audio_path:
            self.process_btn.configure(state="normal")

    def _on_model_error(self, message: str) -> None:
        self.status_label.configure(text=f"Error al cargar el modelo: {message}")
        messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{message}")

    def _pick_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccionar audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"),
                ("Todos", "*.*"),
            ],
        )
        if not path:
            return
        self._stop_playback(reset=True)
        self.audio_path = path
        self.file_label.configure(text=Path(path).name, text_color=TEXT)
        self.status_label.configure(text="Audio listo para procesar.")
        self._results = []
        self.emotion_now_label.configure(text="")
        if self.analyzer and not self._busy:
            self.process_btn.configure(state="normal")

    def _start_process(self) -> None:
        if not self.audio_path or not self.analyzer or self._busy:
            return

        try:
            chunk_seconds = float(self.chunk_var.get().replace(",", "."))
            if chunk_seconds < 1.0 or chunk_seconds > 10.0:
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "Fragmento inválido",
                "Usa un tamaño de fragmento entre 1 y 10 segundos.",
            )
            return

        self._stop_playback(reset=True)
        self._busy = True
        self.process_btn.configure(state="disabled")
        self.play_btn.configure(state="disabled")
        self.seek_slider.configure(state="disabled")
        self.progress.set(0)
        self.status_label.configure(text="Analizando fragmentos…")
        self.summary_label.configure(text="")
        self.emotion_now_label.configure(text="")

        path = self.audio_path
        analyzer = self.analyzer

        def progress(value: float, label: str) -> None:
            self.after(0, lambda: self._update_progress(value, label))

        def work() -> None:
            try:
                results = analyzer.analyze(
                    path,
                    chunk_seconds=chunk_seconds,
                    progress_callback=progress,
                )
                audio_data, sample_rate, duration = self._load_playback_audio(path)
                self.after(
                    0,
                    lambda: self._on_results(
                        results, duration, audio_data, sample_rate, chunk_seconds
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_process_error(str(exc)))

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _load_playback_audio(path: str) -> tuple[np.ndarray, int, float]:
        waveform, sample_rate = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        data = waveform.squeeze(0).numpy().astype(np.float32)
        peak = np.max(np.abs(data)) if data.size else 0.0
        if peak > 1.0:
            data = data / peak
        duration = float(data.shape[0]) / float(sample_rate) if sample_rate else 0.0
        return data, int(sample_rate), duration

    def _update_progress(self, value: float, label: str) -> None:
        self.progress.set(value)
        self.status_label.configure(text=label)

    def _on_process_error(self, message: str) -> None:
        self._busy = False
        self.process_btn.configure(state="normal")
        self.status_label.configure(text="Error durante el análisis.")
        messagebox.showerror("Error", message)

    def _on_results(
        self,
        results: list,
        duration: float,
        audio_data: np.ndarray,
        sample_rate: int,
        chunk_seconds: float,
    ) -> None:
        self._busy = False
        self.process_btn.configure(state="normal")
        self.progress.set(1)

        if not results:
            self.status_label.configure(text="No se obtuvieron fragmentos válidos.")
            return

        self._results = results
        self._audio_data = audio_data
        self._audio_sr = sample_rate

        times = [r.time_sec for r in results]
        vals = [r.valence_pct for r in results]
        colors = [color_for_result(r.dominant, r.valence_pct) for r in results]

        last_end = times[-1] + (chunk_seconds / 2)
        self._duration = max(duration, last_end, max(times))
        xmax = self._duration * 1.02

        self._style_axes(empty=False, xmax=xmax)
        self.ax.scatter(
            times,
            vals,
            c=colors,
            s=90,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            alpha=0.95,
        )
        self.ax.plot(times, vals, color="#ffffff", alpha=0.15, linewidth=1.2, zorder=2)
        self._ensure_playhead()
        self._set_playhead(0)
        self.canvas.draw_idle()

        happy = sum(1 for r in results if r.dominant == "joy")
        sad = sum(1 for r in results if r.dominant == "sadness")
        angry = sum(1 for r in results if r.dominant == "anger")
        avg = sum(vals) / len(vals)
        tone = "más alegre" if avg >= 55 else "más triste" if avg <= 45 else "mixta"
        self.summary_label.configure(
            text=(
                f"{len(results)} fragmentos · tono {tone} · "
                f"alegre {happy} · triste {sad} · enojo {angry}"
            )
        )
        self.status_label.configure(
            text="Análisis listo (modelo ES). Reproduce y sigue la línea blanca."
        )
        self._prepare_player()

    def _prepare_player(self) -> None:
        if self._audio_data is None or self._duration <= 0:
            return
        self._seek_pos = 0.0
        self._playing = False
        self.play_btn.configure(state="normal", text="▶")
        self.seek_slider.configure(state="normal", to=max(self._duration, 0.1))
        self._updating_slider = True
        self.seek_slider.set(0)
        self._updating_slider = False
        self.time_label.configure(text=f"0:00 / {format_time(self._duration)}")

    def _toggle_play(self) -> None:
        if self._audio_data is None or self._duration <= 0:
            return
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if self._audio_data is None:
            return

        pos = self._seek_pos
        if pos >= self._duration - 0.05:
            pos = 0.0
            self._seek_pos = 0.0

        start = int(pos * self._audio_sr)
        start = max(0, min(start, len(self._audio_data) - 1))
        chunk = self._audio_data[start:]

        try:
            sd.stop()
            sd.play(chunk, self._audio_sr, blocking=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Reproducción", str(exc))
            return

        self._playing = True
        self._play_anchor_pos = pos
        self._play_anchor_mono = time.monotonic()
        self.play_btn.configure(text="⏸")
        self._schedule_tick()

    def _pause(self) -> None:
        self._seek_pos = self._current_position()
        sd.stop()
        self._playing = False
        self.play_btn.configure(text="▶")
        self._set_playhead(self._seek_pos)
        self._sync_slider(self._seek_pos)
        self.time_label.configure(
            text=f"{format_time(self._seek_pos)} / {format_time(self._duration)}"
        )

    def _stop_playback(self, reset: bool = False) -> None:
        try:
            sd.stop()
        except Exception:  # noqa: BLE001
            pass
        self._playing = False
        if reset:
            self._seek_pos = 0.0
            self._duration = 0.0
            self._audio_data = None
            self.play_btn.configure(state="disabled", text="▶")
            self.seek_slider.configure(state="disabled")
            self._updating_slider = True
            self.seek_slider.set(0)
            self._updating_slider = False
            self.time_label.configure(text="0:00 / 0:00")
        else:
            self.play_btn.configure(text="▶")

    def _schedule_tick(self) -> None:
        if self._tick_job is not None:
            try:
                self.after_cancel(self._tick_job)
            except Exception:  # noqa: BLE001
                pass
        self._tick_job = self.after(40, self._tick)

    def _tick(self) -> None:
        self._tick_job = None
        if not self._playing:
            return

        pos = self._current_position()
        if pos >= self._duration - 0.02:
            self._seek_pos = self._duration
            self._stop_playback(reset=False)
            self._set_playhead(self._duration)
            self._sync_slider(self._duration)
            self.time_label.configure(
                text=f"{format_time(self._duration)} / {format_time(self._duration)}"
            )
            return

        self._set_playhead(pos)
        self._sync_slider(pos)
        self.time_label.configure(
            text=f"{format_time(pos)} / {format_time(self._duration)}"
        )
        self._schedule_tick()

    def _sync_slider(self, pos: float) -> None:
        self._updating_slider = True
        self.seek_slider.set(pos)
        self._updating_slider = False

    def _on_seek_drag(self, value: float) -> None:
        if self._updating_slider:
            return
        t = float(value)
        self.time_label.configure(
            text=f"{format_time(t)} / {format_time(self._duration)}"
        )
        self._set_playhead(t)

    def _on_seek_release(self, _event=None) -> None:
        if self.seek_slider.cget("state") == "disabled":
            return
        t = float(self.seek_slider.get())
        self._seek_to(t)

    def _on_chart_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or not self._results:
            return
        t = float(max(0.0, min(self._duration, event.xdata)))
        self._seek_to(t, resume=True)

    def _seek_to(self, t: float, resume: bool = False) -> None:
        was_playing = self._playing
        t = max(0.0, min(self._duration, t))
        self._seek_pos = t
        self._sync_slider(t)
        self._set_playhead(t)
        self.time_label.configure(
            text=f"{format_time(t)} / {format_time(self._duration)}"
        )

        sd.stop()
        self._playing = False
        self.play_btn.configure(text="▶")

        if was_playing or resume:
            self._play()

    def _on_close(self) -> None:
        self._stop_playback(reset=True)
        self.destroy()


def main() -> None:
    app = EmotionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
