"""Gráficas del informe: oscilación, zoom del periodo y espectro."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.signal import find_peaks

from .analisis import Espectro, MedicionPeriodo, Pico


def _figura(titulo: str):
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    return fig, ax


def graficar_oscilacion(x: np.ndarray, fs: int) -> Figure:
    """Figura 1: oscilación en el tiempo (señal completa)."""
    t = np.arange(x.size) / fs
    fig, ax = _figura("Oscilación en el tiempo")
    ax.plot(t, x, lw=0.6)
    ax.set(xlabel="Tiempo (s)", ylabel="Amplitud (u. arb.)")
    return fig


def graficar_zoom_periodo(x: np.ndarray, fs: int, med: MedicionPeriodo,
                          n_periodos: int = 4) -> Figure:
    """Figura 2: zoom central de unos pocos periodos, con T marcado."""
    ventana = int(round(n_periodos * med.periodo * fs))
    inicio = max(0, (x.size - ventana) // 2)
    seg = x[inicio:inicio + ventana]
    t = (np.arange(seg.size) + inicio) / fs

    fig, ax = _figura(f"Zoom para medir el periodo  (T ≈ {med.periodo * 1e3:.3f} ms)")
    ax.plot(t, seg, lw=1.2, marker=".", ms=2)
    picos, _ = find_peaks(seg, height=0.5 * seg.max())
    if picos.size:
        t1, y = t[picos[0]], seg[picos[0]]
        ax.annotate("", xy=(t1 + med.periodo, y), xytext=(t1, y),
                    arrowprops=dict(arrowstyle="<->", color="crimson"))
        ax.text(t1 + med.periodo / 2, y * 1.03, "T", color="crimson", ha="center")
    ax.set(xlabel="Tiempo (s)", ylabel="Amplitud (u. arb.)")
    return fig


def graficar_espectro(esp: Espectro, picos: list[Pico], fmax: float = 5000.0) -> Figure:
    """Figura 3: espectro de frecuencias completo con los picos señalados."""
    fig, ax = _figura("Espectro de frecuencias")
    m = esp.frecuencias <= fmax
    ax.plot(esp.frecuencias[m], esp.amplitudes[m], lw=0.9)
    for p in picos:
        if p.frecuencia <= fmax:
            ax.plot(p.frecuencia, p.amplitud, "rv", ms=6)
            ax.annotate(f"{p.frecuencia:.1f} Hz", (p.frecuencia, p.amplitud),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set(xlabel="Frecuencia (Hz)", ylabel="Amplitud (u. arb.)")
    return fig


def graficar_zoom_espectro(esp: Espectro, pico: Pico) -> Figure:
    """Figura 4: zoom alrededor del pico de mayor amplitud."""
    ancho = max(0.15 * pico.frecuencia, 30.0)
    m = (esp.frecuencias >= pico.frecuencia - ancho) & (esp.frecuencias <= pico.frecuencia + ancho)
    fig, ax = _figura(f"Zoom del espectro: pico principal en {pico.frecuencia:.2f} Hz")
    ax.plot(esp.frecuencias[m], esp.amplitudes[m], lw=1.2)
    ax.axvline(pico.frecuencia, color="crimson", ls="--", lw=0.9)
    ax.set(xlabel="Frecuencia (Hz)", ylabel="Amplitud (u. arb.)")
    return fig
