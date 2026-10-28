"""Procesamiento de señal: espectro de Fourier, picos y periodo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Espectro:
    frecuencias: np.ndarray  # Hz
    amplitudes: np.ndarray   # amplitud de cada senoide, en unidades de la señal


@dataclass(frozen=True)
class Pico:
    frecuencia: float  # Hz
    amplitud: float


@dataclass(frozen=True)
class MedicionPeriodo:
    periodo: float             # s
    frecuencia: float          # Hz  (f = 1/T)
    frecuencia_angular: float  # rad/s  (ω = 2πf)


def recortar_silencio(x: np.ndarray, fs: int, umbral: float = 0.1,
                      ventana_s: float = 0.02) -> np.ndarray:
    """Descarta el silencio inicial y final según la energía RMS por bloques."""
    n = max(1, int(ventana_s * fs))
    bloques = len(x) // n
    if bloques == 0:
        return x
    rms = np.sqrt(np.mean(x[: bloques * n].reshape(bloques, n) ** 2, axis=1))
    activos = np.flatnonzero(rms >= umbral * rms.max())
    if activos.size == 0:
        return x
    return x[activos[0] * n:(activos[-1] + 1) * n]


def calcular_espectro(x: np.ndarray, fs: int) -> Espectro:
    """Espectro de amplitud de un solo lado con ventana de Hann.

    Se elimina la componente continua, se aplica la ventana para reducir la
    fuga espectral y se rellena con ceros (zero-padding) para una malla de
    frecuencias más fina. La normalización 2/Σw hace que una senoide de
    amplitud A produzca un pico de altura A.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        raise ValueError("La señal es demasiado corta para calcular el espectro.")
    x = x - x.mean()
    ventana = np.hanning(x.size)
    nfft = 1 << int(np.ceil(np.log2(x.size)) + 1)
    X = np.fft.rfft(x * ventana, n=nfft)
    amplitudes = 2.0 * np.abs(X) / ventana.sum()
    return Espectro(np.fft.rfftfreq(nfft, d=1.0 / fs), amplitudes)


def _interpolar_parabola(y0: float, y1: float, y2: float) -> tuple[float, float]:
    """Desplazamiento (en muestras) y valor del vértice de la parábola por 3 puntos."""
    den = y0 - 2.0 * y1 + y2
    if den == 0:
        return 0.0, y1
    delta = 0.5 * (y0 - y2) / den
    return delta, y1 - 0.25 * (y0 - y2) * delta


def detectar_picos(esp: Espectro, n_max: int = 10, umbral_rel: float = 0.05,
                   fmin: float = 20.0, fmax: float | None = None) -> list[Pico]:
    """Picos principales del espectro, ordenados por frecuencia creciente.

    umbral_rel: altura mínima como fracción del pico máximo.
    La frecuencia de cada pico se refina por interpolación parabólica.
    """
    f, a = esp.frecuencias, esp.amplitudes
    mascara = f >= fmin
    if fmax is not None:
        mascara &= f <= fmax
    indices = np.flatnonzero(mascara)
    if indices.size < 3:
        return []
    picos_locales, _ = find_peaks(a[indices], height=umbral_rel * a[indices].max())
    candidatos = indices[picos_locales]
    candidatos = candidatos[np.argsort(a[candidatos])[::-1][:n_max]]

    df = f[1] - f[0]
    picos = []
    for i in candidatos:
        delta, amp = _interpolar_parabola(a[i - 1], a[i], a[i + 1])
        picos.append(Pico(float(f[i] + delta * df), float(amp)))
    return sorted(picos, key=lambda p: p.frecuencia)


def medir_periodo(x: np.ndarray, fs: int, fmin: float = 50.0,
                  fmax: float = 4000.0) -> MedicionPeriodo:
    """Mide el periodo con la autocorrelación (independiente de la FFT).

    Se toma el primer máximo de la autocorrelación cuya altura sea ≥ 90 % del
    mayor, lo que evita confundir la fundamental con un submúltiplo (error de octava).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    r = np.fft.irfft(X * np.conj(X), nfft)[:n]
    if r[0] <= 0:
        raise ValueError("Señal nula: no se puede medir el periodo.")
    r = r / r[0]

    lag_min = max(2, int(fs / fmax))
    lag_max = min(int(fs / fmin), n - 2)
    if lag_max <= lag_min:
        raise ValueError("Señal demasiado corta para el rango de frecuencias pedido.")
    seg = r[lag_min:lag_max + 1]
    picos, _ = find_peaks(seg)
    if picos.size == 0:
        raise ValueError("No se encontró periodicidad en la señal.")
    elegido = picos[np.argmax(seg[picos] >= 0.9 * seg[picos].max())]
    lag = elegido + lag_min
    delta, _ = _interpolar_parabola(r[lag - 1], r[lag], r[lag + 1])
    periodo = (lag + delta) / fs
    f0 = 1.0 / periodo
    return MedicionPeriodo(periodo, f0, 2.0 * np.pi * f0)
