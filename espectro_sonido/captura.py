"""Captura y lectura de audio (tarjeta de sonido o archivo WAV)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

FS_DEFECTO = 44_100  # Hz, estándar de audio de calidad CD


def listar_dispositivos() -> str:
    """Devuelve una tabla de los dispositivos de audio disponibles."""
    import sounddevice as sd

    return str(sd.query_devices())


def grabar(duracion: float, fs: int = FS_DEFECTO, dispositivo: int | str | None = None
           ) -> tuple[np.ndarray, int]:
    """Graba audio mono desde el micrófono.

    Parameters
    ----------
    duracion : segundos de grabación (> 0).
    fs : frecuencia de muestreo en Hz.
    dispositivo : índice o nombre del dispositivo de entrada (None = predeterminado).

    Returns
    -------
    (señal, fs) con la señal en float64 normalizada al rango [-1, 1].
    """
    if duracion <= 0:
        raise ValueError("La duración debe ser mayor que cero.")
    import sounddevice as sd  # import perezoso: el análisis no requiere PortAudio

    señal = sd.rec(int(duracion * fs), samplerate=fs, channels=1,
                   dtype="float64", device=dispositivo)
    sd.wait()
    return señal[:, 0], fs


def leer_wav(ruta: str | Path) -> tuple[np.ndarray, int]:
    """Lee un WAV y lo devuelve como señal mono float64 en [-1, 1]."""
    fs, datos = wavfile.read(str(ruta))
    if np.issubdtype(datos.dtype, np.integer):
        datos = datos / float(np.iinfo(datos.dtype).max + 1)
    datos = datos.astype(np.float64)
    if datos.ndim > 1:  # estéreo -> mono
        datos = datos.mean(axis=1)
    return datos, fs


def guardar_wav(ruta: str | Path, señal: np.ndarray, fs: int) -> None:
    """Guarda la señal (float en [-1, 1]) como WAV de 16 bits."""
    pcm = np.clip(señal, -1.0, 1.0)
    wavfile.write(str(ruta), fs, (pcm * 32767).astype(np.int16))


def revisar_nivel(señal: np.ndarray) -> str | None:
    """Advierte si la grabación quedó muy baja o saturada; None si el nivel es adecuado."""
    pico = float(np.max(np.abs(señal))) if señal.size else 0.0
    if pico < 0.02:
        return (f"Nivel muy bajo (pico {pico:.3f}): acerque el instrumento al micrófono "
                "o suba la ganancia de entrada en el sistema.")
    if pico >= 0.99:
        return ("La señal se saturó (recorte): aleje el instrumento o baje la ganancia, "
                "porque el recorte genera armónicos falsos en el espectro.")
    return None
