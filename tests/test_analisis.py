import numpy as np
import pytest

from espectro_sonido import analisis

FS = 44_100


def tono(f0=440.0, armonicos=(1.0, 0.4, 0.15), dur=1.0, silencio=0.3):
    t = np.arange(int(dur * FS)) / FS
    x = sum(a * np.sin(2 * np.pi * f0 * (k + 1) * t) for k, a in enumerate(armonicos))
    pad = np.zeros(int(silencio * FS))
    return np.concatenate([pad, x, pad])


def test_pico_principal_y_armonicos():
    esp = analisis.calcular_espectro(tono(), FS)
    frecs = [p.frecuencia for p in analisis.detectar_picos(esp)]
    for esperada in (440, 880, 1320):
        assert min(abs(f - esperada) for f in frecs) < 0.5


def test_amplitud_normalizada():
    t = np.arange(FS) / FS
    esp = analisis.calcular_espectro(0.5 * np.sin(2 * np.pi * 1000 * t), FS)
    assert esp.amplitudes.max() == pytest.approx(0.5, rel=0.02)


def test_periodo_y_frecuencia_angular():
    x = analisis.recortar_silencio(tono(), FS)
    m = analisis.medir_periodo(x, FS)
    assert m.frecuencia == pytest.approx(440.0, abs=0.5)
    assert m.periodo == pytest.approx(1 / 440.0, rel=1e-3)
    assert m.frecuencia_angular == pytest.approx(2 * np.pi * 440.0, rel=1e-3)


def test_fundamental_debil_no_da_error_de_octava():
    x = tono(220.0, armonicos=(0.3, 1.0, 0.8, 0.5))
    m = analisis.medir_periodo(analisis.recortar_silencio(x, FS), FS)
    assert m.frecuencia == pytest.approx(220.0, abs=0.5)


def test_recorta_silencio():
    x = tono()
    assert analisis.recortar_silencio(x, FS).size < x.size


def test_senal_nula():
    with pytest.raises(ValueError):
        analisis.medir_periodo(np.zeros(FS), FS)
