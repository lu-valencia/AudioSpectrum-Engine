"""Prueba del flujo de grabación en vivo simulando el micrófono."""
import numpy as np

from espectro_sonido import captura, cli

FS = 44_100


def _señal_falsa(duracion, fs=FS, dispositivo=None):
    t = np.arange(int(duracion * fs)) / fs
    x = 0.5 * np.sin(2 * np.pi * 523.25 * t) + 0.15 * np.sin(2 * np.pi * 1046.5 * t)
    return np.concatenate([np.zeros(fs // 4), x, np.zeros(fs // 4)]), fs


def test_en_vivo_genera_resultados(tmp_path, monkeypatch):
    monkeypatch.setattr(captura, "grabar", _señal_falsa)
    codigo = cli.main(["--en-vivo", "-n", "do5", "-d", "1", "--cuenta", "0",
                       "--no-mostrar", "-o", str(tmp_path)])
    assert codigo == 0
    assert (tmp_path / "do5.wav").exists()
    assert len(list(tmp_path.glob("do5_fig*.png"))) == 4
    assert "523." in (tmp_path / "do5_resultados.txt").read_text(encoding="utf-8")


def test_aviso_nivel_bajo_y_saturacion():
    assert captura.revisar_nivel(np.full(100, 0.001)) is not None
    assert captura.revisar_nivel(np.full(100, 1.0)) is not None
    assert captura.revisar_nivel(np.full(100, 0.5)) is None
