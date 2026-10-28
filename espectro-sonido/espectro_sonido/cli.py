"""Interfaz de línea de comandos (con modo interactivo de grabación en vivo)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

from . import analisis, captura, graficas


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="espectro_sonido",
        description="Captura un sonido, grafica su oscilación y su espectro de Fourier. "
                    "Sin argumentos se abre un menú interactivo.")
    fuente = p.add_mutually_exclusive_group()
    fuente.add_argument("--en-vivo", action="store_true",
                        help="grabar en vivo con el micrófono (opción por defecto)")
    fuente.add_argument("--archivo", type=Path, help="analizar un WAV en lugar de grabar")
    p.add_argument("-n", "--nombre", default="sonido",
                   help="etiqueta para los archivos de salida (p. ej. flauta_la4)")
    p.add_argument("-d", "--duracion", type=float, default=3.0, help="segundos a grabar")
    p.add_argument("--cuenta", type=int, default=3,
                   help="segundos de cuenta regresiva antes de grabar (0 = sin cuenta)")
    p.add_argument("--fs", type=int, default=captura.FS_DEFECTO, help="frecuencia de muestreo (Hz)")
    p.add_argument("--dispositivo", help="índice o nombre del dispositivo de entrada")
    p.add_argument("--listar-dispositivos", action="store_true")
    p.add_argument("-o", "--salida", type=Path, default=Path("resultados"))
    p.add_argument("--fmax", type=float, default=5000.0, help="frecuencia máxima a graficar (Hz)")
    p.add_argument("--no-mostrar", action="store_true", help="solo guardar, sin ventanas")
    return p


def _dispositivo(valor: str | None):
    if valor is None:
        return None
    return int(valor) if valor.isdigit() else valor


def _cuenta_regresiva(segundos: int) -> None:
    for s in range(segundos, 0, -1):
        print(f"  {s}...", flush=True)
        time.sleep(1)


def _obtener_señal(args):
    if args.archivo:
        señal, fs = captura.leer_wav(args.archivo)
        print(f"Archivo leído: {args.archivo} ({señal.size / fs:.2f} s, {fs} Hz)")
        return señal, fs
    print(f"Grabación en vivo de {args.duracion:g} s. Prepare el instrumento.")
    _cuenta_regresiva(args.cuenta)
    print("¡GRABANDO! Toque la nota de forma sostenida.")
    señal, fs = captura.grabar(args.duracion, args.fs, _dispositivo(args.dispositivo))
    print("Grabación terminada.")
    aviso = captura.revisar_nivel(señal)
    if aviso:
        print(f"Aviso: {aviso}")
    return señal, fs


def ejecutar(args) -> int:
    """Captura (o lee), analiza, imprime y guarda los resultados."""
    if args.no_mostrar:
        plt.switch_backend("Agg")
    señal, fs = _obtener_señal(args)

    activa = analisis.recortar_silencio(señal, fs)
    try:
        med = analisis.medir_periodo(activa, fs)
    except ValueError as e:
        print(f"Error: {e} Intente de nuevo, con la nota más fuerte y sostenida.", file=sys.stderr)
        return 1
    esp = analisis.calcular_espectro(activa, fs)
    picos = analisis.detectar_picos(esp, fmax=args.fmax)
    if not picos:
        print("Error: no se detectaron picos en el espectro.", file=sys.stderr)
        return 1
    principal = max(picos, key=lambda p: p.amplitud)

    resumen = [
        f"Resultados — {args.nombre}",
        f"Frecuencia de muestreo: {fs} Hz | Duración analizada: {activa.size / fs:.3f} s",
        f"Periodo T            = {med.periodo * 1e3:.4f} ms",
        f"Frecuencia f = 1/T   = {med.frecuencia:.3f} Hz",
        f"Frecuencia angular ω = {med.frecuencia_angular:.2f} rad/s",
        f"Pico principal (FFT) = {principal.frecuencia:.3f} Hz",
        "Frecuencias componentes (FFT):",
        *[f"  {p.frecuencia:9.2f} Hz   amplitud relativa {p.amplitud / principal.amplitud:.3f}"
          for p in picos],
    ]
    texto = "\n".join(resumen)
    print("\n" + texto)

    args.salida.mkdir(parents=True, exist_ok=True)
    base = args.salida / args.nombre
    captura.guardar_wav(f"{base}.wav", señal, fs)
    Path(f"{base}_resultados.txt").write_text(texto + "\n", encoding="utf-8")
    figuras = {
        "fig1_oscilacion": graficas.graficar_oscilacion(activa, fs),
        "fig2_zoom_periodo": graficas.graficar_zoom_periodo(activa, fs, med),
        "fig3_espectro": graficas.graficar_espectro(esp, picos, args.fmax),
        "fig4_zoom_espectro": graficas.graficar_zoom_espectro(esp, principal),
    }
    for sufijo, fig in figuras.items():
        fig.savefig(f"{base}_{sufijo}.png", dpi=200)
    print(f"\nArchivos guardados en: {args.salida.resolve()}")

    if args.no_mostrar:
        plt.close("all")
    else:
        plt.show()
    return 0


def _preguntar(texto: str, defecto: str) -> str:
    r = input(f"{texto} [{defecto}]: ").strip()
    return r or defecto


def _menu() -> int:
    """Menú interactivo: grabar en vivo, analizar un WAV o ver dispositivos."""
    parser = _parser()
    while True:
        print("\n=== Registro y espectro de un sonido ===")
        print("  1) Grabar en vivo con el micrófono")
        print("  2) Analizar un archivo WAV")
        print("  3) Listar dispositivos de audio")
        print("  4) Salir")
        opcion = input("Elija una opción: ").strip()
        if opcion == "1":
            nombre = _preguntar("Nombre de la muestra (ej. flauta_la4)", "muestra")
            dur = _preguntar("Duración en segundos", "3")
            dispositivo = _preguntar("Dispositivo de entrada (índice, o Enter = predeterminado)", "")
            argv = ["--en-vivo", "-n", nombre, "-d", dur]
            if dispositivo:
                argv += ["--dispositivo", dispositivo]
        elif opcion == "2":
            ruta = input("Ruta del archivo WAV: ").strip().strip('"')
            argv = ["--archivo", ruta, "-n", _preguntar("Nombre de la muestra", Path(ruta).stem)]
        elif opcion == "3":
            try:
                print(captura.listar_dispositivos())
            except Exception as e:  # PortAudio ausente, sin dispositivos, etc.
                print(f"No se pudo consultar el audio: {e}", file=sys.stderr)
            continue
        elif opcion == "4":
            return 0
        else:
            print("Opción no válida.")
            continue
        try:
            ejecutar(parser.parse_args(argv))
        except (OSError, ValueError, SystemExit) as e:
            print(f"No se pudo completar la operación: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return _menu()
    args = _parser().parse_args(argv)
    if args.listar_dispositivos:
        print(captura.listar_dispositivos())
        return 0
    return ejecutar(args)


if __name__ == "__main__":
    raise SystemExit(main())
