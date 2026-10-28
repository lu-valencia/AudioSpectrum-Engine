# Registro de ondas sonoras y análisis de su espectro

Proyecto de aula — Física, Calor y Ondas. Captura un sonido por la tarjeta de sonido,
grafica su oscilación en el tiempo y calcula su espectro de frecuencias (transformada de Fourier).

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso
**Modo interactivo (recomendado):** sin argumentos se abre un menú para grabar en vivo,
analizar un WAV o ver los dispositivos de audio.
```bash
python -m espectro_sonido
```
**Línea de comandos:**
```bash
python -m espectro_sonido --en-vivo -n flauta_la4 -d 3   # graba en vivo 3 s y analiza
python -m espectro_sonido --archivo nota.wav             # analiza un WAV existente
python -m espectro_sonido --listar-dispositivos          # ver micrófonos disponibles
python -m espectro_sonido --help
```
En la grabación en vivo hay una cuenta regresiva (`--cuenta`, 3 s por defecto); cuando aparezca
"¡GRABANDO!", toque la nota de forma sostenida y cerca del micrófono. El programa avisa si el
nivel quedó muy bajo o saturado.

En `resultados/` se guardan: el WAV, `*_resultados.txt` (T, f, ω y picos) y cuatro figuras:
1. Oscilación en el tiempo · 2. Zoom para medir el periodo · 3. Espectro · 4. Zoom del pico principal.

## Método
- **Periodo / f / ω:** autocorrelación de la señal; T = lag del primer máximo, f = 1/T, ω = 2πf.
- **Espectro:** FFT con ventana de Hann y zero-padding; amplitud normalizada (una senoide de
  amplitud A da un pico A). Frecuencias de los picos refinadas por interpolación parabólica.
- Se recorta automáticamente el silencio inicial y final.

## Estructura
```
espectro_sonido/  captura.py · analisis.py · graficas.py · cli.py
tests/            pruebas con señales sintéticas (pytest)
```
Pruebas: `pytest`
