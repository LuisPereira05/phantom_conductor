# Phantom Conductor

Interfaz de dirección en tiempo real para pistas de acompañamiento: controla la reproducción
con gestos de mano o un pedal de pie, y haz que el tempo de la pista siga al detector de BPM
en vivo (micrófono), a golpes manuales con un Arduino, o a un pedal de 3 botones.

Todo corre como un conjunto de hilos daemon que comparten un único objeto de estado
thread-safe, con un dashboard Dear PyGui en el hilo principal.

```
pip install dearpygui mutagen sounddevice pyserial
```

---

## Estructura del proyecto

```
phantom_conductor/
├─ main.py                  — punto de entrada; inicia todos los hilos, luego dpg.run()
│
├─ state.py                 — estado compartido thread-safe + cola de pistas
├─ config.py                — configuración, persistida en phantom_config.json
├─ buffers.py               — singletons de ring buffer / queue compartidos
├─ logger.py                — logger thread-safe con ring buffer
│
├─ audio_input.py           — E/S de micrófono/altavoz (sounddevice)
├─ audio_analysis.py        — cadena DSP de detección de BPM
├─ audio_processing.py      — carga de pistas + reproducción sincronizada con beats
├─ tempo_tapper.py          — puente serial Arduino: tempo tapper Y pedal de botones
│
├─ pedal.py                 — máquina de estados: detección tap/hold/combo de 3 botones
├─ pedal_setup_ui.py        — mixin Dear PyGui para la ventana PEDAL SETUP
│
├─ video_input.py           — cámara OpenCV + MediaPipe HandLandmarker
├─ gesture_recognition.py   — clasificación de gestos + overlay HUD
├─ gesture_trainer.py       — extracción de features + matching de gestos personalizados
├─ gesture_train_ui.py      — mixin Dear PyGui para el flujo de entrenamiento de gestos
│
├─ track_queue.py           — TrackQueue en memoria
├─ tracklist.py             — TrackQueue + persistencia JSON
├─ ui.py                    — dashboard Dear PyGui (hilo principal)
│
├─ audio/                   — biblioteca de pistas
├─ hand_landmarker.task     — modelo MediaPipe (descargado automáticamente si falta)
├─ requirements.txt
│
├─ phantom_config.json      — persistencia de CFG (config.py)
├─ tracklist.json           — persistencia de la cola (tracklist.py)
├─ gesture_pool.bin         — vectores de gestos entrenados (gesture_trainer.py)
├─ gesture_pool.meta.json   — nombres/comandos de gestos (sidecar)
└─ gesture_templates.json   — formato legacy de gestos, migrado en el primer arranque
```

## Mapa de hilos

```
main.py
 ├─ bpm-analysis    → audio_analysis.bpm_analysis_thread
 ├─ backing-track   → audio_processing.backing_track_thread
 ├─ io-manager      → audio_input.io_manager_thread
 │                     ├─ audio-in   → audio_input.input_thread
 │                     └─ audio-out  → audio_input.playback_thread
 ├─ gesture-vision  → gesture_recognition.gesture_vision_thread
 │                     (usa internamente los helpers de video_input)
 └─ tempo-tapper    → tempo_tapper.tempo_tapper_thread
                       (lee golpes de BPM del piezo Arduino Y eventos de botones del pedal)

Hilo principal → ui.PhantomUI.run()   (Dear PyGui debe correr en el hilo principal)
```

El índice de cámara se lee de `phantom_config.json` por defecto; también se puede pasar
por línea de comandos:

```
python main.py 2
```

El hilo tempo-tapper siempre arranca, incluso sin tapper configurado — verifica
`CFG.use_tempo_tapper` en cada línea serial, por lo que el checkbox de Configuración
puede habilitarlo o deshabilitarlo en tiempo de ejecución sin reiniciar. Si no hay
Arduino conectado, solo loguea un aviso periódico "no se encontró puerto serial —
reintentando…" y es inofensivo dejarlo corriendo.

---

## Mapa de módulos

| Módulo | Responsabilidad |
|---|---|
| `state.py` | Estado compartido thread-safe y cola de pistas. Sin E/S, sin hilos, sin GUI. |
| `config.py` | Fuente única de verdad para ajustes del usuario, persistida en `config.json`. |
| `buffers.py` | Singletons de ring buffer / queue compartidos para evitar imports circulares entre módulos de audio. |
| `audio_input.py` | E/S pura: callback del micrófono, streams de entrada/salida, watcher de reinicio de dispositivo. |
| `audio_analysis.py` | Cadena DSP de detección de BPM a partir de muestras crudas del micrófono. |
| `audio_processing.py` | Carga de pistas y reproducción de backing track beat-a-beat con time-stretching en vivo. |
| `tempo_tapper.py` | Puente serial al Arduino. Maneja tanto las líneas `BPM:`/`TAP:` del tempo tapper como las líneas `B<n>p`/`B<n>r` del pedal de botones — **una sola conexión serial, dos dispositivos lógicos**. |
| `pedal.py` | Máquina de estados pura para el pedal de 3 botones: detección de tap, hold y combos por superposición temporal. Sin imports de serial — solo lógica. |
| `pedal_setup_ui.py` | Mixin Dear PyGui para la ventana PEDAL SETUP: indicadores de botones en vivo y tabla de remapeo de comandos. |
| `gesture_recognition.py` | Clasificación de gestos de mano (personalizados + integrados) y overlay HUD. |
| `gesture_trainer.py` | Extracción de features invariante a rotación y almacenamiento/matching de gestos personalizados. |
| `gesture_train_ui.py` | Mixin Dear PyGui para el flujo de entrenamiento de gestos. |
| `video_input.py` | Posee la cámara OpenCV y la sesión MediaPipe HandLandmarker. |
| `track_queue.py` | Cola de pistas en memoria (extraída para evitar import circular). |
| `tracklist.py` | `TrackQueue` + persistencia automática en JSON (`tracklist.json`). |
| `ui.py` | Dashboard Dear PyGui completo. |
| `logger.py` | Logger thread-safe con ring buffer y niveles de severidad. |

---

## Estado compartido — `state.py`

Contenedores de datos puros compartidos por todos los componentes del pipeline. Piezas clave:

- **Secciones de loop** — `loop_section_index`, `loop_sections` (derivadas de marcadores
  ordenados), más `get_loop_section()` / `step_loop_section()`.
- **`dispatch_command()`** — el único lugar que mapea strings de comando
  (`play`, `pause`, `next`, `prev`, `loop_toggle`, `loop_next`, `loop_prev`)
  a mutaciones de estado. `gesture_recognition`, `tempo_tapper` y el pedal
  llaman esto en lugar de llamar `play()` / `pause()` directamente.
- **Flags `skip_to_next` / `skip_to_prev`** — respetadas por `backing_track_thread`,
  para que los skips por gesto, tap o pedal realmente cambien la pista.
- **Ventana de override de tap** — `apply_tap_bpm()` / `apply_audio_bpm()` dejan que
  los golpes manuales del piezo "ganen" sobre el BPM de audio durante un período corto.
- **`last_bpm_analysis_dbg`** — actualizado por el hilo de audio en cada pasada,
  incluso cuando el modo tapper bloquea la escritura del BPM, para que el HUD no
  se quede estancado.

---

## E/S de audio — `audio_input.py`

No posee análisis ni carga de pistas — E/S pura:

- `audio_callback` — llamado por `sounddevice` en su propio hilo de OS; empuja muestras
  mono al ring buffer compartido y actualiza el RMS en `PhantomState`.
- `input_thread` — abre el `sounddevice.InputStream`; sale cuando se setea
  `STATE.io_restart_requested`.
- `playback_thread` — abre el `sounddevice.OutputStream`; vacía `audio_queue`.
- `io_manager_thread` — observa pedidos de reinicio de E/S desde la UI y cicla
  ambos hilos de stream a nuevos dispositivos.

---

## Detección de BPM — `audio_analysis.py`

Lee muestras crudas del micrófono desde el ring buffer compartido y escribe una
estimación de BPM suavizada y corregida por octava de vuelta en `PhantomState`.

**Cadena DSP:**

```
muestras crudas
  → filtro bandpass (40–8000 Hz)
  → HPSS (componente percusiva)
  → onset strength (agregado mediana)
  → autocorrelación completa
  → interpolación parabólica de pico
  → corrección de octava  (½× / 1× / 2× más cercano al anterior)
  → suavizado mediana (ventana de CFG.bpm_median_window beats)
  → STATE.apply_audio_bpm()
```

`MIN_BPM`, `MAX_BPM` y `ANALYZE_EVERY` se leen de `CFG`, por lo que el panel de
Configuración puede sobrescribirlos en tiempo de ejecución.

**Coexistencia con el tapper:** `bpm_analysis_thread` verifica `CFG.use_tempo_tapper`
en cada pasada y omite completamente la escritura del BPM derivado del audio mientras
el modo tapper está habilitado.

---

## Carga y reproducción de pistas — `audio_processing.py`

- `load_track` — carga cualquier archivo de audio via `librosa`; lee BPM desde un tag
  de metadatos o recurre a estimación con `beat_track`.
- `backing_track_thread` — loop de reproducción beat a beat con time-stretching en vivo
  (`pyrubberband`). Lee BPM en vivo desde `PhantomState`, estira cada bloque para
  que coincida, y lo empuja al `audio_queue` compartido para que el hilo de reproducción
  de `audio_input` lo vacíe.

**Phase-lock loop (PLL):** En cada beat, nuevos marcadores de tiempo son registrados
en una instancia de `PhaseLock`. Compara cada marcador detectado con una grilla extrapolada
y produce un multiplicador de corrección de velocidad para mantener la pista sincronizada
con el músico. Se resetea cada vez que se carga una nueva pista.

---

## Tempo tapper por piezo — `tempo_tapper.py`

Lee eventos de línea del Arduino sobre USB serial y enruta dos protocolos independientes
que comparten la misma conexión:

**Protocolo del tempo tapper** (`BPM:`/`TAP:` lines):

| Mensaje | Significado |
|---|---|
| `READY` | Enviado una vez al arrancar |
| `TAP:first` | Primer tap del par, sin BPM aún |
| `BPM:<float>` | Segundo tap en rango, ej. `BPM:128.4` |
| `TAP:out_of_range <bpm>` | Segundo tap fuera de 40–300 BPM, informativo |
| `TAP:timeout` | `waiting_second` limpiado tras `TIMEOUT_MS` |

**Protocolo del pedal** (`B<n>p`/`B<n>r` lines):

| Mensaje | Significado |
|---|---|
| `B1p` | Botón 1 presionado |
| `B1r` | Botón 1 soltado |
| `B2p` / `B2r` | Botón 2 presionado / soltado |
| `B3p` / `B3r` | Botón 3 presionado / soltado |

Las líneas de botones se reenvían a un `PedalController` (ver `pedal.py`). Pasar
`pedal=None` al thread hace que las líneas del pedal se reconozcan pero se ignoren,
exactamente como antes de que se agregara esta feature — sin cambios necesarios en
instalaciones sin pedal.

---

## Pedal de 3 botones — `pedal.py`

Módulo de lógica pura — sin import de serial. Las líneas del protocolo Arduino son
parseadas por `tempo_tapper.py` y reenviadas aquí via `PedalController.handle_event()`.
Esto refleja la separación `audio_analysis.py` / `audio_input.py`: DSP/lógica en un
módulo, E/S cruda en otro.

### Conceptos de eventos

| Tipo | Definición |
|---|---|
| **TAP** | Duración press→release menor a `CFG.pedal_hold_threshold_s` (default 0.35s) |
| **HOLD** | Duración press→release mayor o igual al umbral |
| **COMBO** | Dos o más botones cuyas ventanas `[press_time, release_time]` se superponen en algún instante (transitivamente) |

Un combo solo se resuelve cuando **todos** sus miembros han soltado, y solo si todos
concuerdan en tap vs. hold. Si discrepan (ej. B1 como tap y B2 como hold), el grupo
se descarta silenciosamente (se loguea en `debug`). Esto es intencional: una pulsación
simultánea imprecisa que mezcle tap y hold casi seguro no es un gesto deliberado.

### Nombres canónicos de eventos

```
B1_TAP   B1_HOLD
B2_TAP   B2_HOLD
B3_TAP   B3_HOLD
B1+B2_TAP    B1+B2_HOLD
B1+B3_TAP    B1+B3_HOLD
B2+B3_TAP    B2+B3_HOLD
B1+B2+B3_TAP B1+B2+B3_HOLD
```

Los números de botón dentro del nombre de un combo siempre están ordenados de forma
ascendente, independientemente del orden de pulsación.

### Parámetros configurables

| Parámetro CFG | Default | Descripción |
|---|---|---|
| `pedal_hold_threshold_s` | `0.35` | Umbral tap vs. hold en segundos |
| `pedal_stuck_timeout_s` | `5.0` | Timeout de seguridad: abandona un botón cuya señal de release nunca llega (ej. cable suelto) sin bloquear el pedal indefinidamente |

### Mapeo de comandos por defecto

| Evento | Comando |
|---|---|
| `B1_TAP` | `toggle` |
| `B2_TAP` | `next` |
| `B3_TAP` | `prev` |
| `B1_HOLD` | `loop_toggle` |
| `B2_HOLD` | `loop_next` |
| `B3_HOLD` | `loop_prev` |
| todos los combos | `none` (configurar desde la UI) |

Todo el mapeo es remapeable en tiempo de ejecución desde la ventana **PEDAL SETUP**
sin reiniciar. Cambiar el valor en la tabla escribe inmediatamente en `CFG.pedal_map`.

---

## Reconocimiento de gestos — `gesture_recognition.py`

**Pipeline de clasificación (por frame):**

1. `TRAINER.match(lm)` — vecino más cercano contra plantillas personalizadas guardadas.
   Un match dentro del umbral anula los integrados; su comando viene de `TRAINER.command_for(name)`.
2. `classify_gesture(lm)` — fallback basado en reglas para las cuatro formas integradas
   (mano abierta, puño, señalar con índice, paz/V).
3. El nombre resultante se estabiliza sobre 10 frames (`GestureStabilizer`) y luego
   se mantiene `CFG.gesture_hold_frames` frames antes de ejecutar via `state.dispatch_command()`.

**Gestos integrados:**

| Gesto | Forma | Comando por defecto |
|---|---|---|
| `PLAY` | mano abierta (5 dedos) | `play` |
| `PAUSE` | puño (0 dedos) | `pause` |
| `POINT` | solo índice (1 dedo) | `next` |
| `PEACE` | V / paz (2 dedos) | `loop_toggle` |

---

## Entrenamiento de gestos personalizados — `gesture_trainer.py` + `gesture_train_ui.py`

Reconocimiento de gestos estáticos invariante a rotación, entrenado desde clips de video
cortos en lugar de snapshots individuales.

### Extracción de features (19 floats por frame)

| Rango | Feature | Cantidad |
|---|---|---|
| `[0:5]` | Ratios de extensión (distancia tip–MCP / escala de palma) | 5 |
| `[5:10]` | Ángulos de curvatura PIP (radianes, 0 = recto) | 5 |
| `[10:14]` | Ángulos de apertura (3 pares adyacentes + 1 par exterior) | 4 |
| `[14]` | Ratio de oposición del pulgar | 1 |
| `[15:19]` | Distancias entre puntas de dedos (normalizadas) | 4 |

Todos los valores son ratios adimensionales o ángulos acotados calculados en una base
local anclada a la mano misma (muñeca → MCP del dedo medio como eje principal),
lo que los hace invariantes a la rotación de la cámara, traslación y escala de la mano.

### Almacenamiento

- **Pool binario** (`gesture_pool.bin`) — float32 IEEE-754 crudo; ~50–60% más pequeño
  que el formato JSON legacy, con tiempo de carga cayendo de ~10–20 ms a menos de 1 ms.
- **Sidecar de metadatos** (`gesture_pool.meta.json`) — nombres, comandos y conteos de clips legibles.
- Una matriz de pool pre-normalizada en caché (`_PoolCache`) se construye una vez tras
  la carga o tras `finish()` y nunca se reconstruye dentro de `match()`; `match()` en sí
  es un único producto matricial BLAS (`pool_normed @ query`).

### Flujo de entrenamiento

```
[COMENZAR SESIÓN] → [● GRABAR] (mantener mientras se mueve la mano) → [■ PARAR CLIP]
                     repetir desde distintos ángulos               → [GUARDAR GESTO]
```

Un gesto necesita al menos `MIN_CLIPS` (3) clips antes de poder guardarse. El matching
usa k-NN de similitud coseno (`K_NEIGHBORS = 7`) con voto mayoritario entre los vecinos
más cercanos.

---

## Entrada de video — `video_input.py`

Posee el `VideoCapture` de OpenCV y la sesión `HandLandmarker` de MediaPipe. No dibuja
nada y no clasifica gestos — todo eso vive en `gesture_recognition.py`.

```python
open_camera(cam_idx)    # → cv2.VideoCapture (lanza RuntimeError si falla)
read_frame(cap)          # → (frame_bgr, mp_image) o (None, None) en EOF/error
make_landmarker()        # → HandLandmarker como context manager
```

---

## Cola de pistas y persistencia — `track_queue.py` + `tracklist.py`

`TrackQueue` (en memoria) está definida en su propio módulo para que tanto `state.py`
como `tracklist.py` puedan importarla sin dependencia circular.

`PersistentQueue` la envuelve con persistencia automática en JSON en `tracklist.json`,
guardado junto al script:

```json
[
  {"name": "song.mp3", "path": "music/song.mp3", "bpm": 128.0, "duration": 214.5}
]
```

Los paths se guardan relativos al directorio que contiene `tracklist.json`, para que
el proyecto sea portátil al moverse.

---

## Dashboard — `ui.py`

Interfaz Dear PyGui de estilo dark rack-unit. Debe correr en el hilo principal.

**Paneles:**

- **Detección de BPM** — lectura de BPM en vivo, barra de ratio, fila de debug. Las
  píldoras `[AUDIO]` / `[TAP]` reflejan `snap["bpm_source"]` en vivo.
- **Nivel de entrada** — barras de forma de onda, medidores RMS/pico.
- **Pista de fondo** — controles de transporte, scrubber de línea de tiempo, marcadores.
- **Configuración** — E/S de audio, sliders de ganancia, dispositivo de video, mapeador
  de comandos de gestos, toggles de pedal y tempo tapper.
- **Time-Stretch** — editor de BPM de referencia, buffer fill, suavizado α.
- **Control de gestos** — feed de cámara en vivo (sin parpadeo), barra de hold.
- **Cola de pistas** — lista scrollable, editor de BPM inline, reordenar/cargar/quitar,
  persistido automáticamente en `tracklist.json`.
- **PEDAL SETUP** *(ventana emergente)* — indicadores en vivo de B1/B2/B3, umbral tap/hold,
  timeout de seguridad, tabla de remapeo completa para los 14 nombres de eventos
  (singles, pares y triple). Abre desde el header principal.
- **Entrenar gestos** *(ventana emergente)* — flujo de grabación de clips para gestos personalizados.
- **Log del sistema** — drain de log scrollable.

---

## Logger — `logger.py`

Logger thread-safe con ring buffer y niveles de severidad. Cada componente del pipeline
lo importa; la UI lo vacía una vez por frame para el panel de Log del Sistema.

---

## Configuración — `config.py`

Fuente única de verdad para todos los ajustes del usuario, persistida en `config.json`
junto al script. Cada otro módulo importa valores desde aquí en lugar de hardcodearlos;
el panel de Configuración de la UI lee y escribe este objeto directamente.

**Campos relevantes al pedal:**

| Campo | Default | Descripción |
|---|---|---|
| `use_pedal` | `False` | Habilita/deshabilita el dispatch de comandos del pedal |
| `pedal_map` | *(ver pedal.py)* | Dict `{event_name: command}` |
| `pedal_hold_threshold_s` | `0.35` | Umbral tap vs. hold |
| `pedal_stuck_timeout_s` | `5.0` | Timeout de seguridad para botones sin release |

---

## Buffers compartidos — `buffers.py`

Singletons a nivel de módulo para que `audio_input`, `audio_analysis` y
`audio_processing` compartan exactamente los mismos objetos de ring buffer y queue
sin imports circulares ni paso de argumentos:

```python
from buffers import audio_buffer, audio_queue
```
