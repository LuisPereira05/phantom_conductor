# Phantom Conductor

A real-time conducting interface for backing tracks: control playback with
hand gestures or a footswitch, and have the track's tempo follow either a
live BPM detector listening to a microphone, or manual foot-taps from an
Arduino piezo pedal.

Everything runs as a set of daemon threads sharing one thread-safe state
object, with a Dear PyGui dashboard on the main thread.

```
pip install dearpygui mutagen sounddevice
```

---

## Project layout

```
phantom_conductor/
├─ main.py                  — entry point; starts all threads, then dpg.run()
│
├─ state.py                 — shared thread-safe state + track queue
├─ config.py                — settings, persisted to phantom_config.json
├─ buffers.py                — shared ring buffer / queue singletons
├─ logger.py                — thread-safe ring-buffer logger
│
├─ audio_input.py           — mic/speaker I/O (sounddevice)
├─ audio_analysis.py        — BPM detection DSP chain
├─ audio_processing.py      — track loading + beat-synced playback
├─ tempo_tapper.py          — Arduino foot-tapper serial bridge
│
├─ video_input.py           — OpenCV camera + MediaPipe HandLandmarker
├─ gesture_recognition.py   — gesture classification + HUD overlay
├─ gesture_trainer.py       — custom gesture feature extraction/matching
├─ gesture_train_ui.py      — Dear PyGui training workflow mixin
│
├─ track_queue.py           — in-memory TrackQueue
├─ tracklist.py             — TrackQueue + JSON persistence
├─ ui.py                    — Dear PyGui dashboard (main thread)
│
├─ audio/                   — track library
├─ hand_landmarker.task     — MediaPipe model (auto-downloaded if missing)
├─ requirements.txt
│
├─ phantom_config.json      — CFG persistence (config.py)
├─ tracklist.json           — track queue persistence (tracklist.py)
├─ gesture_pool.bin         — trained gesture vectors (gesture_trainer.py)
├─ gesture_pool.meta.json   — gesture names/commands sidecar
└─ gesture_templates.json   — legacy gesture format, migrated on first load
```

## Thread map

```
main.py
 ├─ bpm-analysis    → audio_analysis.bpm_analysis_thread
 ├─ backing-track   → audio_processing.backing_track_thread
 ├─ io-manager      → audio_input.io_manager_thread
 │                     ├─ audio-in   → audio_input.input_thread
 │                     └─ audio-out  → audio_input.playback_thread
 ├─ gesture-vision  → gesture_recognition.gesture_vision_thread
 │                     (internally uses video_input helpers)
 └─ tempo-tapper    → tempo_tapper.tempo_tapper_thread
                       (reads BPM taps from the Arduino piezo over serial)

Main thread → ui.PhantomUI.run()   (Dear PyGui must run on the main thread)
```

Camera index is read from `phantom_config.json` by default; a command-line
override still works:

```
python main.py 2
```

The tempo-tapper thread always starts, even with no tapper configured —
it checks `CFG.use_tempo_tapper` live on every serial line, so the Settings
checkbox can enable or disable it at runtime with no restart. If no Arduino
is plugged in, it just logs a periodic "no serial port found — retrying…"
warning and is otherwise harmless to leave running.

---

## Module map

| Module | Responsibility |
|---|---|
| `state.py` | Thread-safe shared state and track queue. No I/O, no threads, no GUI. |
| `config.py` | Single source of truth for user-adjustable settings, persisted to `config.json`. |
| `buffers.py` | Shared ring buffer / queue singletons so audio modules avoid circular imports. |
| `audio_input.py` | Pure I/O: mic callback, input/output streams, device-restart watcher. |
| `audio_analysis.py` | BPM detection DSP chain from raw mic samples. |
| `audio_processing.py` | Track loading and beat-synchronous backing-track playback with live time-stretching. |
| `tempo_tapper.py` | Serial bridge to the Arduino foot-tapper. |
| `gesture_recognition.py` | Hand-gesture classification (custom + built-in) and HUD overlay. |
| `gesture_trainer.py` | Rotation-invariant feature extraction and custom gesture storage/matching. |
| `gesture_train_ui.py` | Dear PyGui mixin for the gesture training workflow. |
| `video_input.py` | Owns the OpenCV camera and MediaPipe HandLandmarker session. |
| `track_queue.py` | In-memory track queue (extracted to avoid a circular import). |
| `tracklist.py` | `TrackQueue` + automatic JSON persistence (`tracklist.json`). |
| `ui.py` | The full Dear PyGui dashboard. |
| `logger.py` | Thread-safe ring-buffer logger with severity levels. |

---

## Shared state — `state.py`

Pure data containers shared by every pipeline component. Key pieces:

- **Loop sections** — `loop_section_index`, `loop_sections` (derived from
  sorted markers), plus `get_loop_section()` / `step_loop_section()`.
- **`dispatch_command()`** — the single place that maps command strings
  (`play`, `pause`, `next`, `prev`, `loop_toggle`, `loop_next`, `loop_prev`)
  to state mutations. `gesture_recognition` calls this instead of calling
  `play()` / `pause()` directly.
- **`skip_to_next` / `skip_to_prev` flags** — honoured by
  `backing_track_thread`, so gesture- or tap-triggered skips actually move
  the track, not just flip a flag nothing reads.
- **Tap-override window** — `apply_tap_bpm()` / `apply_audio_bpm()` let
  manual foot-taps "win" over the smoothed audio BPM for a short period,
  instead of both writers fighting over `set_bpm()`. `set_bpm()` itself is
  untouched. `tap_connected` exposes serial status to the Settings/HUD panel.
- **`last_bpm_analysis_dbg`** — kept fresh by the audio thread every pass
  regardless of whether tapper mode gates the actual BPM write, so HUD/debug
  views don't go stale while tapper mode is active. The live
  `CFG.use_tempo_tapper` check itself lives in `audio_analysis.py` and
  `tempo_tapper.py`, checked every loop pass in both — the override window
  here is a secondary safety net, not the only thing keeping the two BPM
  sources from fighting.

---

## Audio I/O — `audio_input.py`

Owns **no** analysis and **no** track loading — pure I/O:

- `audio_callback` — called by `sounddevice` on its own OS thread; pushes
  mono samples into the shared ring buffer and updates the RMS waveform on
  `PhantomState`.
- `input_thread` — opens the `sounddevice.InputStream`; exits when
  `STATE.io_restart_requested` is set.
- `playback_thread` — opens the `sounddevice.OutputStream`; drains
  `audio_queue`.
- `io_manager_thread` — watches for I/O restart requests from the UI and
  cycles both stream threads onto new devices.

---

## BPM detection — `audio_analysis.py`

Reads raw microphone samples from the shared ring buffer and writes a
smoothed, octave-corrected BPM estimate back to `PhantomState`.

**DSP chain:**

```
raw samples
  → bandpass filter (40–3000 Hz)
  → HPSS (percussive component)
  → onset strength (median aggregate)
  → full autocorrelation
  → parabolic peak interpolation
  → octave correction  (½× / 1× / 2× closest to previous)
  → exponential smoothing (α = CFG.smooth_alpha)
  → STATE.apply_audio_bpm()
```

`MIN_BPM`, `MAX_BPM`, `SMOOTH_ALPHA`, and `ANALYZE_EVERY` are read from
`CFG` so the Settings panel can override them at runtime.

**Tempo-tapper coexistence:** `bpm_analysis_thread` checks
`CFG.use_tempo_tapper` on every pass, not just at startup, and skips writing
audio-derived BPM entirely while tapper mode is enabled. Previously the only
thing preventing audio from overwriting a tap was the 4 s override window in
`state.apply_audio_bpm()`, so the mic would silently take back over a few
seconds after every tap. Checking the live config flag means flipping the
Settings checkbox takes effect immediately in both directions, with no
thread restart needed. The thread still calls `estimate_bpm()` even while
gated off, so `bpm_analysis_dbg` stays fresh for diagnostics — it just
doesn't push the result into state when tapper mode owns the BPM.

---

## Track loading & playback — `audio_processing.py`

- `load_track` — loads any audio file via `librosa`; reads BPM from a
  metadata tag or falls back to `beat_track` estimation.
- `backing_track_thread` — beat-by-beat playback loop with live
  time-stretching (`pyrubberband`). Reads live BPM from `PhantomState`,
  stretches each beat block to match, and pushes it to the shared
  `audio_queue` for `audio_input`'s playback thread to drain.

Reads from `PhantomState` (BPM, gain, flags) and writes only track-progress
fields (position, duration, `bpm_original`). Never touches `sounddevice`
directly — that's `audio_input`'s job.

**Skip handling:** `backing_track_thread` now actually consumes
`state.skip_to_next` / `state.skip_to_prev`. Previously
`state.dispatch_command("next"/"prev")` only set these flags — nothing
downstream read them, so gesture- or tap-triggered next/prev silently did
nothing even though the dispatch itself worked correctly (visible in the
log as `gesture [...] POINT -> next` with no actual track change). The flag
is read-and-cleared under the lock in one step — the same pattern already
used for `load_new_track` — so a gesture firing twice in quick succession
can't queue up two skips. Routed through `state.queue.next_track()` /
`prev_track()`, the same calls the UI's Prev/Next buttons already use, so
behavior stays consistent regardless of whether the skip came from a
button, a gesture, or a second tapper input in the future.

---

## Foot tempo tapper — `tempo_tapper.py`

Reads line-based BPM events from the Arduino piezo tapper over USB serial
and writes them into `PhantomState` via `state.apply_tap_bpm()` — the same
pairing `audio_analysis.py` uses via `apply_audio_bpm()`.

**Wire protocol** (from `tempo_tapper.ino`):

| Message | Meaning |
|---|---|
| `READY` | Sent once on boot |
| `TAP:first` | First tap of a new pair, no BPM yet |
| `BPM:<float>` | Second tap landed in range, e.g. `BPM:128.4` |
| `TAP:out_of_range <bpm>` | Second tap outside 40–300 BPM, informational |
| `TAP:timeout` | `waiting_second` cleared after `TIMEOUT_MS` |

**Why a "source" flag:** `bpm_analysis_thread` (audio) and this thread both
want to own `state.bpm_live`. Coordination happens in two layers:

1. `CFG.use_tempo_tapper` is checked **live** by both threads, every pass.
   - `audio_analysis_thread` skips `apply_audio_bpm()` entirely while this
     is `True`.
   - This thread still keeps the serial port open and drains incoming
     lines while it's `False` (so the OS read buffer doesn't back up and
     nothing is lost when it's flipped back on), but doesn't call
     `state.apply_tap_bpm()` until it's `True` again.
2. `state.apply_tap_bpm()` / `apply_audio_bpm()` still maintain the short
   override window as a second line of defence — useful if someone flips
   the checkbox mid-tap, or a stray audio analysis pass lands in the same
   instant a tap comes in.

Both checks matter: (1) makes the checkbox mean something continuously, not
just at thread startup; (2) avoids a race in the brief moment around the
toggle. This module never decides PLAY/PAUSE — it only ever writes BPM. A
second piezo/button for transport should route through
`state.dispatch_command()`, the same way `gesture_recognition` does.

---

## Gesture recognition — `gesture_recognition.py`

**Classification pipeline (per frame):**

1. `TRAINER.match(lm)` — nearest-neighbour against saved custom templates.
   A match within threshold overrides the built-ins; its command comes from
   `TRAINER.command_for(name)`.
2. `classify_gesture(lm)` — rule-based fallback for the four built-in
   shapes (open hand, fist, index point, peace/V).
3. The resulting name is stabilised over 10 frames (`GestureStabilizer`)
   then held for `CFG.gesture_hold_frames` before executing via
   `state.dispatch_command()`.

**Built-in gestures:**

| Gesture | Shape | Default command |
|---|---|---|
| `PLAY` | open hand (5 fingers) | `play` |
| `PAUSE` | fist (0 fingers) | `pause` |
| `POINT` | index only (1 finger) | `next` |
| `PEACE` | V / peace (2 fingers) | `loop_toggle` |

Also includes `_draw_hand` / `_draw_hud` (OpenCV drawing helpers for the
camera window) and `gesture_vision_thread`, the main loop that reads frames
via `video_input`, classifies gestures, writes to `PhantomState`, and
displays the annotated camera window.

---

## Custom gesture training — `gesture_trainer.py` + `gesture_train_ui.py`

Rotation-invariant static gesture recognition, trained from short video
clips rather than single snapshots.

### Feature extraction (19 floats per frame)

| Range | Feature | Count |
|---|---|---|
| `[0:5]` | Extension ratios (tip–MCP distance / palm scale) | 5 |
| `[5:10]` | PIP curl angles (radians, 0 = straight) | 5 |
| `[10:14]` | Spread angles (3 adjacent finger-base pairs + 1 outer pair) | 4 |
| `[14]` | Thumb-opposition ratio | 1 |
| `[15:19]` | Fingertip pairwise distances (normalised) | 4 |

All values are dimensionless ratios or bounded angles computed in a local
basis anchored to the hand itself (wrist → middle-MCP as the primary axis),
making them invariant to camera rotation, hand translation, and scale.

### Storage

- **Binary pool** (`gesture_pool.bin`) — raw IEEE-754 float32, no text
  parsing on load; roughly 50–60% smaller on disk than the legacy JSON
  format, with load time dropping from ~10–20 ms to under 1 ms.
- **Meta sidecar** (`gesture_pool.meta.json`) — human-readable names,
  commands, and clip counts.
- A cached, pre-normalized pool matrix (`_PoolCache`) is built once after
  load or after `finish()` and never rebuilt inside `match()`; row norms are
  pre-computed at cache-build time. `match()` itself is a single BLAS
  matrix–vector multiply (`pool_normed @ query`), running at
  memory-bandwidth speed.
- Legacy `gesture_templates.json` files are read once for migration. Any
  gesture whose stored vectors don't match the current feature-vector
  length is flagged `needs_retrain` and kept (name/command preserved) with
  an empty pool, rather than silently producing garbage similarity scores.

### Training workflow

```
[START SESSION] → [● REC] (hold while moving hand) → [■ STOP CLIP]
                   repeat from different angles      → [SAVE GESTURE]
```

A gesture needs at least `MIN_CLIPS` (3) clips, recorded from different
angles, before it can be saved — this is what gives the matcher enough
variety to recognize the pose under rotation. Matching is cosine-similarity
k-NN (`K_NEIGHBORS = 7`) with majority vote among the nearest neighbours.

The training UI's recording flag is `state.clip_recording_active`; the
vision thread calls `TRAINER.capture_frame(lm)` every frame while it's set.

---

## Video input — `video_input.py`

Owns the OpenCV `VideoCapture` and the MediaPipe `HandLandmarker` session.
Does **not** draw anything and does **not** classify gestures — all of that
lives in `gesture_recognition.py`.

```python
open_camera(cam_idx)          # → cv2.VideoCapture (raises RuntimeError on failure)
read_frame(cap)                # → (frame_bgr, mp_image) or (None, None) on EOF/error
make_landmarker()              # → HandLandmarker context manager
detect(landmarker, mp_image)   # → HandLandmarkerResult
```

---

## Track queue & persistence — `track_queue.py` + `tracklist.py`

`TrackQueue` (in-memory) is defined in its own module so both `state.py`
and `tracklist.py` can import it without a circular dependency.

`PersistentQueue` wraps it with automatic JSON persistence to
`tracklist.json`, stored next to the script:

```json
[
  {"name": "song.mp3", "path": "music/song.mp3", "bpm": 128.0, "duration": 214.5}
]
```

Paths are stored relative to the directory containing `tracklist.json`, so
the project stays portable when moved.

```python
from tracklist import PersistentQueue

q = PersistentQueue()                                  # loads saved list automatically
q.add("/abs/path/to/song.mp3", bpm=128.0, duration=214.5)
q.remove(idx)
q.snapshot()      # → (list_of_dicts, current_index)
q.load_state()    # reload from disk (e.g. after external edit)
```

All `TrackQueue` methods still work — `PersistentQueue` subclasses it.

---

## Dashboard — `ui.py`

Dark rack-unit style Dear PyGui interface. Must run on the main thread.

**Panels:**

- **BPM Detection** — live BPM readout, ratio bar, debug row. The
  `[AUDIO]` / `[TAP]` pills reflect `snap["bpm_source"]` live: whichever
  source actually wrote `bpm_live` lights up, the other dims — paired with
  the live `CFG.use_tempo_tapper` gating in `audio_analysis.py` /
  `tempo_tapper.py`, so the pills show which source is actually in control,
  not just whether the ratio is synced.
- **Input Level** — waveform bars, RMS / peak meters.
- **Backing Track** — transport controls, timeline scrubber, markers.
- **Settings** — audio I/O, gain sliders, video device, hand command
  mapper, pedal toggle, tempo-tapper toggle. Audio I/O, gain, camera index,
  gesture map, pedal, and tempo-tapper settings all live in one collapsible
  panel rather than being split across two.
- **Time-Stretch** — reference BPM editor, buffer fill, smoothing α.
- **Gesture Control** — live camera feed (flicker-free), hold bar.
- **Track Queue** — scrollable list, inline BPM editor, reorder/load/remove,
  persisted to `tracklist.json` automatically.
- **System Log** — scrollable log drain.

**Camera feed performance:** the feed no longer flashes —
texture upload is rate-limited to a max of 30 fps via a frame counter,
upload converts BGR→RGBA once into a pre-allocated buffer, and gesture-draw
items are updated with `dpg.configure_item` instead of being deleted and
redrawn every frame (eliminating the one-frame blank flash).

---

## Logging — `logger.py`

Thread-safe ring-buffer logger with severity levels. Every pipeline
component imports it; the UI drains it once per frame for the System Log
panel.

---

## Configuration — `config.py`

Single source of truth for all user-adjustable settings, persisted to
`config.json` next to the script. Every other module imports values from
here rather than hard-coding them; the UI's Settings panel reads and writes
this object directly.

---

## Shared buffers — `buffers.py`

Module-level singletons so `audio_input`, `audio_analysis`, and
`audio_processing` all share the exact same ring buffer and queue objects
without circular imports or argument passing:

```python
from buffers import audio_buffer, audio_queue
```
