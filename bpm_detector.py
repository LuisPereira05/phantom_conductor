import sounddevice as sd
import numpy as np
import librosa
from collections import deque
from scipy.signal import butter, lfilter
import time
import threading
import mido
from mido import Message
import mutagen # <<< agora com mutagen
from queue import Queue
import pyrubberband as pyrb
from mutagen.mp3 import MP3  # substitui WAVE


# ===================== CONFIG =====================
SR = 48000
BUFFER_SEC = 10
HOP_STREAM_SEC = 0.05
ANALYZE_EVERY_SEC = 0.25
HOP_LENGTH = 256
MIN_BPM, MAX_BPM = 70, 190
SMOOTH_ALPHA = 0.3
BANDPASS = (40, 3000)
ONSET_AGG = np.median
BLOCK_SEC = 1.0      # 1 segundo por bloco
OVERLAP_SEC = 0.05   # 50ms de overlap

# ===================== ESTADO =====================
buffer = deque(maxlen=int(SR * BUFFER_SEC))
last_analysis_time = 0.0
bpm_smooth = None
bpm_lock = threading.Lock()
midi_running = True
bpm_original = None  # <<< novo
audio_queue = Queue(maxsize=20)  # fila de blocos prontos


# ===================== UTILS =====================
def list_input_devices():
    devs = sd.query_devices()
    print("=== Dispositivos de ENTRADA ===")
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            print(f"{i:>3}  {d['name']}  (in:{d['max_input_channels']}  out:{d['max_output_channels']})")
    print("===============================")

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(1e-3, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(y, fs, lowcut, highcut):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, y)

def frames_to_bpm(lag_frames, sr=SR, hop=HOP_LENGTH):
    return 60.0 * sr / (hop * lag_frames)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def pick_peak_autocorr(ac, lag_min, lag_max):
    segment = ac[lag_min:lag_max+1]
    if segment.size == 0:
        return None
    i_rel = int(np.argmax(segment))
    i = lag_min + i_rel
    if i > 0 and i < len(ac) - 1:
        y0, y1, y2 = ac[i-1], ac[i], ac[i+1]
        denom = (y0 - 2*y1 + y2)
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            return i + delta
    return float(i)

def octave_correct(bpm_est, bpm_prev):
    if bpm_prev is None or not np.isfinite(bpm_prev):
        return bpm_est
    cands = [bpm_est/2, bpm_est, bpm_est*2]
    return min(cands, key=lambda x: abs(x - bpm_prev))

# ===================== Núcleo de ANÁLISE =====================
def estimate_bpm_from_buffer(y, sr=SR, bpm_prev=None):
    if len(y) < sr * 2:
        return None, {"msg": "buffer curto"}
    y = librosa.util.normalize(y.astype(np.float32))
    y = apply_bandpass(y, sr, BANDPASS[0], BANDPASS[1])
    y_h, y_p = librosa.effects.hpss(y)
    onset = librosa.onset.onset_strength(y=y_p, sr=sr, hop_length=HOP_LENGTH, aggregate=ONSET_AGG)
    if onset.size < 8 or np.max(onset) < 1e-3:
        return None, {"msg": "onset fraco/curto", "onset_max": float(np.max(onset))}
    
    ac_full = np.correlate(onset, onset, mode='full')
    ac = ac_full[ac_full.size // 2:]
    lag_min = int(np.floor((60.0 * sr) / (MAX_BPM * HOP_LENGTH)))
    lag_max = int(np.ceil((60.0 * sr) / (MIN_BPM * HOP_LENGTH)))
    lag_min = max(2, lag_min)
    lag_max = min(lag_max, len(ac) - 1)
    if lag_min >= lag_max:
        return None, {"msg": "faixa de lag inválida", "lag_min": lag_min, "lag_max": lag_max}
    
    lag_peak = pick_peak_autocorr(ac, lag_min, lag_max)
    if lag_peak is None or not np.isfinite(lag_peak) or lag_peak <= 0:
        return None, {"msg": "lag_peak inválido"}
    
    bpm_raw = frames_to_bpm(lag_peak, sr=sr, hop=HOP_LENGTH)
    bpm_corr = octave_correct(bpm_raw, bpm_prev)
    bpm_corr = clamp(bpm_corr, MIN_BPM, MAX_BPM)

    # --- Suavização condicional ---
    if bpm_prev is None or not np.isfinite(bpm_prev):
        bpm_s = bpm_corr
    else:
        if abs(bpm_corr - bpm_prev) < 5.0:  # tolerância de 5 BPM
            bpm_s = SMOOTH_ALPHA * bpm_corr + (1 - SMOOTH_ALPHA) * bpm_prev
        else:
            bpm_s = bpm_corr  # reseta se diferença grande

    debug = {
        "onset_max": float(np.max(onset)),
        "lag_peak": float(lag_peak),
        "bpm_raw": float(bpm_raw),
        "bpm_corr": float(bpm_corr)
    }

    return bpm_s, debug


# ===================== CALLBACK =====================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[audio] {status}")
    mono = np.mean(indata, axis=1)
    buffer.extend(mono)

# ===================== THREAD MIDI NOTE =====================
def midi_note_thread(outport, note=60, velocity=100):
    global bpm_smooth, midi_running
    while midi_running:
        with bpm_lock:
            bpm = bpm_smooth if bpm_smooth else 120.0
        interval = 60.0 / bpm
        outport.send(Message('note_on', note=note, velocity=velocity))
        time.sleep(interval * 0.1)
        outport.send(Message('note_off', note=note, velocity=0))
        time.sleep(interval * 0.9)

# ===================== TIME STRETCHING ======================


def process_backing_track_thread(mp3_file, sr, bpm_original, bpm_getter, audio_queue, midi_running_getter):
    """
    Thread que processa o backing track e envia blocos sincronizados no beat para a fila.
    
    mp3_file        : caminho do arquivo MP3
    sr              : sample rate
    bpm_original    : BPM do arquivo original
    bpm_getter      : função que retorna o BPM ao vivo
    audio_queue     : fila de blocos de áudio para reprodução
    midi_running_getter : função que retorna True enquanto o playback deve continuar
    """
    # --- carrega o track ---
    y_full, _ = librosa.load(mp3_file, sr=sr, mono=True)

    # tamanho de 1 beat em samples no BPM original
    beat_duration_sec = 60.0 / bpm_original
    beat_size = int(beat_duration_sec * sr)

    pos = 0
    next_beat_time = time.time()  # inicia imediatamente

    while pos < len(y_full) and midi_running_getter():
        # pega 1 beat do track
        end = min(pos + beat_size, len(y_full))
        block = y_full[pos:end].astype(np.float32)

        # pega o BPM ao vivo
        bpm_live = bpm_getter() or bpm_original
        rate = bpm_live / bpm_original  # time-stretch ratio

        # aplica time-stretch
        if len(block) > 1:
            block_stretched = pyrb.time_stretch(block, sr, rate=rate)
        else:
            block_stretched = block

        # espera até o próximo beat
        now = time.time()
        wait_time = next_beat_time - now
        if wait_time > 0:
            time.sleep(wait_time)

        # coloca na fila
        while midi_running_getter():
            try:
                audio_queue.put(block_stretched, timeout=0.1)
                break
            except:
                continue

        # atualiza posição e tempo do próximo beat
        pos += beat_size
        next_beat_time += 60.0 / bpm_live


# ========================= REPRODUÇÃO DA QUEUE =====================

def play_blocks_thread(audio_queue, sr, midi_running_getter):
    """
    Thread que consome blocos da fila e toca continuamente.
    """
    with sd.OutputStream(samplerate=sr, channels=1, dtype='float32') as out:
        while midi_running_getter():
            try:
                block = audio_queue.get(timeout=0.1).astype(np.float32)
                out.write(block)
            except:
                # sem bloco disponível, pausa curtinha
                time.sleep(0.01)


# ===================== MAIN =====================



def main():
    global last_analysis_time, bpm_smooth, midi_running, bpm_original, audio_queue

    # --- Seleção do arquivo backing track (MP3) ---
    mp3_file = input("Digite o caminho do backing track (.mp3): ").strip()
    bpm_original = None
    try:
        audio = MP3(mp3_file)
        tags = audio.tags
        if tags:
            print("=== Metadados encontrados ===")
            for k, v in tags.items():
                print(f"{k} : {v}")

            if "TBPM" in tags:
                bpm_original = float(tags["TBPM"].text[0])
    except Exception as e:
        print(f"Não consegui ler BPM do arquivo (mutagen MP3): {e}")
        bpm_original = None

    if bpm_original:
        print(f"📀 BPM original embutido no arquivo: {bpm_original:.2f}")
    else:
        bpm_original = float(input("BPM não encontrado. Digite o BPM original manualmente: "))

    # --- Seleção do dispositivo de entrada ---
    list_input_devices()
    dev_idx = int(input("Selecione o índice do dispositivo de ENTRADA: ").strip())
    dev = sd.query_devices(dev_idx)
    print(f"\n🎤 Usando: {dev['name']}  | SR={SR} Hz  | buffer={BUFFER_SEC}s")
    print("Ctrl+C para parar.\n")

    blocksize = int(SR * HOP_STREAM_SEC)

    # --- Threads ---
    # thread de processamento do backing track
    threading.Thread(
        target=process_backing_track_thread,
        args=(mp3_file, SR, bpm_original, lambda: bpm_smooth, audio_queue, lambda: midi_running),
        daemon=True
    ).start()

    # thread de reprodução contínua
    threading.Thread(
        target=play_blocks_thread,
        args=(audio_queue, SR, lambda: midi_running),
        daemon=True
    ).start()

    # --- Thread principal de captura de áudio e análise de BPM ---
    with sd.InputStream(device=dev_idx, channels=1, samplerate=SR,
                        blocksize=blocksize, callback=audio_callback):
        try:
            while True:
                now = time.time()
                if now - last_analysis_time >= ANALYZE_EVERY_SEC and len(buffer) >= SR * 2:
                    last_analysis_time = now
                    y = np.array(buffer, dtype=np.float32)
                    rms = float(np.sqrt(np.mean(y**2)))
                    peak = float(np.max(np.abs(y)))
                    bpm_new, dbg = estimate_bpm_from_buffer(y, sr=SR, bpm_prev=bpm_smooth)
                    if bpm_new is not None and np.isfinite(bpm_new):
                        with bpm_lock:
                            bpm_smooth = bpm_new
                        ratio = bpm_smooth / bpm_original
                        print(f"🎵 BPM live: {dbg['bpm_raw']:.2f}  | ref: {bpm_original:.2f}  "
                              f"| ratio:{ratio:.3f}   "
                              f"(rms:{rms:.4f} peak:{peak:.3f}  "
                              f"raw:{dbg['bpm_raw']:.2f} corr:{dbg['bpm_corr']:.2f})", end="\r")
                    else:
                        msg = dbg.get("msg", "indefinido") if dbg else "indefinido"
                        print(f"… sem estimativa (rms:{rms:.4f}, peak:{peak:.3f})  {msg:>20}", end="\r")
                sd.sleep(int(ANALYZE_EVERY_SEC * 1000 // 2))

        except KeyboardInterrupt:
            print("\n🛑 Encerrado.")
            midi_running = False
            time.sleep(0.2)


if __name__ == "__main__":
    main()