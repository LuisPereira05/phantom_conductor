"""
Phantom Conductor — Teste de Pipeline de Onset Detection
==========================================================
Pipeline de teste:

  1) Carrega um .wav + seu _onsets.json (gerados por record_onset_data.py)
  2) "Toca" o áudio em blocos, empurrando amostras em buffers.audio_buffer
     exatamente como audio_input.audio_callback faria ao vivo
  3) Roda a MESMA lógica de bpm_analysis_thread (audio_analysis.py) — janela
     deslizante, RMS gate, estimate_bpm(), dedup de onsets — mas com um
     relógio VIRTUAL (posição no arquivo / sample rate) em vez de
     time.time()/time.sleep(). Isso reproduz fielmente o comportamento "ao
     vivo" sem depender do tempo real de execução nem do tempo real do
     áudio, rodando determinístico e rápido.
  4) Compara os onsets detectados com os taps reais (ground truth) usando
     uma janela de tolerância (±50 ms por padrão), casamento guloso do
     vizinho mais próximo, calcula TP / FP / FN, precisão, recall e F1, e
     plota tudo (forma de onda + marcadores + métricas) num PNG.

Coloque este script na raiz do projeto (mesmo nível de audio_analysis.py,
buffers.py, config.py etc.) antes de rodar.

Uso
---
    python test_onset_pipeline.py --wav take_001.wav --onsets take_001_onsets.json

    python test_onset_pipeline.py --wav take_001.wav --onsets take_001_onsets.json \
        --tolerance 0.05 --chunk 512 --out take_001_eval.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from audio_analysis import estimate_bpm
from buffers import SR as BUF_SR
from buffers import audio_buffer
from config import CFG

# ═══════════════════════════════════════════════════════════════════════════
#  CARREGAMENTO DE ÁUDIO
# ═══════════════════════════════════════════════════════════════════════════


def load_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        y = data.copy()
    else:
        y = data.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULAÇÃO "AO VIVO" — reaproveita estimate_bpm() + a lógica exata de
#  bpm_analysis_thread, trocando o relógio de parede real por um virtual.
# ═══════════════════════════════════════════════════════════════════════════


def simulate_live_detection(
    y: np.ndarray,
    sr: int,
    chunk: int,
    onset_wait_s: float | None = None,
    onset_delta: float | None = None,
    _warned: list | None = None,
) -> list[float]:
    """
    Simulates bpm_analysis_thread using a virtual clock and returns the
    onset timestamps that would have been written to state.recent_beat_times.
    """

    if _warned is None:
        _warned = []

    if sr != BUF_SR:
        raise ValueError(
            f"sample rate do wav ({sr}) difere de buffers.SR ({BUF_SR}) — "
            f"re-exporte o take em {BUF_SR} Hz."
        )

    audio_buffer.clear()

    analyze_every = CFG.analyze_every
    rms_threshold = CFG.get("rms_threshold", 0.01)

    min_onset_gap_s = onset_wait_s if onset_wait_s is not None else 60.0 / CFG.max_bpm

    last_analysis_t = 0.0
    last_emitted_onset_wall_time: float | None = None
    detected_onsets: list[float] = []

    n = len(y)
    pos = 0

    while pos < n:
        block = y[pos : pos + chunk]
        pos += len(block)
        audio_buffer.extend(block)

        virtual_now = pos / sr

        if virtual_now - last_analysis_t < analyze_every:
            continue

        if len(audio_buffer) < sr * 2:
            continue

        last_analysis_t = virtual_now

        window = np.array(audio_buffer, dtype=np.float32)

        rms = float(np.sqrt(np.mean(window**2)))
        if rms < rms_threshold:
            continue

        try:
            bpm_new, dbg = estimate_bpm(
                window,
                sr=sr,
                bpm_original=None,
                logger=None,
                onset_wait_s=onset_wait_s,
                onset_delta=(onset_delta if onset_delta is not None else 0.1),
            )
        except TypeError as e:
            if "unexpected keyword argument" not in str(e):
                raise

            bpm_new, dbg = estimate_bpm(
                window,
                sr=sr,
                bpm_original=None,
                logger=None,
            )

        if bpm_new is None or not np.isfinite(bpm_new):
            continue

        frame_times = dbg.get("onset_frame_times")
        if not frame_times:
            continue

        window_start_wall = virtual_now - (len(window) / sr)

        for ft in frame_times:
            onset_wall_time = window_start_wall + ft

            if (
                last_emitted_onset_wall_time is None
                or onset_wall_time - last_emitted_onset_wall_time >= min_onset_gap_s
            ):
                detected_onsets.append(onset_wall_time)
                last_emitted_onset_wall_time = onset_wall_time

    return detected_onsets


# ═══════════════════════════════════════════════════════════════════════════
#  CASAMENTO GT × DETECTADOS  +  MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════


def match_events(
    ground_truth: list[float], detected: list[float], tolerance: float
) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    """
    Casamento guloso do vizinho mais próximo dentro de `tolerance` segundos.
    Retorna (pares_TP, falsos_positivos, falsos_negativos).
    """
    gt = sorted(ground_truth)
    det = sorted(detected)
    used = [False] * len(det)
    tp_pairs: list[tuple[float, float]] = []
    fn: list[float] = []

    for g in gt:
        best_i, best_d = None, tolerance + 1e-9
        for i, d in enumerate(det):
            if used[i]:
                continue
            dist = abs(d - g)
            if dist <= tolerance and dist < best_d:
                best_i, best_d = i, dist
        if best_i is not None:
            used[best_i] = True
            tp_pairs.append((g, det[best_i]))
        else:
            fn.append(g)

    fp = [det[i] for i in range(len(det)) if not used[i]]
    return tp_pairs, fp, fn


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════════════════


def plot_results(
    y: np.ndarray,
    sr: int,
    ground_truth: list[float],
    detected: list[float],
    tp_pairs: list[tuple[float, float]],
    fp: list[float],
    fn: list[float],
    tolerance: float,
    out_path: str,
):
    n_tp, n_fp, n_fn = len(tp_pairs), len(fp), len(fn)
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    fig, (ax_wave, ax_bar) = plt.subplots(
        2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]}
    )

    t = np.arange(len(y)) / sr
    ax_wave.plot(t, y, color="#888888", linewidth=0.5, zorder=1)

    for g, _ in tp_pairs:
        ax_wave.axvline(g, color="green", alpha=0.6, linewidth=1.2, zorder=2)
    for d in fp:
        ax_wave.axvline(
            d, color="red", linestyle="--", alpha=0.8, linewidth=1.2, zorder=2
        )
    for g in fn:
        ax_wave.axvline(
            g, color="orange", linestyle=":", alpha=0.9, linewidth=1.6, zorder=2
        )

    ax_wave.set_xlim(0, t[-1] if len(t) else 1)
    ax_wave.set_xlabel("tempo (s)")
    ax_wave.set_ylabel("amplitude")
    ax_wave.set_title(
        f"Onsets detectados vs. taps reais  (tolerância ±{tolerance * 1000:.0f} ms)"
    )

    handles = [
        plt.Line2D([0], [0], color="green", lw=2, label=f"Acerto (TP) — {n_tp}"),
        plt.Line2D(
            [0], [0], color="red", lw=2, ls="--", label=f"Falso positivo (FP) — {n_fp}"
        ),
        plt.Line2D(
            [0],
            [0],
            color="orange",
            lw=2,
            ls=":",
            label=f"Falta / falso negativo (FN) — {n_fn}",
        ),
    ]
    ax_wave.legend(handles=handles, loc="upper right")

    metrics = ["Precisão", "Recall", "F1"]
    values = [precision, recall, f1]
    bars = ax_bar.bar(metrics, values, color=["#3a7a4a", "#3a6ea5", "#a5723a"])
    ax_bar.set_ylim(0, 1.05)
    for b, v in zip(bars, values):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    ax_bar.set_title(
        f"TP={n_tp}  FP={n_fp}  FN={n_fn}   "
        f"(taps reais={len(ground_truth)}, detectados={len(detected)})"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    print(f"\nSalvo: {out_path}")
    print(f"Precisão: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")
    print(f"TP={n_tp}  FP={n_fp}  FN={n_fn}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description="Testa a detecção de onsets do pipeline contra taps reais do tempo tapper"
    )
    ap.add_argument(
        "--wav", required=True, help=".wav gravado com record_onset_data.py"
    )
    ap.add_argument(
        "--onsets", required=True, help="_onsets.json correspondente ao take"
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="janela de acerto em segundos (default: 0.05 = ±50ms)",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=512,
        help="tamanho do bloco simulado em amostras, como se fosse o callback de áudio (default: 512)",
    )
    ap.add_argument(
        "--out", default=None, help="caminho do PNG de saída (default: <wav>_eval.png)"
    )
    ap.add_argument("--sweep", action="store_true")

    ap.add_argument("--wait-min", type=float, default=0.08)
    ap.add_argument("--wait-max", type=float, default=0.18)
    ap.add_argument("--wait-step", type=float, default=0.01)

    ap.add_argument("--delta-min", type=float, default=0.02)
    ap.add_argument("--delta-max", type=float, default=0.25)
    ap.add_argument("--delta-step", type=float, default=0.02)
    args = ap.parse_args()

    y, sr = load_wav_mono_f32(args.wav)

    with open(args.onsets, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    ground_truth = gt_data["host_onset_times"]

    print(
        f"Áudio: {len(y) / sr:.1f}s @ {sr} Hz   |   Taps reais (GT): {len(ground_truth)}"
    )

    if args.sweep:
        best = None

        wait_values = np.arange(
            args.wait_min,
            args.wait_max + 1e-9,
            args.wait_step,
        )

        delta_values = np.arange(
            args.delta_min,
            args.delta_max + 1e-9,
            args.delta_step,
        )

        for wait in wait_values:
            for delta in delta_values:
                detected = simulate_live_detection(
                    y,
                    sr,
                    args.chunk,
                    onset_wait_s=wait,
                    onset_delta=delta,
                )

                tp_pairs, fp, fn = match_events(
                    ground_truth,
                    detected,
                    args.tolerance,
                )

                tp = len(tp_pairs)
                fp_n = len(fp)
                fn_n = len(fn)

                precision = tp / (tp + fp_n) if tp + fp_n else 0
                recall = tp / (tp + fn_n) if tp + fn_n else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0
                )

                print(
                    f"wait={wait:.3f}  "
                    f"delta={delta:.3f}  "
                    f"TP={tp:2d}  "
                    f"FP={fp_n:2d}  "
                    f"FN={fn_n:2d}  "
                    f"P={precision:.3f}  "
                    f"R={recall:.3f}  "
                    f"F1={f1:.3f}"
                )

                if best is None or f1 > best["f1"]:
                    best = {
                        "wait": wait,
                        "delta": delta,
                        "tp": tp,
                        "fp": fp_n,
                        "fn": fn_n,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        # save everything needed for plotting
                        "detected": detected,
                        "tp_pairs": tp_pairs,
                        "fp_events": fp,
                        "fn_events": fn,
                    }

        print("\nBest parameters:")
        print(best)

        plot_results(
            y=y,
            sr=sr,
            ground_truth=ground_truth,
            detected=best["detected"],
            tp_pairs=best["tp_pairs"],
            fp=best["fp_events"],
            fn=best["fn_events"],
            tolerance=args.tolerance,
            out_path=(f"best_wait_{best['wait']:.3f}_delta_{best['delta']:.3f}.png"),
        )

        return

    # ---------- Normal single evaluation ----------

    detected = simulate_live_detection(y, sr, args.chunk)

    print(f"Onsets detectados pelo pipeline: {len(detected)}")

    tp_pairs, fp, fn = match_events(
        ground_truth,
        detected,
        args.tolerance,
    )

    out_path = args.out or f"{Path(args.wav).with_suffix('')}_eval.png"

    plot_results(
        y,
        sr,
        ground_truth,
        detected,
        tp_pairs,
        fp,
        fn,
        args.tolerance,
        out_path,
    )


if __name__ == "__main__":
    main()
