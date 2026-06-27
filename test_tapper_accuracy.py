"""
Phantom Conductor — Foot Tapper Diagnostic
===========================================
Evaluates a single take produced by record_onset_data.py
(<out>_onsets.json) and reports timing-consistency and detection-health
stats for the foot tapper, with a 4-panel matplotlib view.

IMPORTANT — what this script can and can't tell you
-----------------------------------------------------
This take has no independent ground truth (no click track, no
audio-derived onset list) — the only thing on disk is the tapper's own
tap log, recorded two ways:

    host_onset_times   wall-clock arrival time of each "TAP:<millis>"
                        line, relative to recording start (this is
                        YOUR rhythm, filtered through the OS/serial
                        stack)
    device_millis       Arduino's own millis() at the moment the piezo
                        crossed threshold (this is the FIRMWARE's
                        clock, with no shared epoch with the host)

Because the two clocks don't share an origin, you cannot subtract them
to get an absolute "detection latency" number — that would silently
fabricate a result. What IS comparable between them is the *shape* of
the timing: if you tapped at intervals of 500, 510, 495 ms, the device
clock should report (within firmware/serial jitter) the same pattern
of intervals. So:

  - "Relay jitter" here = how much the host-observed inter-tap
    intervals disagree with the device-observed inter-tap intervals.
    Large values mean the serial/threading path is adding timing
    noise on top of the actual taps.
  - "Accuracy/precision" here = self-consistency of the tap stream
    against its own locally-stable tempo: debounce violations (taps
    firing closer together than the firmware's 150 ms lockout, which
    would indicate a single hit double-triggering) and statistical
    outlier intervals (likely missed taps or accidental extra hits),
    flagged via a MAD-based test that tolerates genuine tempo drift.

Once an audio-derived onset list exists (from a trained detector or a
hand-annotated reference), point --truth at it and this script will
additionally compute real Precision / Recall / F1 / mean-abs-error
against that ground truth — see compare_to_truth() below, wired but
inert until you have that file.

Usage
-----
    python test_tapper_accuracy.py take_001_onsets.json
    python test_tapper_accuracy.py take_001_onsets.json --out report.png
    python test_tapper_accuracy.py take_001_onsets.json --truth take_001_truth.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEBOUNCE_MS = 150.0  # must match the firmware's DEBOUNCE_MS
MAD_OUTLIER_K = 3.5  # standard robust-outlier multiplier on MAD


# ═══════════════════════════════════════════════════════════════════════════
#  LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_take(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    required = {"sample_rate", "host_onset_times", "device_millis"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path} is missing expected keys: {missing}")
    n_host, n_dev = len(data["host_onset_times"]), len(data["device_millis"])
    if n_host != n_dev:
        print(
            f"[warn] host_onset_times ({n_host}) and device_millis ({n_dev}) "
            f"have different lengths — truncating to the shorter of the two."
        )
        n = min(n_host, n_dev)
        data["host_onset_times"] = data["host_onset_times"][:n]
        data["device_millis"] = data["device_millis"][:n]

    # Both logs are appended in arrival order by record_onset_data.py, so
    # they should already be monotonic. A non-monotonic point would mean
    # something upstream (serial glitch, clock issue) reordered events —
    # that's worth surfacing loudly rather than silently feeding a
    # negative interval into the stats below.
    host_t = np.array(data["host_onset_times"], dtype=np.float64)
    if np.any(np.diff(host_t) < 0):
        bad = int(np.sum(np.diff(host_t) < 0))
        print(
            f"[warn] host_onset_times is not monotonically increasing "
            f"({bad} out-of-order point(s)) — this points to a problem "
            f"upstream of this script (serial reordering, clock jump), "
            f"not something interval stats can meaningfully describe. "
            f"Investigate the take before trusting the numbers below."
        )
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════


def compute_metrics(data: dict) -> dict:
    host_t = np.array(data["host_onset_times"], dtype=np.float64)
    dev_ms = np.array(data["device_millis"], dtype=np.float64)
    n_taps = len(host_t)

    metrics: dict = {"n_taps": n_taps}

    if n_taps < 2:
        metrics["warning"] = "fewer than 2 taps — interval stats unavailable"
        return metrics

    host_intervals_ms = np.diff(host_t) * 1000.0
    dev_intervals_ms = np.diff(dev_ms)

    # ── Relay jitter: how much device-observed intervals disagree with
    #    host-observed intervals for the *same* tap-to-tap gaps ──────────
    jitter_ms = host_intervals_ms - dev_intervals_ms
    metrics["jitter_mean_ms"] = float(np.mean(jitter_ms))
    metrics["jitter_std_ms"] = float(np.std(jitter_ms))
    metrics["jitter_max_abs_ms"] = float(np.max(np.abs(jitter_ms)))

    # ── Debounce violations (firmware-side double-trigger risk) ─────────
    # Use the device clock for this check since DEBOUNCE_MS is enforced
    # on-device against device millis(), not host arrival time.
    debounce_violations = int(np.sum(dev_intervals_ms < DEBOUNCE_MS))
    metrics["debounce_violations"] = debounce_violations
    metrics["debounce_threshold_ms"] = DEBOUNCE_MS

    # ── Outlier intervals via MAD (robust to genuine tempo drift) ────────
    median_iti = float(np.median(host_intervals_ms))
    mad = float(np.median(np.abs(host_intervals_ms - median_iti))) or 1e-9
    # consistent scale estimator (Gaussian-equivalent)
    robust_std = 1.4826 * mad
    z = np.abs(host_intervals_ms - median_iti) / max(robust_std, 1e-9)
    outlier_idx = np.where(z > MAD_OUTLIER_K)[0]  # index into intervals array

    metrics["median_iti_ms"] = median_iti
    metrics["implied_bpm_median"] = 60000.0 / median_iti if median_iti > 0 else None
    metrics["outlier_count"] = int(len(outlier_idx))
    metrics["outlier_interval_indices"] = outlier_idx.tolist()
    # Classify each outlier as a likely double-trigger (too short) or
    # likely missed-tap (too long, ~roughly a multiple of the median)
    classifications = []
    for i in outlier_idx:
        iti = host_intervals_ms[i]
        if iti < median_iti:
            classifications.append("short (possible double-trigger)")
        else:
            ratio = iti / median_iti
            nearest_mult = round(ratio)
            if nearest_mult >= 2 and abs(ratio - nearest_mult) < 0.25:
                classifications.append(
                    f"long (~{nearest_mult}x median, possible missed tap)"
                )
            else:
                classifications.append("long (irregular)")
    metrics["outlier_classifications"] = classifications

    metrics["host_intervals_ms"] = host_intervals_ms.tolist()
    metrics["dev_intervals_ms"] = dev_intervals_ms.tolist()
    metrics["host_t"] = host_t.tolist()

    return metrics


def compare_to_truth(data: dict, truth_path: Path, tolerance_ms: float = 50.0) -> dict:
    """
    Real precision/recall/F1/MAE against an external onset-truth file,
    once one exists (e.g. exported from a trained detector or hand
    annotation). Expected format: JSON list of onset times in seconds,
    or {"onset_times": [...]}.

    Matching follows the standard MIREX-style greedy nearest-neighbor
    approach: each detected tap is matched to the closest unmatched
    truth onset within +/- tolerance_ms; unmatched taps are false
    positives, unmatched truth onsets are false negatives.
    """
    with open(truth_path) as f:
        truth_raw = json.load(f)
    truth_times = truth_raw["onset_times"] if isinstance(truth_raw, dict) else truth_raw
    truth_times = np.array(sorted(truth_times), dtype=np.float64)
    detected = np.array(sorted(data["host_onset_times"]), dtype=np.float64)

    tol = tolerance_ms / 1000.0
    matched_truth = np.zeros(len(truth_times), dtype=bool)
    errors = []
    tp = 0

    for dt in detected:
        candidates = np.where(~matched_truth)[0]
        if len(candidates) == 0:
            break
        diffs = np.abs(truth_times[candidates] - dt)
        j = np.argmin(diffs)
        if diffs[j] <= tol:
            matched_truth[candidates[j]] = True
            errors.append(truth_times[candidates[j]] - dt)
            tp += 1

    fp = len(detected) - tp
    fn = len(truth_times) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_error_ms": float(np.mean(np.abs(errors)) * 1000) if errors else None,
        "tolerance_ms": tolerance_ms,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT (console)
# ═══════════════════════════════════════════════════════════════════════════


def print_report(metrics: dict, truth_metrics: dict | None):
    print("\n" + "=" * 60)
    print("  TAPPER DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"  Taps recorded:           {metrics['n_taps']}")

    if "warning" in metrics:
        print(f"  [warn] {metrics['warning']}")
        print("=" * 60 + "\n")
        return

    print(
        f"  Median inter-tap (ITI):  {metrics['median_iti_ms']:.1f} ms "
        f"(~{metrics['implied_bpm_median']:.1f} BPM)"
    )
    print("-" * 60)
    print("  Relay jitter (host vs. device interval agreement)")
    print(f"    mean:                  {metrics['jitter_mean_ms']:+.2f} ms")
    print(f"    std:                   {metrics['jitter_std_ms']:.2f} ms")
    print(f"    max |jitter|:          {metrics['jitter_max_abs_ms']:.2f} ms")
    print("-" * 60)
    print(
        f"  Debounce violations (< {metrics['debounce_threshold_ms']:.0f} ms, device clock): "
        f"{metrics['debounce_violations']}"
    )
    print("-" * 60)
    print(f"  Outlier intervals (MAD, k={MAD_OUTLIER_K}): {metrics['outlier_count']}")
    for idx, cls in zip(
        metrics["outlier_interval_indices"], metrics["outlier_classifications"]
    ):
        iti = metrics["host_intervals_ms"][idx]
        print(f"    taps #{idx + 1}->#{idx + 2}: {iti:.1f} ms — {cls}")
    print("=" * 60)

    if truth_metrics is not None:
        print("  GROUND-TRUTH COMPARISON")
        print(f"    tolerance:             ±{truth_metrics['tolerance_ms']:.0f} ms")
        print(
            f"    TP / FP / FN:          {truth_metrics['tp']} / {truth_metrics['fp']} / {truth_metrics['fn']}"
        )
        print(f"    precision:             {truth_metrics['precision'] * 100:.1f}%")
        print(f"    recall:                {truth_metrics['recall'] * 100:.1f}%")
        print(f"    F1:                    {truth_metrics['f1'] * 100:.1f}%")
        if truth_metrics["mean_abs_error_ms"] is not None:
            print(
                f"    mean abs timing error: {truth_metrics['mean_abs_error_ms']:.1f} ms"
            )
        print("=" * 60)
    else:
        print("  No --truth file supplied — precision/recall against an")
        print("  independent reference are not available for this take.")
        print("=" * 60)
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════


def make_figure(metrics: dict, out_path: Path | None):
    import matplotlib

    if out_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "warning" in metrics:
        print(f"[skip plot] {metrics['warning']}")
        return

    host_t = np.array(metrics["host_t"])
    host_iti = np.array(metrics["host_intervals_ms"])
    dev_iti = np.array(metrics["dev_intervals_ms"])
    jitter = host_iti - dev_iti
    outlier_idx = metrics["outlier_interval_indices"]
    median_iti = metrics["median_iti_ms"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Foot Tapper Diagnostic", fontsize=14, fontweight="bold")

    # Panel 1 — tap timeline with outliers marked
    ax = axes[0, 0]
    ax.scatter(
        host_t, np.zeros_like(host_t), s=40, c="steelblue", zorder=3, label="tap"
    )
    for idx in outlier_idx:
        # outlier interval i is between tap i and tap i+1
        ax.scatter(
            host_t[idx + 1],
            0,
            s=90,
            facecolors="none",
            edgecolors="crimson",
            linewidths=2,
            zorder=4,
        )
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title("Tap timeline (red ring = flagged interval)")
    ax.set_ylim(-1, 1)

    # Panel 2 — ITI in implied BPM over the take, with median band
    ax = axes[0, 1]
    bpm = 60000.0 / host_iti
    mid_t = (host_t[:-1] + host_t[1:]) / 2.0
    median_bpm = 60000.0 / median_iti
    ax.plot(mid_t, bpm, "o-", color="darkorange", markersize=4, linewidth=1)
    ax.axhline(
        median_bpm,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"median {median_bpm:.1f} BPM",
    )
    if len(outlier_idx):
        ax.scatter(
            mid_t[outlier_idx],
            bpm[outlier_idx],
            s=90,
            facecolors="none",
            edgecolors="crimson",
            linewidths=2,
            zorder=4,
            label="outlier",
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("implied BPM")
    ax.set_title("Tempo stability (inter-tap interval)")
    ax.legend(fontsize=8)

    # A single missed-tap (very long gap) or double-trigger (very short
    # gap) can swing implied BPM by 5-10x, which would compress the
    # genuinely-interesting jitter in the rest of the take into a flat
    # line. Clip the visible range to a robust window around the median
    # and annotate any point that falls outside it with its real value,
    # rather than letting one outlier dictate the whole y-scale.
    in_range_mask = (bpm >= median_bpm * 0.5) & (bpm <= median_bpm * 1.5)
    if np.any(in_range_mask):
        visible_bpm = bpm[in_range_mask]
        pad = max(5.0, (visible_bpm.max() - visible_bpm.min()) * 0.3)
        y_lo = max(0.0, visible_bpm.min() - pad)
        y_hi = visible_bpm.max() + pad
    else:
        y_lo, y_hi = median_bpm * 0.5, median_bpm * 1.5
    off_scale = np.where(~in_range_mask)[0]
    for i in off_scale:
        y_clip = y_hi * 0.97 if bpm[i] > y_hi else y_lo * 1.03
        ax.annotate(
            f"{bpm[i]:.0f}",
            xy=(mid_t[i], y_clip),
            xytext=(0, 8 if bpm[i] > y_hi else -12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="crimson",
        )
    ax.set_ylim(y_lo, y_hi)

    # Panel 3 — host vs device interval agreement
    ax = axes[1, 0]
    lims = [0, max(host_iti.max(), dev_iti.max()) * 1.05]
    ax.plot(
        lims, lims, color="gray", linestyle="--", linewidth=1, label="perfect agreement"
    )
    ax.scatter(dev_iti, host_iti, s=30, c="seagreen", alpha=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("device-clock interval (ms)")
    ax.set_ylabel("host-clock interval (ms)")
    ax.set_title("Host vs. device interval agreement")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    # Panel 4 — jitter histogram
    ax = axes[1, 1]
    ax.hist(
        jitter,
        bins=max(5, min(30, len(jitter) // 2)),
        color="slateblue",
        edgecolor="white",
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("jitter = host interval − device interval (ms)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Relay jitter distribution (μ={metrics['jitter_mean_ms']:+.1f}, "
        f"σ={metrics['jitter_std_ms']:.1f} ms)"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure -> {out_path}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description="Diagnose foot-tapper timing consistency from a record_onset_data.py take"
    )
    ap.add_argument("onsets_json", type=Path, help="path to <out>_onsets.json")
    ap.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="optional ground-truth onset times JSON for real precision/recall",
    )
    ap.add_argument(
        "--tolerance-ms",
        type=float,
        default=50.0,
        help="matching tolerance when --truth is supplied (default 50ms)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="save the figure to this path instead of opening a window "
        "(e.g. report.png) — also writes a metrics_<out>.json next to it",
    )
    args = ap.parse_args()

    if not args.onsets_json.exists():
        print(f"[error] file not found: {args.onsets_json}", file=sys.stderr)
        sys.exit(1)

    data = load_take(args.onsets_json)
    metrics = compute_metrics(data)

    truth_metrics = None
    if args.truth is not None:
        if not args.truth.exists():
            print(f"[error] truth file not found: {args.truth}", file=sys.stderr)
            sys.exit(1)
        truth_metrics = compare_to_truth(
            data, args.truth, tolerance_ms=args.tolerance_ms
        )

    print_report(metrics, truth_metrics)
    make_figure(metrics, args.out)

    if args.out is not None:
        summary_path = args.out.with_name(f"metrics_{args.out.stem}.json")
        export = {
            k: v
            for k, v in metrics.items()
            if k not in ("host_intervals_ms", "dev_intervals_ms", "host_t")
        }
        if truth_metrics is not None:
            export["truth_comparison"] = truth_metrics
        with open(summary_path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"Saved metrics -> {summary_path}")


if __name__ == "__main__":
    main()
