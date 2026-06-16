"""
Phantom Conductor — Gesture Trainer
=====================================
Records hand-landmark samples for custom gestures, persists templates to
gesture_templates.json, and exposes a nearest-neighbour matcher used by
gesture_recognition.py as a first-pass classifier (overrides built-ins).

Feature vector (22 floats, all dimensionless ratios / angles)
--------------------------------------------------------------
Rather than raw or globally-normalised XYZ coordinates (which overfit
to the exact capture position), we extract geometric relationships that
are invariant to hand size, distance from camera, and moderate rotation.

  EXTENSION RATIOS  (5 values)
    For each finger: distance(tip → MCP) / palm_scale
    High = extended, low = curled.  palm_scale = distance(wrist → middle MCP).

  CURL ANGLES  (5 values)
    Angle at the PIP joint: vectors (MCP→PIP) and (PIP→TIP).
    0 rad = fully extended, π rad = fully curled.

  SPREAD ANGLES  (4 values)
    Angle between adjacent finger base-vectors from the wrist
    (wrist→index_MCP, wrist→middle_MCP, …).
    Captures finger splay / grouping.

  THUMB OPPOSITION  (1 value)
    distance(thumb_tip → index_MCP) / palm_scale.
    Low = thumb opposing index (pinch-like), high = thumb abducted.

  PALM NORMAL  (3 values)
    Unit normal of the palm plane, expressed in camera space.
    Computed as cross(wrist→index_MCP, wrist→pinky_MCP), normalised.
    Distinguishes palm-up / palm-down / palm-facing-camera etc.

  FINGERTIP PAIRWISE DISTANCES  (4 values)
    distance(index_tip → middle_tip) / palm_scale
    distance(middle_tip → ring_tip)  / palm_scale
    distance(ring_tip   → pinky_tip) / palm_scale
    distance(thumb_tip  → index_tip) / palm_scale
    Captures spread and pinch between adjacent fingertips.

Total: 5 + 5 + 4 + 1 + 3 + 4 = 22 floats.

All values are in [0, ~2] (ratios) or [-1, 1] (normal) or [0, π] (angles),
so no whitening is needed for nearest-neighbour comparison.

Matching
--------
L2 distance on the 22-float feature vector.  Threshold is tighter than the
old raw-coordinate version because the feature space is more compact.

Public API
----------
  trainer = GestureTrainer()
  trainer.start_recording("MY_GESTURE")
  trainer.capture_sample(lm_array)       # lm_array: (21,3) float32
  trainer.finish(command="loop_next")
  trainer.delete(name)
  trainer.match(lm_array) -> (name, dist) | (None, inf)
  trainer.list_gestures() -> [{"name":…, "command":…, "samples":N}, …]
"""

import json
import math
import os
import threading
import numpy as np

TEMPLATE_PATH  = os.path.join(os.path.dirname(__file__), "gesture_templates.json")
SAMPLES_NEEDED = 100
MATCH_THRESHOLD = 0.66   # L2 on 22-float feature vec; tune if needed

# ── Landmark indices ───────────────────────────────────────────────────────────
WRIST       = 0
THUMB_MCP, THUMB_IP,  THUMB_TIP  =  2,  3,  4
INDEX_MCP,  INDEX_PIP,  INDEX_TIP  =  5,  6,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP =  9, 10, 12
RING_MCP,   RING_PIP,   RING_TIP   = 13, 14, 16
PINKY_MCP,  PINKY_PIP,  PINKY_TIP  = 17, 18, 20


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (radians) between two 3-D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.acos(cos))


def extract_features(lm: np.ndarray) -> np.ndarray:
    """
    Convert a (21, 3) landmark array into a 22-float feature vector.
    The input can be in any coordinate system / scale — all outputs are
    dimensionless ratios or bounded angles.
    """
    # Palm scale: wrist → middle MCP distance
    palm_scale = float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]))
    if palm_scale < 1e-9:
        palm_scale = 1.0   # degenerate frame guard

    feats = []

    # ── Extension ratios (5) ─────────────────────────────────────────────────
    for tip, mcp in [
        (THUMB_TIP,  THUMB_MCP),
        (INDEX_TIP,  INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP,   RING_MCP),
        (PINKY_TIP,  PINKY_MCP),
    ]:
        feats.append(np.linalg.norm(lm[tip] - lm[mcp]) / palm_scale)

    # ── Curl angles at PIP (5) ───────────────────────────────────────────────
    for mcp, pip, tip in [
        (THUMB_MCP,  THUMB_IP,  THUMB_TIP),
        (INDEX_MCP,  INDEX_PIP,  INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP,   RING_PIP,   RING_TIP),
        (PINKY_MCP,  PINKY_PIP,  PINKY_TIP),
    ]:
        v1 = lm[pip] - lm[mcp]
        v2 = lm[tip] - lm[pip]
        feats.append(_angle(v1, v2))

    # ── Spread angles between adjacent finger bases (4) ──────────────────────
    bases = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    for i in range(len(bases) - 1):
        v1 = lm[bases[i]]     - lm[WRIST]
        v2 = lm[bases[i + 1]] - lm[WRIST]
        feats.append(_angle(v1, v2))

    # ── Thumb opposition (1) ─────────────────────────────────────────────────
    feats.append(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_MCP]) / palm_scale)

    # ── Palm normal (3) ──────────────────────────────────────────────────────
    v_index = lm[INDEX_MCP] - lm[WRIST]
    v_pinky = lm[PINKY_MCP] - lm[WRIST]
    normal  = np.cross(v_index, v_pinky)
    n_norm  = np.linalg.norm(normal)
    if n_norm > 1e-9:
        normal = normal / n_norm
    feats.extend(normal.tolist())

    # ── Fingertip pairwise distances (4) ─────────────────────────────────────
    tip_pairs = [
        (INDEX_TIP,  MIDDLE_TIP),
        (MIDDLE_TIP, RING_TIP),
        (RING_TIP,   PINKY_TIP),
        (THUMB_TIP,  INDEX_TIP),
    ]
    for a, b in tip_pairs:
        feats.append(np.linalg.norm(lm[a] - lm[b]) / palm_scale)

    return np.array(feats, dtype=np.float32)   # shape (22,)


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTURE TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class GestureTrainer:

    def __init__(self):
        self._lock             = threading.Lock()
        self._gestures: list   = []
        self._recording_name   = None
        self._pending_samples  = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(TEMPLATE_PATH):
            return
        try:
            with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            with self._lock:
                self._gestures = raw
            print(f"[trainer] loaded {len(raw)} custom gesture(s)")
        except Exception as e:
            print(f"[trainer] load failed: {e}")

    def _save(self):
        try:
            with self._lock:
                data = list(self._gestures)
            with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[trainer] save failed: {e}")

    # ── Recording lifecycle ───────────────────────────────────────────────────

    def start_recording(self, name: str):
        name = name.strip().upper()
        if not name:
            raise ValueError("Gesture name cannot be empty")
        with self._lock:
            self._recording_name  = name
            self._pending_samples = []

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording_name is not None

    def recording_name(self) -> str | None:
        with self._lock:
            return self._recording_name

    def sample_count(self) -> int:
        with self._lock:
            return len(self._pending_samples)

    def capture_sample(self, lm: np.ndarray) -> int:
        """
        Extract features from lm (21,3) and store.
        Returns new sample count, or -1 if not recording.
        """
        with self._lock:
            if self._recording_name is None:
                return -1
            vec = extract_features(lm).tolist()
            self._pending_samples.append(vec)
            return len(self._pending_samples)

    def finish(self, command: str = "none") -> str:
        """Average all pending feature vectors into one template and persist."""
        with self._lock:
            name    = self._recording_name
            samples = list(self._pending_samples)
            self._recording_name  = None
            self._pending_samples = []

        if not name or not samples:
            raise RuntimeError("No recording in progress or no samples captured")

        template = np.mean(np.array(samples, dtype=np.float32), axis=0).tolist()
        entry = {
            "name":     name,
            "command":  command,
            "samples":  samples,
            "template": template,
        }

        with self._lock:
            self._gestures = [g for g in self._gestures if g["name"] != name]
            self._gestures.append(entry)

        self._save()
        print(f"[trainer] saved '{name}' ({len(samples)} samples) -> {command}")
        return name

    def cancel_recording(self):
        with self._lock:
            self._recording_name  = None
            self._pending_samples = []

    # ── Management ────────────────────────────────────────────────────────────

    def delete(self, name: str):
        with self._lock:
            self._gestures = [g for g in self._gestures
                              if g["name"] != name.upper()]
        self._save()

    def set_command(self, name: str, command: str):
        with self._lock:
            for g in self._gestures:
                if g["name"] == name.upper():
                    g["command"] = command
                    break
        self._save()

    def list_gestures(self) -> list[dict]:
        with self._lock:
            return [
                {"name": g["name"], "command": g.get("command", "none"),
                 "samples": len(g.get("samples", []))}
                for g in self._gestures
            ]

    # ── Matching ──────────────────────────────────────────────────────────────

    def match(self, lm: np.ndarray) -> tuple[str | None, float]:
        """
        Extract features from lm and compare against all saved templates.
        Returns (name, distance) or (None, inf) if no match within threshold.
        """
        with self._lock:
            gestures = list(self._gestures)

        if not gestures:
            return None, float("inf")

        query     = extract_features(lm)
        best_name = None
        best_dist = float("inf")

        for g in gestures:
            tmpl = np.array(g["template"], dtype=np.float32)
            if tmpl.shape != query.shape:
                # Skip templates saved by older version of the code
                continue
            dist = float(np.linalg.norm(query - tmpl))
            if dist < best_dist:
                best_dist = dist
                best_name = g["name"]

        if best_dist > MATCH_THRESHOLD:
            return None, best_dist

        return best_name, best_dist

    def command_for(self, name: str) -> str:
        with self._lock:
            for g in self._gestures:
                if g["name"] == name:
                    return g.get("command", "none")
        return "none"


# Module-level singleton
TRAINER = GestureTrainer()