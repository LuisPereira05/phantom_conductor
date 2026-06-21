"""
gesture_trainer.py  –  rotation-invariant static gesture recognition
======================================================================

Changes from the snapshot version
───────────────────────────────────
• Feature vector (19 floats instead of 22)
    - Kept:  5 extension ratios, 5 PIP curl angles, 4 inter-base spread
             angles, 1 thumb-opposition ratio, 4 fingertip pair distances
    - Dropped: palm normal (3 floats) — it encodes raw hand orientation,
               which is the main reason the old matcher broke under rotation
    - Added:   3 palm-plane/axis angles that are rotation-invariant

• Template storage  →  vector POOL (every raw sample) instead of one average.
  Matching uses cosine-similarity k-NN (k=7) with majority vote.
  This naturally handles pose variation without any statistics.

• Training flow  →  continuous clip recording.
  Call start_clip() / stop_clip() each time the user records a 2-3 s clip;
  capture_frame() is called every camera frame while a clip is live.
  Repeat for N clips from different angles; finish() persists the full pool.

Public API (unchanged for existing callers)
───────────────────────────────────────────
  TRAINER.start_recording(name)     – alias: begins a new gesture session
  TRAINER.start_clip()              – begin one recording clip
  TRAINER.stop_clip()               – end the current clip, bank its frames
  TRAINER.capture_frame(lm)         – feed one (21,3) landmark array
  TRAINER.is_recording()            – True while a session is open
  TRAINER.recording_name()          – current gesture name or None
  TRAINER.clip_count()              – clips banked so far
  TRAINER.frame_count()             – frames banked across all clips
  TRAINER.finish(command)           – save and close the session
  TRAINER.cancel_recording()
  TRAINER.match(lm)                 – (name|None, score 0-1)
  TRAINER.list_gestures()
  TRAINER.delete(name)
  TRAINER.set_command(name, command)
  TRAINER.command_for(name)
"""

import json
import math
import os
import threading

import numpy as np

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "gesture_templates.json")

# Minimum clips before a gesture can be saved (encourages angle variety)
MIN_CLIPS = 3
# Frames per clip are uncapped; the UI should record ~2-3 s @ 30 fps → ~60-90
# k-NN parameters
K_NEIGHBORS = 7  # must be odd
MATCH_THRESHOLD = 0.82  # cosine similarity; tune between 0 and 1

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (19 floats, fully rotation-invariant)
# ═════════════════════════════════════════════════════════════════════════════


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (radians) between two 3-D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.acos(cos))


def _build_local_basis(lm: np.ndarray):
    """
    Build an orthonormal basis anchored to the hand itself.

    e1  =  wrist → middle-MCP   (main palm axis, "up the hand")
    e2  =  component of (wrist → index-MCP) perpendicular to e1  ("across")
    e3  =  e1 × e2              ("out of palm")

    All subsequent feature angles are expressed in this frame, making them
    invariant to global rotation and position.
    """
    e1 = lm[MIDDLE_MCP] - lm[WRIST]
    n1 = np.linalg.norm(e1)
    if n1 < 1e-9:
        return np.eye(3)
    e1 = e1 / n1

    raw2 = lm[INDEX_MCP] - lm[WRIST]
    e2 = raw2 - np.dot(raw2, e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-9:
        # Degenerate: pick any perpendicular
        e2 = np.array([1.0, 0.0, 0.0]) - np.dot([1, 0, 0], e1) * e1
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
    else:
        e2 = e2 / n2

    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=0)  # (3, 3)


def extract_features(lm: np.ndarray) -> np.ndarray:
    """
    Convert a (21, 3) landmark array into a 19-float feature vector.

    All values are dimensionless ratios or bounded angles computed in
    the hand's own coordinate frame — invariant to rotation, translation,
    and scale.

    Layout:
      [0:5]   extension ratios          (tip–MCP distance / palm scale)
      [5:10]  PIP curl angles           (radians, 0 = straight)
      [10:14] inter-base spread angles  (radians)
      [14]    thumb opposition ratio
      [15:19] fingertip pair distances  (normalised)
    """
    basis = _build_local_basis(lm)
    palm_scale = float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]))
    if palm_scale < 1e-9:
        palm_scale = 1.0

    # Project all landmarks into the local frame for angle calculations
    centred = lm - lm[WRIST]
    local = centred @ basis.T  # (21, 3) in local coords

    feats = []

    # ── Extension ratios (5) ─────────────────────────────────────────────────
    for tip, mcp in [
        (THUMB_TIP, THUMB_MCP),
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ]:
        feats.append(np.linalg.norm(local[tip] - local[mcp]) / palm_scale)

    # ── PIP curl angles (5) ──────────────────────────────────────────────────
    # Angle at PIP between (MCP→PIP) and (PIP→TIP) vectors — in local frame
    for mcp, pip, tip in [
        (THUMB_MCP, THUMB_IP, THUMB_TIP),
        (INDEX_MCP, INDEX_PIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP),
    ]:
        v1 = local[pip] - local[mcp]
        v2 = local[tip] - local[pip]
        feats.append(_angle(v1, v2))

    # ── Spread angles between adjacent finger bases (4) ──────────────────────
    bases = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    for i in range(len(bases) - 1):
        v1 = local[bases[i]]
        v2 = local[bases[i + 1]]
        feats.append(_angle(v1, v2))

    # ── Thumb opposition (1) ─────────────────────────────────────────────────
    feats.append(np.linalg.norm(local[THUMB_TIP] - local[INDEX_MCP]) / palm_scale)

    # ── Fingertip pairwise distances (4) ─────────────────────────────────────
    for a, b in [
        (INDEX_TIP, MIDDLE_TIP),
        (MIDDLE_TIP, RING_TIP),
        (RING_TIP, PINKY_TIP),
        (THUMB_TIP, INDEX_TIP),
    ]:
        feats.append(np.linalg.norm(local[a] - local[b]) / palm_scale)

    return np.array(feats, dtype=np.float32)  # shape (19,)


# ═════════════════════════════════════════════════════════════════════════════
#  MATCHING  (cosine k-NN)
# ═════════════════════════════════════════════════════════════════════════════


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _knn_vote(query: np.ndarray, pool: np.ndarray, labels: list[str], k: int):
    """
    Find the k nearest neighbours of query in pool (by cosine similarity).
    Return the majority-vote label and its average similarity score.

    pool   : (N, D) float32
    labels : list of N strings (one per row)
    """
    sims = pool @ query / (np.linalg.norm(pool, axis=1) * np.linalg.norm(query) + 1e-9)
    top_k = np.argsort(sims)[-k:][::-1]
    top_labels = [labels[i] for i in top_k]
    top_sims = sims[top_k]

    # Majority vote (weighted by similarity to break ties gracefully)
    votes: dict[str, float] = {}
    for lbl, sim in zip(top_labels, top_sims):
        votes[lbl] = votes.get(lbl, 0.0) + float(sim)

    winner = max(votes, key=lambda x: votes[x])
    # Average similarity of the winner's neighbours
    winner_sims = [s for l, s in zip(top_labels, top_sims) if l == winner]
    avg_sim = float(np.mean(winner_sims))

    return winner, avg_sim


# ═════════════════════════════════════════════════════════════════════════════
#  GESTURE TRAINER
# ═════════════════════════════════════════════════════════════════════════════


class GestureTrainer:
    """
    Thread-safe trainer.

    Training workflow
    -----------------
    1. start_recording(name)                      – open a session
    2.   start_clip()                             – begin a clip
    3.     capture_frame(lm) × N                  – feed frames (call every cam frame)
    4.   stop_clip()                              – bank the clip
    5.   Repeat 2-4 from different angles
    6. finish(command)                            – save to disk; close session

    At least MIN_CLIPS clips are required before finish() is accepted.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._gestures: list = []

        # Session state
        self._recording_name: str | None = None
        self._banked_vectors: list = []  # all frames from finished clips
        self._banked_clip_count: int = 0
        # Current in-flight clip
        self._clip_active: bool = False
        self._clip_vectors: list = []

        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

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

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start_recording(self, name: str):
        """Open a new training session for gesture `name`."""
        name = name.strip().upper()
        if not name:
            raise ValueError("Gesture name cannot be empty")
        with self._lock:
            self._recording_name = name
            self._banked_vectors = []
            self._banked_clip_count = 0
            self._clip_active = False
            self._clip_vectors = []

    def start_clip(self):
        """Begin recording a new clip.  Must have an open session."""
        with self._lock:
            if self._recording_name is None:
                return
            self._clip_active = True
            self._clip_vectors = []

    def stop_clip(self):
        """
        End the current clip and bank its frames.
        Silently ignored if no clip is active or the clip is empty.
        """
        with self._lock:
            if not self._clip_active or not self._clip_vectors:
                self._clip_active = False
                return
            self._banked_vectors.extend(self._clip_vectors)
            self._banked_clip_count += 1
            self._clip_active = False
            self._clip_vectors = []

    def capture_frame(self, lm: np.ndarray) -> int:
        """
        Extract features from one (21,3) landmark frame and store them
        in the active clip.  Returns the running frame count, or -1 if
        neither a session nor a clip is open.
        """
        with self._lock:
            if self._recording_name is None or not self._clip_active:
                return -1
            vec = extract_features(lm).tolist()
            self._clip_vectors.append(vec)
            return len(self._banked_vectors) + len(self._clip_vectors)

    # Backwards-compat alias used by vision thread
    def capture_sample(self, lm: np.ndarray) -> int:
        return self.capture_frame(lm)

    def finish(self, command: str = "none") -> str:
        """Persist the full feature pool and close the session."""
        with self._lock:
            # Bank any clip that was left open
            if self._clip_active and self._clip_vectors:
                self._banked_vectors.extend(self._clip_vectors)
                self._banked_clip_count += 1
            name = self._recording_name
            vectors = list(self._banked_vectors)
            clip_count = self._banked_clip_count
            # Reset session
            self._recording_name = None
            self._banked_vectors = []
            self._banked_clip_count = 0
            self._clip_active = False
            self._clip_vectors = []

        if not name:
            raise RuntimeError("No recording session in progress")
        if not vectors:
            raise RuntimeError("No frames captured")
        if clip_count < MIN_CLIPS:
            raise RuntimeError(
                f"Only {clip_count} clip(s) recorded; need at least {MIN_CLIPS}"
            )

        entry = {
            "name": name,
            "command": command,
            "clips": clip_count,
            "frames": len(vectors),
            "pool": vectors,  # list of 19-float lists
        }
        with self._lock:
            self._gestures = [g for g in self._gestures if g["name"] != name]
            self._gestures.append(entry)
        self._save()
        print(
            f"[trainer] saved '{name}' — {clip_count} clips, "
            f"{len(vectors)} frames → command '{command}'"
        )
        return name

    def cancel_recording(self):
        with self._lock:
            self._recording_name = None
            self._banked_vectors = []
            self._banked_clip_count = 0
            self._clip_active = False
            self._clip_vectors = []

    # ── Status queries ────────────────────────────────────────────────────────

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording_name is not None

    def is_clip_active(self) -> bool:
        with self._lock:
            return self._clip_active

    def recording_name(self) -> str | None:
        with self._lock:
            return self._recording_name

    def clip_count(self) -> int:
        with self._lock:
            return self._banked_clip_count

    def frame_count(self) -> int:
        with self._lock:
            return len(self._banked_vectors) + (
                len(self._clip_vectors) if self._clip_active else 0
            )

    # Backwards-compat alias
    def sample_count(self) -> int:
        return self.frame_count()

    # ── Management ────────────────────────────────────────────────────────────

    def delete(self, name: str):
        with self._lock:
            self._gestures = [g for g in self._gestures if g["name"] != name.upper()]
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
                {
                    "name": g["name"],
                    "command": g.get("command", "none"),
                    "clips": g.get("clips", 1),
                    "frames": len(g.get("pool", g.get("samples", []))),
                }
                for g in self._gestures
            ]

    # ── Matching ──────────────────────────────────────────────────────────────

    def match(self, lm: np.ndarray) -> tuple[str | None, float]:
        """
        Extract features from lm and run cosine k-NN against every saved pool.

        Returns (gesture_name, similarity_score) or (None, 0.0).
        similarity_score is in [0, 1]; higher is better.
        """
        with self._lock:
            gestures = list(self._gestures)
        if not gestures:
            return None, 0.0

        query = extract_features(lm)

        # Build a combined pool + label list across all gestures
        all_vecs: list[np.ndarray] = []
        all_labels: list[str] = []

        for g in gestures:
            raw = g.get("pool") or g.get("samples")  # compat with old format
            if not raw:
                continue
            arr = np.array(raw, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != len(query):
                # Trained with a different feature vector length — skip
                continue
            all_vecs.append(arr)
            all_labels.extend([g["name"]] * len(arr))

        if not all_vecs:
            return None, 0.0

        pool = np.concatenate(all_vecs, axis=0)
        k = min(K_NEIGHBORS, len(all_labels))
        name, sim = _knn_vote(query, pool, all_labels, k)

        if sim < MATCH_THRESHOLD:
            return None, sim
        return name, sim

    def command_for(self, name: str) -> str:
        with self._lock:
            for g in self._gestures:
                if g["name"] == name:
                    return g.get("command", "none")
        return "none"


# Module-level singleton
TRAINER = GestureTrainer()
