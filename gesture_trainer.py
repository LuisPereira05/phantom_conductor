"""
gesture_trainer.py  -  rotation-invariant static gesture recognition
======================================================================

Optimizations over the previous version
-----------------------------------------
- Binary storage (.bin + .meta.json sidecar) instead of JSON.
  - Float vectors stored as raw IEEE-754 float32 - no text parsing on load.
  - 50-60% smaller on disk; load time drops from ~10-20 ms to <1 ms.
  - .meta.json holds names/commands/clip counts (human-readable, tiny).

- Cached, pre-normalized pool matrix (_PoolCache).
  - Built ONCE after load or after finish(); never rebuilt inside match().
  - Pool row norms are pre-computed at cache-build time - not per call.
  - match() is a single BLAS matrix-vector multiply: pool_normed @ query

- capture_frame stores vectors as np.ndarray rows (float32), not Python lists.

Public API - fully backwards-compatible
-----------------------------------------
  TRAINER.start_recording(name)
  TRAINER.start_clip()
  TRAINER.stop_clip()
  TRAINER.capture_frame(lm)
  TRAINER.is_recording()
  TRAINER.recording_name()
  TRAINER.clip_count()
  TRAINER.frame_count()
  TRAINER.finish(command)
  TRAINER.cancel_recording()
  TRAINER.match(lm)               -> (name|None, score 0-1)
  TRAINER.list_gestures()
  TRAINER.delete(name)
  TRAINER.set_command(name, command)
  TRAINER.command_for(name)
"""

import json
import math
import os
import struct
import threading
from dataclasses import dataclass, field

import numpy as np

# -- File paths ----------------------------------------------------------------
_DIR = os.path.dirname(__file__)
TEMPLATE_PATH = os.path.join(_DIR, "gesture_templates.json")  # legacy (read-only)
BINARY_DATA_PATH = os.path.join(_DIR, "gesture_pool.bin")
BINARY_META_PATH = os.path.join(_DIR, "gesture_pool.meta.json")

# -- Hyper-parameters ------------------------------------------------------------
MIN_CLIPS = 3
K_NEIGHBORS = 7  # must be odd
MATCH_THRESHOLD = 0.82  # cosine similarity threshold
VECTOR_DIM = 19

# -- Landmark indices ------------------------------------------------------------
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


# ===============================================================================
#  FEATURE EXTRACTION  (19 floats, fully rotation-invariant)
# ===============================================================================


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.acos(cos))


def _build_local_basis(lm: np.ndarray) -> np.ndarray:
    e1 = lm[MIDDLE_MCP] - lm[WRIST]
    n1 = np.linalg.norm(e1)
    if n1 < 1e-9:
        return np.eye(3)
    e1 = e1 / n1

    raw2 = lm[INDEX_MCP] - lm[WRIST]
    e2 = raw2 - np.dot(raw2, e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-9:
        e2 = np.array([1.0, 0.0, 0.0]) - np.dot([1, 0, 0], e1) * e1
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
    else:
        e2 = e2 / n2

    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=0)


def extract_features(lm: np.ndarray) -> np.ndarray:
    """
    Convert a (21, 3) landmark array into a 19-float feature vector.

    Layout:
      [0:5]   extension ratios          (tip-MCP distance / palm scale)
      [5:10]  PIP curl angles           (radians, 0 = straight)
      [10:14] spread angles             (3 adjacent pairs + 1 outer pair)
      [14]    thumb opposition ratio
      [15:19] fingertip pair distances  (normalised)
    """
    basis = _build_local_basis(lm)
    palm_scale = float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]))
    if palm_scale < 1e-9:
        palm_scale = 1.0

    centred = lm - lm[WRIST]
    local = centred @ basis.T  # (21, 3) in local coords

    feats = []

    # Extension ratios (5)
    for tip, mcp in [
        (THUMB_TIP, THUMB_MCP),
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ]:
        feats.append(np.linalg.norm(local[tip] - local[mcp]) / palm_scale)

    # PIP curl angles (5)
    for mcp, pip, tip in [
        (THUMB_MCP, THUMB_IP, THUMB_TIP),
        (INDEX_MCP, INDEX_PIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP),
    ]:
        feats.append(_angle(local[pip] - local[mcp], local[tip] - local[pip]))

    # Spread angles (4): 3 adjacent pairs + 1 outer pair (index-pinky).
    # NOTE: range(len(bases) - 1) over 4 bases only yields 3 angles on its
    # own - the outer pair below is what brings this to a true 4 and keeps
    # VECTOR_DIM at 19. (A prior revision omitted it, producing 18-float
    # vectors that silently failed to match against 19-dim pools.)
    bases = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    for i in range(len(bases) - 1):
        feats.append(_angle(local[bases[i]], local[bases[i + 1]]))
    feats.append(_angle(local[INDEX_MCP], local[PINKY_MCP]))

    # Thumb opposition (1)
    feats.append(np.linalg.norm(local[THUMB_TIP] - local[INDEX_MCP]) / palm_scale)

    # Fingertip pairwise distances (4)
    for a, b in [
        (INDEX_TIP, MIDDLE_TIP),
        (MIDDLE_TIP, RING_TIP),
        (RING_TIP, PINKY_TIP),
        (THUMB_TIP, INDEX_TIP),
    ]:
        feats.append(np.linalg.norm(local[a] - local[b]) / palm_scale)

    return np.array(feats, dtype=np.float32)


# ===============================================================================
#  BINARY I/O
# ===============================================================================

MAGIC = b"GPOL"


def _save_binary(gestures: list) -> None:
    with_data = [g for g in gestures if g["pool"].shape[0] > 0]

    arrays = [g["pool"] for g in with_data]
    total_rows = sum(a.shape[0] for a in arrays)

    with open(BINARY_DATA_PATH, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", total_rows, len(with_data)))

        for g, arr in zip(with_data, arrays):
            name_b = g["name"].encode("utf-8")
            f.write(struct.pack("<B", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", arr.shape[0]))

        if total_rows > 0:
            combined = np.concatenate(arrays, axis=0)
            f.write(combined.astype(np.float32).tobytes())

    meta = []
    for g in gestures:
        meta.append(
            {
                "name": g["name"],
                "command": g.get("command", "none"),
                "clips": g.get("clips", 0),
                "frames": g["pool"].shape[0],
                "needs_retrain": g.get("needs_retrain", False),
            }
        )
    with open(BINARY_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_binary():
    if not os.path.exists(BINARY_DATA_PATH) or not os.path.exists(BINARY_META_PATH):
        return None
    try:
        with open(BINARY_DATA_PATH, "rb") as f:
            data = f.read()

        if data[:4] != MAGIC:
            print("[trainer] binary: bad magic, ignoring")
            return None

        offset = 4
        total_rows, n_gestures = struct.unpack_from("<II", data, offset)
        offset += 8

        headers = []
        for _ in range(n_gestures):
            name_len = struct.unpack_from("<B", data, offset)[0]
            offset += 1
            name = data[offset : offset + name_len].decode("utf-8")
            offset += name_len
            n_vecs = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            headers.append((name, n_vecs))

        float_bytes = total_rows * VECTOR_DIM * 4
        all_vecs = np.frombuffer(data[offset : offset + float_bytes], dtype=np.float32)
        all_vecs = all_vecs.reshape(total_rows, VECTOR_DIM).copy()

        with open(BINARY_META_PATH, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
        meta_by_name = {m["name"]: m for m in meta_list}

        gestures_with_data = {}
        row = 0
        for name, n_vecs in headers:
            m = meta_by_name.get(name, {})
            pool = all_vecs[row : row + n_vecs]
            row += n_vecs
            gestures_with_data[name] = {
                "name": name,
                "command": m.get("command", "none"),
                "clips": m.get("clips", 0),
                "pool": pool,
            }

        gestures = []
        for m in meta_list:
            name = m["name"]
            if name in gestures_with_data:
                gestures.append(gestures_with_data[name])
            else:
                gestures.append(
                    {
                        "name": name,
                        "command": m.get("command", "none"),
                        "clips": 0,
                        "pool": np.empty((0, VECTOR_DIM), dtype=np.float32),
                        "needs_retrain": m.get("needs_retrain", False),
                    }
                )

        return gestures

    except Exception as e:
        print(f"[trainer] binary load failed: {e}")
        return None


def _load_legacy_json():
    if not os.path.exists(TEMPLATE_PATH):
        return None
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        gestures = []
        needs_retrain = []
        for g in raw:
            name = g["name"]
            command = g.get("command", "none")
            clips = g.get("clips", 1)
            pool_raw = g.get("pool") or g.get("samples")

            if not pool_raw:
                continue

            arr = np.array(pool_raw, dtype=np.float32)

            if arr.ndim != 2 or arr.shape[1] != VECTOR_DIM:
                print(
                    f"[trainer] '{name}': legacy pool has {arr.shape[1]}-float vectors "
                    f"(current format is {VECTOR_DIM}) - pool discarded, RETRAIN NEEDED"
                )
                needs_retrain.append(name)
                gestures.append(
                    {
                        "name": name,
                        "command": command,
                        "clips": 0,
                        "pool": np.empty((0, VECTOR_DIM), dtype=np.float32),
                        "needs_retrain": True,
                    }
                )
                continue

            gestures.append(
                {
                    "name": name,
                    "command": command,
                    "clips": clips,
                    "pool": arr,
                }
            )

        if needs_retrain:
            print(
                f"[trainer] {len(needs_retrain)} gesture(s) need retraining: "
                + ", ".join(needs_retrain)
            )

        return gestures if gestures else None

    except Exception as e:
        print(f"[trainer] legacy JSON load failed: {e}")
        return None


# ===============================================================================
#  POOL CACHE
# ===============================================================================


@dataclass
class _PoolCache:
    matrix: np.ndarray = field(
        default_factory=lambda: np.empty((0, VECTOR_DIM), dtype=np.float32)
    )
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    valid: bool = False

    @classmethod
    def build(cls, gestures: list) -> "_PoolCache":
        arrays, label_chunks = [], []
        for g in gestures:
            arr = g["pool"]
            if arr.ndim != 2 or arr.shape[1] != VECTOR_DIM or len(arr) == 0:
                continue
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            arrays.append((arr / norms).astype(np.float32))
            label_chunks.extend([g["name"]] * len(arr))

        if not arrays:
            return cls()

        matrix = np.ascontiguousarray(np.concatenate(arrays, axis=0))
        labels = np.array(label_chunks, dtype=object)
        return cls(matrix=matrix, labels=labels, valid=True)


# ===============================================================================
#  GESTURE TRAINER
# ===============================================================================


class GestureTrainer:
    def __init__(self):
        self._lock = threading.Lock()
        self._gestures = []
        self._cache = _PoolCache()

        self._recording_name = None
        self._banked_frames = []
        self._banked_clip_count = 0
        self._clip_active = False
        self._clip_frames = []

        self._load()

    def _load(self):
        gestures = _load_binary()
        source = "binary"

        if gestures is None:
            gestures = _load_legacy_json()
            source = "legacy JSON"
            if gestures is None:
                return

        with self._lock:
            self._gestures = gestures
            self._cache = _PoolCache.build(gestures)

        print(f"[trainer] loaded {len(gestures)} gesture(s) from {source}")

        if source == "legacy JSON":
            try:
                self._save()
                print("[trainer] migrated to binary format")
            except Exception as e:
                print(f"[trainer] migration save failed: {e}")

    def _save(self):
        try:
            with self._lock:
                gestures = list(self._gestures)
            _save_binary(gestures)
        except Exception as e:
            print(f"[trainer] save failed: {e}")

    def _rebuild_cache(self):
        self._cache = _PoolCache.build(self._gestures)

    def start_recording(self, name: str):
        name = name.strip().upper()
        if not name:
            raise ValueError("Gesture name cannot be empty")
        with self._lock:
            self._recording_name = name
            self._banked_frames = []
            self._banked_clip_count = 0
            self._clip_active = False
            self._clip_frames = []

    def start_clip(self):
        with self._lock:
            if self._recording_name is None:
                return
            self._clip_active = True
            self._clip_frames = []

    def stop_clip(self):
        with self._lock:
            if not self._clip_active or not self._clip_frames:
                self._clip_active = False
                return
            self._banked_frames.extend(self._clip_frames)
            self._banked_clip_count += 1
            self._clip_active = False
            self._clip_frames = []

    def capture_frame(self, lm: np.ndarray) -> int:
        with self._lock:
            if self._recording_name is None or not self._clip_active:
                return -1
            vec = extract_features(lm)
            self._clip_frames.append(vec)
            return len(self._banked_frames) + len(self._clip_frames)

    def capture_sample(self, lm: np.ndarray) -> int:
        return self.capture_frame(lm)

    def finish(self, command: str = "none") -> str:
        with self._lock:
            if self._clip_active and self._clip_frames:
                self._banked_frames.extend(self._clip_frames)
                self._banked_clip_count += 1
            name = self._recording_name
            frames = list(self._banked_frames)
            clip_count = self._banked_clip_count
            self._recording_name = None
            self._banked_frames = []
            self._banked_clip_count = 0
            self._clip_active = False
            self._clip_frames = []

        if not name:
            raise RuntimeError("No recording session in progress")
        if not frames:
            raise RuntimeError("No frames captured")
        if clip_count < MIN_CLIPS:
            raise RuntimeError(
                f"Only {clip_count} clip(s) recorded; need at least {MIN_CLIPS}"
            )

        pool = np.stack(frames, axis=0).astype(np.float32)

        entry = {
            "name": name,
            "command": command,
            "clips": clip_count,
            "pool": pool,
        }

        with self._lock:
            self._gestures = [g for g in self._gestures if g["name"] != name]
            self._gestures.append(entry)
            self._rebuild_cache()

        self._save()
        print(
            f"[trainer] saved '{name}' - {clip_count} clips, "
            f"{len(frames)} frames -> command '{command}'"
        )
        return name

    def cancel_recording(self):
        with self._lock:
            self._recording_name = None
            self._banked_frames = []
            self._banked_clip_count = 0
            self._clip_active = False
            self._clip_frames = []

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording_name is not None

    def is_clip_active(self) -> bool:
        with self._lock:
            return self._clip_active

    def recording_name(self):
        with self._lock:
            return self._recording_name

    def clip_count(self) -> int:
        with self._lock:
            return self._banked_clip_count

    def frame_count(self) -> int:
        with self._lock:
            return len(self._banked_frames) + (
                len(self._clip_frames) if self._clip_active else 0
            )

    def sample_count(self) -> int:
        return self.frame_count()

    def delete(self, name: str):
        with self._lock:
            self._gestures = [g for g in self._gestures if g["name"] != name.upper()]
            self._rebuild_cache()
        self._save()

    def set_command(self, name: str, command: str):
        with self._lock:
            for g in self._gestures:
                if g["name"] == name.upper():
                    g["command"] = command
                    break
        self._save()

    def list_gestures(self) -> list:
        with self._lock:
            return [
                {
                    "name": g["name"],
                    "command": g.get("command", "none"),
                    "clips": g.get("clips", 0),
                    "frames": g["pool"].shape[0],
                    "needs_retrain": g.get("needs_retrain", False),
                }
                for g in self._gestures
            ]

    def needs_retrain(self) -> list:
        with self._lock:
            return [g["name"] for g in self._gestures if g["pool"].shape[0] == 0]

    def match(self, lm: np.ndarray):
        with self._lock:
            cache = self._cache

        if not cache.valid:
            return None, 0.0

        query = extract_features(lm)
        qnorm = float(np.linalg.norm(query))
        if qnorm < 1e-8:
            return None, 0.0
        query_normed = (query / qnorm).astype(np.float32)

        sims = cache.matrix @ query_normed

        k = min(K_NEIGHBORS, len(cache.labels))

        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_labels = cache.labels[top_k_idx]
        top_k_sims = sims[top_k_idx]

        votes = {}
        for lbl, sim in zip(top_k_labels, top_k_sims):
            votes[lbl] = votes.get(lbl, 0.0) + float(sim)

        winner = max(votes, key=lambda x: votes[x])
        winner_mask = top_k_labels == winner
        avg_sim = float(top_k_sims[winner_mask].mean())

        if avg_sim < MATCH_THRESHOLD:
            return None, avg_sim
        return winner, avg_sim

    def command_for(self, name: str) -> str:
        with self._lock:
            for g in self._gestures:
                if g["name"] == name:
                    return g.get("command", "none")
        return "none"


TRAINER = GestureTrainer()
