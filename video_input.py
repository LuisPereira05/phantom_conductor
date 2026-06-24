"""
Phantom Conductor — Video Input
================================
Owns the OpenCV VideoCapture and the MediaPipe HandLandmarker session.

Public API
----------
open_camera(cam_idx)  → cv2.VideoCapture   (raises RuntimeError on failure)
read_frame(cap)       → (frame_bgr, mp_image) or (None, None) on EOF/error
make_landmarker()     → HandLandmarker context manager
detect(landmarker, mp_image) → HandLandmarkerResult

This module does NOT draw anything and does NOT classify gestures.
All drawing lives in gesture_recognition.py.
"""

import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

from logger import Logger

import sys
cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)  # or CAP_DSHOW on Windows
print("opened:", cap.isOpened())
ret, frame = cap.read()
print("read:", ret, "shape:", frame.shape if ret else None)
cap.release()

# ── MediaPipe model ────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

FRAME_W = 640   # match what your camera actually supports
FRAME_H = 480


def download_model(logger: Logger):
    """Download the MediaPipe hand landmarker model if not already present."""
    if os.path.exists(MODEL_PATH):
        return
    logger.info("Downloading MediaPipe model (~8 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.ok("Model downloaded successfully.")
    except Exception as e:
        logger.err(f"Failed to download model: {e}")
        raise SystemExit(1)


def open_camera(cam_idx: int, logger: Logger) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_idx}")
    
    # Request resolution and log what we actually got
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.ok(f"Camera {cam_idx} open: {actual_w}x{actual_h}")  # ASCII x, not *
    return cap


def read_frame(cap: cv2.VideoCapture):
    """
    Read one frame from the capture.

    Returns
    -------
    (frame_bgr_flipped, mp_image)  on success
    (None, None)                   on read failure
    """
    ret, frame = cap.read()
    if not ret:
        return None, None
    frame  = cv2.flip(frame, 1)
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    return frame, mp_img


def make_landmarker() -> HandLandmarker:
    """
    Create and return a HandLandmarker configured for single-image mode.
    Use as a context manager:

        with make_landmarker() as detector:
            result = detector.detect(mp_image)
    """
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.55,
    )
    return HandLandmarker.create_from_options(options)
