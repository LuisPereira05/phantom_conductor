"""
Phantom Conductor — Shared Audio Buffers
=========================================
Module-level singletons so audio_input, audio_analysis, and
audio_processing all share the exact same objects without
circular imports or argument passing.

Import pattern:
    from buffers import audio_buffer, audio_queue
"""

from collections import deque
from queue import Queue

# Sample rate assumed everywhere
SR         = 44100
BUFFER_SEC = 10

# Raw microphone samples (float32 mono).
# Written by audio_input; read by audio_analysis.
audio_buffer: deque = deque(maxlen=int(SR * BUFFER_SEC))

# Beat-aligned blocks ready for playback (float32 arrays).
# Written by audio_processing; drained by audio_playback.
audio_queue: Queue = Queue(maxsize=80)
