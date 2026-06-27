from collections import deque
from queue import Queue

# Sample rate
SR = 44100
BUFFER_SEC = 10

# Entrada cruda del micrófono (float32 mono)
# Escrito por audio_input.py, consumido por audio_analysis.py
audio_buffer: deque = deque(maxlen=int(SR * BUFFER_SEC))

# Bloques alineados
audio_queue: Queue = Queue(maxsize=80)
