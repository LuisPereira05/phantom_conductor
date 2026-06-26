import collections

import numpy as np


class SpeedController:
    """
    Δt > 0: Musician is BEHIND → SPEED UP
    Δt < 0: Musician is AHEAD → SLOW DOWN
    """

    def __init__(
        self,
        k_p: float = 2.0,
        k_i: float = 0.1,
        k_d: float = 0.5,
        max_speed: float = 1.15,
        min_speed: float = 0.85,
        median_window: int = 8,
        smoothing_alpha: float = 0.15,
    ):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.smoothing_alpha = smoothing_alpha

        self.delta_window = collections.deque(maxlen=median_window)
        self.integral = 0.0
        self.prev_delta = 0.0
        self.prev_speed = 1.0

    def update(self, delta_t_raw: float) -> float:
        # 1. Median filter (spike rejection — same principle as your v0.6)
        self.delta_window.append(delta_t_raw)
        delta_t = float(np.median(self.delta_window))

        # 2. PID control
        p_term = self.k_p * delta_t
        self.integral += delta_t
        self.integral = np.clip(self.integral, -1.0, 1.0)
        i_term = self.k_i * self.integral
        d_term = self.k_d * (delta_t - self.prev_delta)
        self.prev_delta = delta_t

        control = p_term + i_term + d_term
        target_speed = 1.0 + control

        # 3. Exponential smoothing
        smoothed = (
            self.smoothing_alpha * target_speed
            + (1 - self.smoothing_alpha) * self.prev_speed
        )
        self.prev_speed = smoothed

        return float(np.clip(smoothed, self.min_speed, self.max_speed))

    def reset(self):
        self.delta_window.clear()
        self.integral = 0.0
        self.prev_delta = 0.0
        self.prev_speed = 1.0
