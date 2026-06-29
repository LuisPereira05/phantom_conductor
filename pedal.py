from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from config import CFG
from logger import Logger

N_BUTTONS = 3

DEFAULT_HOLD_THRESHOLD_S = 0.35
DEFAULT_STUCK_TIMEOUT_S = 5.0


# ── EVENT NAMING ────────────────────────────────────────────────────────────


def event_name(button_ids: frozenset[int], kind: str) -> str:
    """('TAP' | 'HOLD', {2, 1}) -> 'B1+B2_TAP'"""
    parts = "+".join(f"B{b}" for b in sorted(button_ids))
    return f"{parts}_{kind}"


# Backwards-compatible alias for internal call sites in this module.
_event_name = event_name


# ── PER-BUTTON RECORD ───────────────────────────────────────────────────────


@dataclass
class _ButtonWindow:
    button_id: int
    press_time: float
    release_time: float | None = None  # None while still held down

    def overlaps(self, other: "_ButtonWindow", now: float) -> bool:
        """
        Two windows overlap if their [press, release-or-now] intervals
        intersect. Using `now` for an still-open (not yet released)
        window lets a not-yet-finished press still register as
        overlapping with an already-finished one.
        """
        a_end = self.release_time if self.release_time is not None else now
        b_end = other.release_time if other.release_time is not None else now
        return self.press_time <= b_end and other.press_time <= a_end


# ── CONTROLLER ──────────────────────────────────────────────────────────────


class PedalController:
    """
    Feed it (button_id, is_press) pairs via handle_event(); it calls
    back into on_resolved_event(event_name) whenever a tap/hold/combo
    fully resolves. Thread-safe — handle_event() is expected to be
    called from the same serial-reading thread that already drives
    tempo_tapper.py, but the lock makes it safe regardless.
    """

    def __init__(self, on_resolved_event, logger: Logger | None = None):
        self._lock = threading.Lock()
        self._logger = logger
        self._on_resolved_event = on_resolved_event

        # Buttons currently down, or released-but-still-waiting-on-an-
        # overlapping-sibling-to-release. Keyed by button_id.
        self._pending: dict[int, _ButtonWindow] = {}

        # Live "is this button physically down right now" state, purely
        # for UI display (PEDAL SETUP window light-up indicators).
        self._live_down: set[int] = set()

    # ── live state for UI ──────────────────────────────────────────

    def live_down_buttons(self) -> set[int]:
        with self._lock:
            return set(self._live_down)

    # ── event ingestion ─────────────────────────────────────────────

    def handle_event(self, button_id: int, is_press: bool) -> None:
        now = time.time()
        with self._lock:
            if is_press:
                self._live_down.add(button_id)
                if button_id in self._pending:
                    # A press line for a button already mid-window (e.g.
                    # a duplicate or a missed release) — restart its
                    # window rather than compounding stale state.
                    if self._logger:
                        self._logger.warn(
                            f"pedal: B{button_id} pressed again while already "
                            f"pending — resetting its window"
                        )
                self._pending[button_id] = _ButtonWindow(
                    button_id=button_id, press_time=now
                )
            else:
                self._live_down.discard(button_id)
                win = self._pending.get(button_id)
                if win is None or win.release_time is not None:
                    # Release with no matching open press — ignore.
                    if self._logger:
                        self._logger.warn(
                            f"pedal: B{button_id} release with no open press — ignored"
                        )
                    return
                win.release_time = now

            self._try_resolve(now)
            self._flush_stuck(now)

    # ── resolution ───────────────────────────────────────────────────

    def _connected_group(self, seed_id: int, now: float) -> set[int]:
        """
        Transitive closure of overlapping windows starting from
        seed_id, over everything currently in self._pending.
        """
        group = {seed_id}
        changed = True
        while changed:
            changed = False
            for bid, win in self._pending.items():
                if bid in group:
                    continue
                seed_win = self._pending[seed_id]
                if any(self._pending[g].overlaps(win, now) for g in group):
                    group.add(bid)
                    changed = True
        return group

    def _try_resolve(self, now: float) -> None:
        """
        Look for any connected group where every member has released,
        and resolve it (emit or drop) if so. Repeats until no more
        groups are resolvable in this pass — resolving one group can't
        affect another since groups are disjoint by definition, but
        the pending dict mutates as we go so we re-scan defensively.
        """
        while True:
            resolved_any = False
            for bid in list(self._pending.keys()):
                if bid not in self._pending:
                    continue  # already swept into a group resolved this pass
                group_ids = self._connected_group(bid, now)
                windows = [self._pending[g] for g in group_ids]
                if any(w.release_time is None for w in windows):
                    continue  # someone in the group is still held down
                self._resolve_group(group_ids, windows)
                resolved_any = True
            if not resolved_any:
                break

    def _resolve_group(
        self, group_ids: set[int], windows: list["_ButtonWindow"]
    ) -> None:
        threshold = CFG.get("pedal_hold_threshold_s", DEFAULT_HOLD_THRESHOLD_S)
        kinds = {
            "HOLD" if (w.release_time - w.press_time) >= threshold else "TAP"
            for w in windows
        }

        for g in group_ids:
            self._pending.pop(g, None)

        if len(kinds) > 1:
            # Disagreement (some tapped, some held) — strict mode drops it.
            if self._logger:
                durs = ", ".join(
                    f"B{w.button_id}={w.release_time - w.press_time:.2f}s"
                    for w in windows
                )
                self._logger.debug(
                    f"pedal: group {sorted(group_ids)} disagreed on tap/hold "
                    f"({durs}) — dropped"
                )
            return

        kind = kinds.pop()
        event_name = _event_name(frozenset(group_ids), kind)
        if self._logger:
            self._logger.info(f"pedal: {event_name}")
        self._on_resolved_event(event_name)

    def _flush_stuck(self, now: float) -> None:
        """
        Safety valve: if a button is pressed but its release line never
        arrives (loose wire, dropped serial byte, etc.), every sibling
        that overlaps it — including ones that already released and
        are just waiting for it — would otherwise block forever. Any
        button that has been continuously DOWN (no release_time) for
        longer than the timeout is considered abandoned.

        Subtlety: an abandoned button's window is open-ended (it never
        got a release), so a naive overlap test against "now" would
        make it overlap ANY other open window forever, including one
        that happens to start in this very handle_event() call. That
        would incorrectly drag a brand-new press into a years-old stuck
        button's group the instant it arrives. To avoid that, once a
        button is identified as stuck we only pull in OTHER pending
        buttons whose press_time is no later than
        (stuck_button.press_time + timeout) — i.e. buttons that were
        already part of the same physical gesture *before* the stuck
        button was declared abandoned — never something that shows up
        afterwards.
        """
        timeout = CFG.get("pedal_stuck_timeout_s", DEFAULT_STUCK_TIMEOUT_S)

        stuck_open_ids = {
            bid
            for bid, win in self._pending.items()
            if win.release_time is None and (now - win.press_time) >= timeout
        }
        if not stuck_open_ids:
            return

        visited: set[int] = set()
        for seed in stuck_open_ids:
            if seed in visited:
                continue

            seed_win = self._pending[seed]
            cutoff = seed_win.press_time + timeout

            # Only consider buttons that existed (were pressed) before
            # this stuck button crossed its own abandonment threshold —
            # this is what keeps a press arriving "now" (long after
            # cutoff) from being swept in.
            eligible_ids = {
                bid for bid, win in self._pending.items() if win.press_time <= cutoff
            }

            group_ids = self._connected_group(seed, now) & eligible_ids
            visited |= group_ids

            still_down_ids = {
                g for g in group_ids if self._pending[g].release_time is None
            }
            released_ids = group_ids - still_down_ids

            if still_down_ids and self._logger:
                self._logger.warn(
                    f"pedal: stuck-window timeout — B"
                    f"{sorted(still_down_ids)} never released, dropping "
                    f"from group {sorted(group_ids)}"
                )

            for g in still_down_ids:
                self._pending.pop(g, None)
                self._live_down.discard(g)

            if released_ids:
                windows = [self._pending[g] for g in released_ids]
                self._resolve_group(released_ids, windows)
            else:
                for g in group_ids:
                    self._pending.pop(g, None)

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._live_down.clear()


# ── CONFIG-BACKED COMMAND MAPPING ───────────────────────────────────────────

ALL_EVENT_NAMES: list[str] = []
for _n in range(1, N_BUTTONS + 1):
    from itertools import combinations

    for _combo in combinations(range(1, N_BUTTONS + 1), _n):
        for _kind in ("TAP", "HOLD"):
            ALL_EVENT_NAMES.append(_event_name(frozenset(_combo), _kind))

_DEFAULT_PEDAL_MAP: dict[str, str] = {
    "B1_TAP": "toggle",
    "B2_TAP": "next",
    "B3_TAP": "prev",
    "B1_HOLD": "loop_toggle",
    "B2_HOLD": "loop_next",
    "B3_HOLD": "loop_prev",
}


def get_pedal_map() -> dict[str, str]:
    """Current event->command mapping, defaults filled in for any
    event name missing from CFG (e.g. after a fresh install)."""
    saved = CFG.get("pedal_map", {}) or {}
    merged = dict(_DEFAULT_PEDAL_MAP)
    merged.update(saved)
    for name in ALL_EVENT_NAMES:
        merged.setdefault(name, "none")
    return merged


def set_pedal_command(event_name: str, command: str) -> None:
    pedal_map = get_pedal_map()
    pedal_map[event_name] = command
    CFG.set("pedal_map", pedal_map)


# ── WIRING INTO STATE / DISPATCH ─────────────────────────────────────────────


def make_pedal_dispatch(state, logger: Logger):
    """
    Returns a PedalController already wired to call
    state.dispatch_command() using the live CFG.pedal_map — exactly the
    same dispatch surface gesture_recognition.py and tempo_tapper.py's
    sibling (tempo, not pedal) use. CFG is re-read on every event so
    remapping in the PEDAL SETUP window takes effect immediately, no
    restart required (same pattern as CFG.use_tempo_tapper).
    """

    def _on_event(event_name: str):
        if not CFG.get("use_pedal", False):
            logger.info(f"pedal: {event_name} fired but pedal disabled — ignored")
            return
        command = get_pedal_map().get(event_name, "none")
        if command and command != "none":
            state.dispatch_command(command, logger)
        else:
            logger.info(f"pedal: {event_name} mapped to 'none' — skipped")

    return PedalController(on_resolved_event=_on_event, logger=logger)
