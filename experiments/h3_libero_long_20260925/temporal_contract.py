"""CPU-only causal indexing for the proposed H3 -> LIBERO transfer.

An observation at HDF5 row i is AFTER action i. This implementation must only
be used after the input manifest has been audited as post_action.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalWindow:
    anchor: int
    boundary: int
    lower_observations: tuple[int, ...]
    upper_observations: tuple[int, ...]
    upper_actions: tuple[int | None, ...]
    upper_action_mask: int
    future_actions: tuple[int, ...]
    upper_teacher: int
    remaining_condition: float = 1.0


def causal_window(length: int, anchor: int, *, observation_timing: str) -> TemporalWindow:
    if observation_timing != "post_action":
        raise ValueError("require an audited post_action manifest")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (length, anchor)):
        raise ValueError("integer length and anchor required")
    if length < 9 or anchor < 0 or anchor % 8 or anchor + 8 >= length:
        raise ValueError("need a chunk-aligned anchor and eight real future actions")
    boundary = (anchor // 16) * 16
    return TemporalWindow(
        anchor=anchor,
        boundary=boundary,
        lower_observations=tuple(max(0, i) for i in range(anchor - 7, anchor + 1)),
        upper_observations=tuple(max(0, i) for i in range(boundary - 15, boundary + 1)),
        upper_actions=(tuple(range(boundary - 15, boundary + 1))
                       if boundary else (None,) * 16),
        upper_action_mask=int(boundary > 0),
        future_actions=tuple(range(anchor + 1, anchor + 9)),
        upper_teacher=min(boundary + 16, length - 1),
    )


def valid_anchors(length: int) -> range:
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("positive integer episode length required")
    return range(0, max(0, length - 8), 8)


class Clock:
    """Pure control clock: no teacher, episode length, or future observation."""

    def __init__(self) -> None:
        self.step = 0
        self.last_boundary: int | None = None

    def needs_upper(self) -> bool:
        return self.last_boundary is None or self.step - self.last_boundary >= 16

    def record_upper(self) -> None:
        if not self.needs_upper() or self.step % 16:
            raise ValueError("upper state may only be refreshed at a clock boundary")
        self.last_boundary = self.step

    def record_chunk(self, executed_actions: int = 8) -> None:
        if self.last_boundary is None or self.needs_upper():
            raise ValueError("must compute the current upper state before action")
        if isinstance(executed_actions, bool) or executed_actions != 8:
            raise ValueError("only complete chunks advance this training/evaluation clock")
        self.step += executed_actions
