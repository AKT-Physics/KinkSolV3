# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:01:45 2026

@author: Adam
"""

# diagnostics/timers.py

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class SolveTimers:
    """
    Accumulates wall-clock timings for named phases.
    """

    durations: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, dt: float) -> None:
        self.durations[name] = self.durations.get(name, 0.0) + float(dt)

    @contextmanager
    def phase(self, name: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            self.add(name, dt)

    def total(self) -> float:
        return float(sum(self.durations.values()))

    def to_metadata(self) -> dict[str, float]:
        data = dict(self.durations)
        data["total"] = self.total()
        return data