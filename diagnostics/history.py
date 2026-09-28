# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:00:32 2026

@author: Adam
"""

# diagnostics/history.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from solvers.result import SolverResult


@dataclass
class PhaseRecord:
    """
    Diagnostic record for a single solve phase.
    """

    name: str
    converged: bool
    residual_inf_norm: float
    iterations: int
    message: str = ""


@dataclass
class SolveHistory:
    """
    Aggregated history for a multi-phase solve.

    Stores:
        - raw residual history
        - phase ordering
        - the best retained iterate among the initial seed and phase endpoints
    """

    residual_history: list[float] = field(default_factory=list)
    phases: list[PhaseRecord] = field(default_factory=list)

    best_residual: float = np.inf
    best_x: np.ndarray | None = None

    def seed_best(self, x: np.ndarray, residual_inf_norm: float) -> None:
        self.best_x = np.array(x, copy=True)
        self.best_residual = float(residual_inf_norm)

        if not self.residual_history:
            self.residual_history.append(float(residual_inf_norm))

    def update_best(self, x: np.ndarray, residual_inf_norm: float) -> bool:
        residual_inf_norm = float(residual_inf_norm)
        if residual_inf_norm < self.best_residual:
            self.best_x = np.array(x, copy=True)
            self.best_residual = residual_inf_norm
            return True
        return False

    def append_residual(self, residual_inf_norm: float) -> None:
        self.residual_history.append(float(residual_inf_norm))

    def append_result(
        self,
        phase_name: str,
        result: SolverResult,
    ) -> None:
        self.phases.append(
            PhaseRecord(
                name=phase_name,
                converged=bool(result.converged),
                residual_inf_norm=float(result.residual_inf_norm),
                iterations=int(result.iterations),
                message=str(result.message),
            )
        )

        self.residual_history.extend(float(v) for v in result.residual_history)
        self.update_best(result.x, result.residual_inf_norm)

    def phase_names(self) -> list[str]:
        return [phase.name for phase in self.phases]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "phases": self.phase_names(),
            "best_residual": float(self.best_residual),
            "phase_records": [
                {
                    "name": phase.name,
                    "converged": phase.converged,
                    "residual_inf_norm": phase.residual_inf_norm,
                    "iterations": phase.iterations,
                    "message": phase.message,
                }
                for phase in self.phases
            ],
        }
