# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:31:41 2026

@author: Adam Thomasson
"""

# solvers/result.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class SolverResult:
    """
    Standard solver return object.

    This is intentionally lightweight. It gives every solver a common return
    format without imposing heavy inheritance or framework structure.
    """
    x: np.ndarray
    converged: bool
    iterations: int
    residual_inf_norm: float

    solver_name: str = ""
    message: str = ""

    residual_history: list[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def best_residual(self) -> float:
        if self.residual_history:
            return float(min(self.residual_history))
        return float(self.residual_inf_norm)

    def copy(self) -> "SolverResult":
        return SolverResult(
            x=np.array(self.x, copy=True),
            converged=bool(self.converged),
            iterations=int(self.iterations),
            residual_inf_norm=float(self.residual_inf_norm),
            solver_name=str(self.solver_name),
            message=str(self.message),
            residual_history=list(self.residual_history),
            metadata=dict(self.metadata),
        )