# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:58:48 2026

@author: Adam
"""

# strategy/rescue.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from solvers.newton_pointwise import (
    PointwiseNewtonOptions,
    PointwiseNewtonSolver,
)
from solvers.result import SolverResult


@dataclass
class RescueOptions:
    """
    Options controlling hybrid rescue behaviour.
    """

    enabled: bool = True
    pointwise_tol: float = 1e-1
    pointwise_max_iter: int = 50
    pointwise_relaxation: float = 1.0
    pointwise_freq_out: int = 1


@dataclass
class RescueAttemptResult:
    """
    Structured output from a rescue attempt.
    """

    attempted: bool
    rescue_result: SolverResult | None
    retry_result: SolverResult | None

    @property
    def succeeded(self) -> bool:
        return self.retry_result is not None and self.retry_result.converged


class RescuePolicy:
    """
    Encapsulates the fallback pattern:

        best_x
          -> pointwise rescue
          -> retry final phase

    The final retry phase is injected as a callable so this class stays independent
    of whether the final solver is trust-region Newton, sparse Newton, or something else.
    """

    def __init__(
        self,
        pointwise_solver: PointwiseNewtonSolver,
    ):
        self.pointwise_solver = pointwise_solver

    def attempt(
        self,
        best_x: np.ndarray,
        final_phase_runner: Callable[[np.ndarray], SolverResult],
        options: RescueOptions | None = None,
    ) -> RescueAttemptResult:
        if options is None:
            options = RescueOptions()

        if not options.enabled:
            return RescueAttemptResult(
                attempted=False,
                rescue_result=None,
                retry_result=None,
            )

        print("[Rescue] starting pointwise rescue phase...")

        rescue_result = self.pointwise_solver.solve(
            np.array(best_x, copy=True),
            PointwiseNewtonOptions(
                tol=options.pointwise_tol,
                max_iter=options.pointwise_max_iter,
                relaxation=options.pointwise_relaxation,
                freq_out=options.pointwise_freq_out,
            ),
        )

        print(
            "[Rescue] pointwise rescue finished: "
            f"converged={rescue_result.converged}, "
            f"||F||_inf={rescue_result.residual_inf_norm:.6e}"
        )

        print("[Rescue] retrying final phase...")
        retry_result = final_phase_runner(rescue_result.x)

        print(
            "[Rescue] retry finished: "
            f"converged={retry_result.converged}, "
            f"||F||_inf={retry_result.residual_inf_norm:.6e}"
        )

        return RescueAttemptResult(
            attempted=True,
            rescue_result=rescue_result,
            retry_result=retry_result,
        )