# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:33:46 2026

@author: Adam Thomasson
"""

# solvers/pointwise_newton.py

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from core.autodiff import deriv, row_derivatives, to_real_1d
from core.operators import ProblemOperators
from core.problem import CompiledProblem
from .result import SolverResult


@dataclass
class PointwiseNewtonOptions:
    """
    Options for the pointwise nonlinear Gauss-Seidel / local Newton solver.
    """
    tol: float = 1e-1
    max_iter: int = 1000
    relaxation: float = 1.0
    freq_out: int = 10


def _solve_local_block(J: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Solve the local Newton system:

        J * delta = F

    using a direct solve, with least-squares fallback if J is singular or
    ill-conditioned.
    """
    try:
        return np.linalg.solve(J, F)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(J, F, rcond=None)[0]


class PointwiseNewtonSolver:
    """
    Nonlinear Gauss-Seidel solver with local block Newton updates.

    At each spatial site, solve the coupled n_fields x n_fields local Newton
    system while keeping neighboring sites fixed.

    This is slower and more approximate than the global sparse Newton solve,
    but often more robust for difficult or poorly conditioned problems.
    """

    def __init__(self, problem: CompiledProblem, ops: ProblemOperators | None = None):
        self.problem = problem
        if ops is not None and ops.problem is not problem:
            raise ValueError("Operators must belong to the supplied problem.")
        self.ops = ops if ops is not None else ProblemOperators(problem)

    def solve(
        self,
        x0,
        options: PointwiseNewtonOptions | None = None,
    ) -> SolverResult:
        if options is None:
            options = PointwiseNewtonOptions()

        x = np.asarray(self.ops.to_real(x0), dtype=object).reshape(-1)

        n_fields = self.problem.n_fields
        n_grid = self.problem.n_grid
        idx = self.problem.indexer.field_index
        mode = self.problem.config.get("jacobian_autodiff", "auto")
        if mode not in ("auto", "scalar", "row"):
            raise ValueError("jacobian_autodiff must be 'auto', 'scalar', or 'row'.")
        use_row = mode == "row" or (mode == "auto" and 5 <= n_fields <= 64)
        basis = np.eye(n_fields) if use_row else None

        residual_history: list[float] = []

        t_start = time.time()
        converged = False
        message = "Maximum iterations reached."

        # --------------------------------------------------------------
        # Main nonlinear sweep loop
        # --------------------------------------------------------------
        for iteration in range(options.max_iter):
            # ----------------------------------------------------------
            # One full nonlinear Gauss-Seidel sweep over spatial sites
            # ----------------------------------------------------------
            for site in range(n_grid):
                rows = [idx(a, site) for a in range(n_fields)]

                F_local = np.zeros(n_fields, dtype=float)
                J_local = np.zeros((n_fields, n_fields), dtype=float)

                # Evaluate local residual block
                for a, row in enumerate(rows):
                    func = self.problem.func_list[self.problem.func_index[row]]
                    value = func(x, row)
                    F_local[a] = float(value.real) if hasattr(value, "real") else float(value)

                # Evaluate local Jacobian block with respect to same-site fields
                for a, row in enumerate(rows):
                    func = self.problem.func_list[self.problem.func_index[row]]
                    if use_row:
                        J_local[a, :] = row_derivatives(
                            func, x, rows, row, basis, mode,
                            self.ops._row_compatible,
                        )
                    else:
                        for b, col in enumerate(rows):
                            J_local[a, b] = deriv(func, x, col, row)

                delta = _solve_local_block(J_local, F_local)

                # Newton update: x <- x - w * delta
                for b in range(n_fields):
                    x[idx(b, site)] -= options.relaxation * float(delta[b])

            # ----------------------------------------------------------
            # Global residual check after a full sweep
            # ----------------------------------------------------------
            F = self.ops.residual(x)
            res_inf = float(np.max(np.abs(F)))
            residual_history.append(res_inf)

            if iteration % max(1, options.freq_out) == 0:
                print(
                    f"[Pointwise] iter={iteration:03d}  "
                    f"||F||_inf={res_inf:.6e}"
                )

            if res_inf <= options.tol:
                converged = True
                message = "Converged to requested tolerance."
                break

        elapsed = time.time() - t_start

        if residual_history:
            final_residual = residual_history[-1]
        else:
            final_residual = float(np.max(np.abs(self.ops.residual(x))))
            residual_history.append(final_residual)
            if final_residual <= options.tol:
                converged = True
                message = "Converged to requested tolerance."

        return SolverResult(
            x=to_real_1d(x),
            converged=converged,
            iterations=len(residual_history) if options.max_iter > 0 else 0,
            residual_inf_norm=float(final_residual),
            solver_name="pointwise_newton",
            message=message,
            residual_history=residual_history,
            metadata={
                "elapsed_sec": float(elapsed),
                "n_fields": int(n_fields),
                "n_grid": int(n_grid),
                "relaxation": float(options.relaxation),
            },
        )
