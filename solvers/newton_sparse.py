# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:39:04 2026

@author: Adam Thomasson
"""

# solvers/newton_sparse.py

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

from core.operators import ProblemOperators
from core.problem import CompiledProblem
from .result import SolverResult


@dataclass
class SparseNewtonOptions:
    """
    Options for the global sparse Newton solver.
    """
    tol: float = 1e-6
    max_iter: int = 100
    freq_out: int = 1
    step_scale: float = 1.0


class SparseNewtonSolver:
    """
    Global sparse Newton solver.

    This solver assembles the full residual vector and full sparse Jacobian,
    solves the Newton system

        J(x_k) delta_k = -F(x_k),

    and updates

        x_{k+1} = x_k + alpha * delta_k

    where alpha = step_scale.

    This is the fast solver for near-solution convergence, but it is less robust
    than the pointwise solver on difficult initial guesses or badly conditioned
    branches.
    """

    def __init__(self, problem: CompiledProblem, ops: ProblemOperators | None = None):
        self.problem = problem
        if ops is not None and ops.problem is not problem:
            raise ValueError("Operators must belong to the supplied problem.")
        self.ops = ops if ops is not None else ProblemOperators(problem)

    def solve(
        self,
        x0,
        options: SparseNewtonOptions | None = None,
    ) -> SolverResult:
        if options is None:
            options = SparseNewtonOptions()

        x = self.ops.to_real(x0)

        residual_history: list[float] = []
        t_start = time.time()

        converged = False
        message = "Maximum iterations reached."
        pending_step = False

        for iteration in range(options.max_iter):
            # ----------------------------------------------------------
            # Residual
            # ----------------------------------------------------------
            F = self.ops.residual(x)
            res_inf = float(np.max(np.abs(F)))
            residual_history.append(res_inf)
            pending_step = False

            if iteration % max(1, options.freq_out) == 0:
                print(
                    f"[NewtonSparse] iter={iteration:03d}  "
                    f"||F||_inf={res_inf:.6e}"
                )

            if res_inf <= options.tol:
                converged = True
                message = "Converged to requested tolerance."
                break

            # ----------------------------------------------------------
            # Jacobian and Newton step
            # ----------------------------------------------------------
            J = self.ops.jacobian_csr(x)

            try:
                delta = spsolve(J, -F)
            except Exception as exc:
                message = f"Sparse linear solve failed: {exc}"
                break

            if not np.all(np.isfinite(delta)):
                message = "Sparse linear solve produced non-finite Newton step."
                break

            x_trial = x + options.step_scale * np.asarray(delta, dtype=float)
            if not np.all(np.isfinite(x_trial)):
                message = "State became non-finite after Newton update."
                break
            x = x_trial
            pending_step = True

        # The final allowed update must be checked before it can be reported as
        # a solution; otherwise x and residual_inf_norm refer to different states.
        if pending_step or not residual_history:
            final_residual = float(np.max(np.abs(self.ops.residual(x))))
            residual_history.append(final_residual)
            if final_residual <= options.tol:
                converged = True
                message = "Converged to requested tolerance."
        else:
            final_residual = residual_history[-1]
        elapsed = time.time() - t_start

        return SolverResult(
            x=np.array(x, copy=True),
            converged=converged,
            iterations=max(0, len(residual_history) - 1),
            residual_inf_norm=float(final_residual),
            solver_name="newton_sparse",
            message=message,
            residual_history=residual_history,
            metadata={
                "elapsed_sec": float(elapsed),
                "n_unknowns": int(self.problem.n_unknowns),
                "step_scale": float(options.step_scale),
                "jacobian_nnz": int(self.ops.structure.nnz),
            },
        )
