# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:05:06 2026

@author: Adam Thomasson
"""

# solvers/trust_region.py

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

from core.operators import ProblemOperators
from core.problem import CompiledProblem
from .result import SolverResult


@dataclass
class TrustRegionOptions:
    """
    Options for the trust-region sparse Newton solver.
    """
    tol: float = 1e-6
    max_iter: int = 100
    freq_out: int = 1

    initial_radius: float = 2.0
    min_radius: float = 1e-6
    max_radius: float = 50.0

    shrink: float = 0.75
    grow: float = 1.5

    max_trials: int = 10

    # Accept step if residual does not worsen "too much"
    accept_growth: float = 50

    # If the step is genuinely good, grow the trust radius
    good_improve: float = 0.75


class TrustRegionNewtonSolver:
    """
    Global sparse Newton solver with trust-region step control.

    This solver computes the full Newton direction from the sparse Jacobian, then
    uses a trust-region style acceptance loop to decide how much of that step to
    accept.

    Compared to plain sparse Newton:
    - more robust
    - slower per successful iteration
    - much better for difficult/stiff systems
    """

    def __init__(self, problem: CompiledProblem, ops: ProblemOperators | None = None):
        self.problem = problem
        if ops is not None and ops.problem is not problem:
            raise ValueError("Operators must belong to the supplied problem.")
        self.ops = ops if ops is not None else ProblemOperators(problem)

    def solve(
        self,
        x0,
        options: TrustRegionOptions | None = None,
    ) -> SolverResult:
        if options is None:
            options = TrustRegionOptions()

        x = self.ops.to_real(x0)

        radius = float(options.initial_radius)
        residual_history: list[float] = []

        converged = False
        message = "Maximum iterations reached."
        pending_step = False

        accepted_steps = 0
        rejected_steps = 0

        t_start = time.time()

        for iteration in range(options.max_iter):
            # ----------------------------------------------------------
            # Evaluate current residual
            # ----------------------------------------------------------
            F = self.ops.residual(x)
            res_inf = float(np.max(np.abs(F)))
            residual_history.append(res_inf)
            pending_step = False

            if iteration % max(1, options.freq_out) == 0:
                print(
                    f"[TrustRegion] iter={iteration:03d}  "
                    f"||F||_inf={res_inf:.6e}  "
                    f"radius={radius:.3e}"
                )

            if res_inf <= options.tol:
                converged = True
                message = "Converged to requested tolerance."
                break

            # ----------------------------------------------------------
            # Compute full Newton direction
            # ----------------------------------------------------------
            J = self.ops.jacobian_csr(x)

            try:
                delta = np.asarray(spsolve(J, -F), dtype=float)
            except Exception as exc:
                message = f"Sparse linear solve failed: {exc}"
                break

            if not np.all(np.isfinite(delta)):
                message = "Sparse linear solve produced non-finite Newton step."
                break

            # ----------------------------------------------------------
            # Restrict by trust radius if necessary
            # ----------------------------------------------------------
            step_norm = float(np.linalg.norm(delta))
            if step_norm > radius and step_norm > 0.0:
                delta = delta * (radius / step_norm)
                step_norm = radius

            accepted = False
            trial_count = 0
            old_res = res_inf

            # ----------------------------------------------------------
            # Trial loop: shrink step if needed until acceptable
            # ----------------------------------------------------------
            for trial in range(options.max_trials):
                trial_count = trial + 1

                x_trial = x + delta

                if not np.all(np.isfinite(x_trial)):
                    rejected_steps += 1
                    delta = delta * options.shrink
                    radius = max(options.min_radius, radius * options.shrink)
                    if radius <= options.min_radius:
                        break
                    continue

                F_trial = self.ops.residual(x_trial)
                trial_res = float(np.max(np.abs(F_trial)))

                # Accept if trial residual is finite and not dramatically worse
                if np.isfinite(trial_res) and trial_res <= options.accept_growth * old_res:
                    x = x_trial
                    accepted = True
                    accepted_steps += 1
                    pending_step = True

                    # Expand radius if the step was genuinely successful
                    if trial_res < options.good_improve * old_res:
                        radius = min(options.max_radius, radius * options.grow)

                    break

                # Otherwise reject and shrink
                rejected_steps += 1
                delta = delta * options.shrink
                radius = max(options.min_radius, radius * options.shrink)

                if radius <= options.min_radius:
                    break

            if not accepted:
                if radius <= options.min_radius:
                    message = "Trust region radius reached minimum without acceptable step."
                else:
                    message = "Trust region failed to find acceptable step."
                break

            if iteration % max(1, options.freq_out) == 0:
                print(
                    f"[TrustRegion] accepted step  "
                    f"trial={trial_count}/{options.max_trials}  "
                    f"step_norm={step_norm:.3e}  "
                    f"new_radius={radius:.3e}"
                )

        # A step accepted on the last iteration has not yet had its residual
        # recorded by the next loop iteration.
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
            solver_name="trust_region_newton",
            message=message,
            residual_history=residual_history,
            metadata={
                "elapsed_sec": float(elapsed),
                "n_unknowns": int(self.problem.n_unknowns),
                "final_radius": float(radius),
                "accepted_steps": int(accepted_steps),
                "rejected_steps": int(rejected_steps),
                "jacobian_nnz": int(self.ops.structure.nnz),
            },
        )
