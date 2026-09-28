# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:12:36 2026

@author: Adam Thomasson
"""

# strategy/hybrid.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.operators import ProblemOperators
from core.problem import CompiledProblem
from diagnostics.history import SolveHistory
from diagnostics.timers import SolveTimers
from solvers.newton_pointwise import (
    PointwiseNewtonOptions,
    PointwiseNewtonSolver,
)
from solvers.newton_sparse import (
    SparseNewtonOptions,
    SparseNewtonSolver,
)
from solvers.newton_trust_region import (
    TrustRegionNewtonSolver,
    TrustRegionOptions,
)
from solvers.result import SolverResult
from strategy.rescue import RescueOptions, RescuePolicy


@dataclass
class HybridOptions:
    """
    Orchestration options for the hybrid solve strategy.
    """

    # --------------------------------------------------------------
    # Phase selection
    # --------------------------------------------------------------
    use_pointwise_phase: bool = True
    use_trust_region: bool = True

    # --------------------------------------------------------------
    # Pointwise phase
    # --------------------------------------------------------------
    pointwise_tol: float = 1e-1
    pointwise_max_iter: int = 100
    pointwise_relaxation: float = 1.0
    pointwise_freq_out: int = 1

    # --------------------------------------------------------------
    # Final solve phase
    # --------------------------------------------------------------
    final_tol: float = 1e-6
    final_max_iter: int = 100
    final_freq_out: int = 1
    final_step_scale: float = 1.0

    # --------------------------------------------------------------
    # Trust-region parameters
    # --------------------------------------------------------------
    trust_initial_radius: float = 1.0
    trust_min_radius: float = 1e-6
    trust_max_radius: float = 50.0
    trust_shrink: float = 0.75
    trust_grow: float = 1.5
    trust_max_trials: int = 10
    trust_accept_growth: float = 50.0
    trust_good_improve: float = 0.75

    # --------------------------------------------------------------
    # Rescue policy
    # --------------------------------------------------------------
    enable_rescue: bool = True
    rescue_pointwise_max_iter: int = 50
    rescue_pointwise_relaxation: float = 1.0

    # --------------------------------------------------------------
    # Near-solution acceptance
    # --------------------------------------------------------------
    accept_best_near_tol: bool = False
    near_tol_factor: float = 10.0


class HybridSolverStrategy:
    """
    Hybrid solve orchestration.

    Workflow:
        1. Optional robust pointwise pre-phase
        2. Final global phase:
             - trust-region sparse Newton, or
             - plain sparse Newton
        3. Optional rescue:
             - pointwise rescue from best retained phase endpoint
             - retry final phase
        4. Return converged solution or best retained phase endpoint
    """

    def __init__(self, problem: CompiledProblem):
        self.problem = problem
        self.ops = ProblemOperators(problem)

        self.pointwise_solver = PointwiseNewtonSolver(problem, ops=self.ops)
        self.sparse_solver = SparseNewtonSolver(problem, ops=self.ops)
        self.trust_solver = TrustRegionNewtonSolver(problem, ops=self.ops)
        self.rescue_policy = RescuePolicy(self.pointwise_solver)

    def solve(
        self,
        x0,
        options: HybridOptions | None = None,
    ) -> SolverResult:
        if options is None:
            options = HybridOptions()

        x_current = self.ops.to_real(x0)

        history = SolveHistory()
        timers = SolveTimers()

        initial_residual = self._residual_inf(x_current)
        history.seed_best(x_current, initial_residual)

        print(f"[Hybrid] initial ||F||_inf={initial_residual:.6e}")

        # ----------------------------------------------------------
        # Phase 1: robust pointwise preconditioning
        # ----------------------------------------------------------
        if options.use_pointwise_phase:
            print("[Hybrid] starting pointwise pre-phase...")

            with timers.phase("pointwise_pre"):
                pointwise_result = self.pointwise_solver.solve(
                    x_current,
                    PointwiseNewtonOptions(
                        tol=options.pointwise_tol,
                        max_iter=options.pointwise_max_iter,
                        relaxation=options.pointwise_relaxation,
                        freq_out=options.pointwise_freq_out,
                    ),
                )

            x_current = pointwise_result.x
            history.append_result("pointwise_pre", pointwise_result)

            print(
                f"[Hybrid] after pointwise pre-phase: "
                f"converged={pointwise_result.converged}, "
                f"||F||_inf={pointwise_result.residual_inf_norm:.6e}"
            )

        # ----------------------------------------------------------
        # Phase 2: final global solve
        # ----------------------------------------------------------
        with timers.phase("final_phase"):
            final_result = self._run_final_phase(x_current, options)

        x_current = final_result.x
        history.append_result(final_result.solver_name, final_result)

        # ----------------------------------------------------------
        # Success case
        # ----------------------------------------------------------
        if final_result.converged:
            return self._build_success_result(
                result=final_result,
                history=history,
                timers=timers,
                message="Hybrid solve converged.",
            )

        # ----------------------------------------------------------
        # Phase 3: rescue policy
        # ----------------------------------------------------------
        if options.enable_rescue and history.best_x is not None:
            rescue_options = RescueOptions(
                enabled=options.enable_rescue,
                pointwise_tol=options.pointwise_tol,
                pointwise_max_iter=options.rescue_pointwise_max_iter,
                pointwise_relaxation=options.rescue_pointwise_relaxation,
                pointwise_freq_out=options.pointwise_freq_out,
            )

            with timers.phase("rescue_total"):
                rescue_attempt = self.rescue_policy.attempt(
                    best_x=history.best_x,
                    final_phase_runner=lambda x: self._run_final_phase(x, options),
                    options=rescue_options,
                )

            if rescue_attempt.attempted and rescue_attempt.rescue_result is not None:
                history.append_result("pointwise_rescue", rescue_attempt.rescue_result)

            if rescue_attempt.attempted and rescue_attempt.retry_result is not None:
                history.append_result(
                    rescue_attempt.retry_result.solver_name + "_retry",
                    rescue_attempt.retry_result,
                )

            if rescue_attempt.succeeded:
                return self._build_success_result(
                    result=rescue_attempt.retry_result,
                    history=history,
                    timers=timers,
                    message="Hybrid solve converged after rescue.",
                )

        # ----------------------------------------------------------
        # Near-solution acceptance
        # ----------------------------------------------------------
        if (
            options.accept_best_near_tol
            and history.best_residual <= options.near_tol_factor * options.final_tol
            and history.best_x is not None
        ):
            metadata = self._build_metadata(history, timers)
            metadata["accepted_near_solution"] = True

            return SolverResult(
                x=np.array(history.best_x, copy=True),
                converged=False,
                iterations=sum(phase.iterations for phase in history.phases),
                residual_inf_norm=float(history.best_residual),
                solver_name="hybrid",
                message="Returned best near-solution iterate.",
                residual_history=list(history.residual_history),
                metadata=metadata,
            )

        metadata = self._build_metadata(history, timers)
        metadata["accepted_near_solution"] = False

        return SolverResult(
            x=np.array(history.best_x, copy=True),
            converged=False,
            iterations=sum(phase.iterations for phase in history.phases),
            residual_inf_norm=float(history.best_residual),
            solver_name="hybrid",
            message="Hybrid solve did not converge; returning best retained phase endpoint.",
            residual_history=list(history.residual_history),
            metadata=metadata,
        )

    def _run_final_phase(
        self,
        x,
        options: HybridOptions,
    ) -> SolverResult:
        """
        Run the final global phase using either trust-region Newton or plain
        sparse Newton.
        """
        if options.use_trust_region:
            print("[Hybrid] starting trust-region final phase...")
            return self.trust_solver.solve(
                x,
                TrustRegionOptions(
                    tol=options.final_tol,
                    max_iter=options.final_max_iter,
                    freq_out=options.final_freq_out,
                    initial_radius=options.trust_initial_radius,
                    min_radius=options.trust_min_radius,
                    max_radius=options.trust_max_radius,
                    shrink=options.trust_shrink,
                    grow=options.trust_grow,
                    max_trials=options.trust_max_trials,
                    accept_growth=options.trust_accept_growth,
                    good_improve=options.trust_good_improve,
                ),
            )

        print("[Hybrid] starting plain sparse Newton final phase...")
        return self.sparse_solver.solve(
            x,
            SparseNewtonOptions(
                tol=options.final_tol,
                max_iter=options.final_max_iter,
                freq_out=options.final_freq_out,
                step_scale=options.final_step_scale,
            ),
        )

    def _build_success_result(
        self,
        result: SolverResult,
        history: SolveHistory,
        timers: SolveTimers,
        message: str,
    ) -> SolverResult:
        metadata = self._build_metadata(history, timers)
        metadata["final_solver"] = result.solver_name
        metadata["final_message"] = result.message

        return SolverResult(
            x=np.array(result.x, copy=True),
            converged=True,
            iterations=sum(phase.iterations for phase in history.phases),
            residual_inf_norm=float(result.residual_inf_norm),
            solver_name="hybrid",
            message=message,
            residual_history=list(history.residual_history),
            metadata=metadata,
        )

    def _build_metadata(
        self,
        history: SolveHistory,
        timers: SolveTimers,
    ) -> dict:
        metadata = history.to_metadata()
        metadata["timings"] = timers.to_metadata()
        if history.phases:
            metadata["final_solver"] = history.phases[-1].name
            metadata["final_message"] = history.phases[-1].message
        return metadata

    def _residual_inf(self, x) -> float:
        F = self.ops.residual(x)
        return float(np.max(np.abs(F)))
