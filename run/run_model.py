# run/run_model.py

from __future__ import annotations

from core.problem import compile_problem
from strategy.hybrid import HybridSolverStrategy, HybridOptions
from run.output import (
    build_filename_stem,
    build_output_paths,
    get_output_config,
    handle_figure,
    maybe_print_output_path,
    should_save_point_outputs,
    save_final_field,
    save_point_summary,
)


def run_single(
    model,
    config: dict,
    params: dict,
    initial_field=None,
    point_index: int | None = None,
    output_paths: dict | None = None,
):
    """
    Compile and solve one parameter point.

    Returns
    -------
    field : ndarray
        Solution reshaped to the model field layout.
    result : SolverResult
        Full structured solver result.
    problem : CompiledProblem
        Compiled problem used for the solve.
    observable_val : float | None
        Computed observable if available, else None/NaN.
    """
    out_cfg = get_output_config(config)
    if output_paths is None:
        output_paths = build_output_paths(config)

    problem = compile_problem(model, config, params)
    x0, seed_label = _build_initial_field(problem, initial_field)

    strategy = HybridSolverStrategy(problem)
    options = HybridOptions(**config["solver"])
    result = strategy.solve(x0, options)

    field = problem.reshape_solution(result.x)
    observable_val = _compute_observable_if_available(model, problem, field)

    if out_cfg["print_point_summary"]:
        _print_run_summary(
            result=result,
            seed_label=seed_label,
        )

    if output_paths is not None:
        summary_path = save_point_summary(
            config=config,
            point_index=point_index,
            params=params,
            result=result,
            observable_val=observable_val,
            seed_label=seed_label,
            paths=output_paths,
        )
        maybe_print_output_path(config, "point summary", summary_path)

        field_path = save_final_field(
            config=config,
            point_index=point_index,
            field=field,
            result=result,
            paths=output_paths,
        )
        maybe_print_output_path(config, "final field", field_path)

    _handle_solution_plot(
        model=model,
        config=config,
        params=params,
        field=field,
        point_index=point_index,
        output_paths=output_paths,
        converged=bool(result.converged),
    )

    return field, result, problem, observable_val


def _build_initial_field(problem, initial_field):
    if initial_field is None:
        print("[RunSingle] seed source: analytic initial guess")
        return problem.build_initial_guess(), "analytic"

    print("[RunSingle] seed source: continuation guess")
    return problem.build_continuation_guess(initial_field), "continuation"


def _compute_observable_if_available(model, problem, field):
    if hasattr(model, "compute_observable"):
        return float(problem.compute_observable(field))
    return None


def _handle_solution_plot(
    *,
    model,
    config: dict,
    params: dict,
    field,
    point_index: int | None,
    output_paths: dict | None,
    converged: bool,
) -> None:
    out_cfg = get_output_config(config)

    want_show = bool(out_cfg["show_solution_plots"])
    want_save = bool(out_cfg["save_solution_plots"]) and should_save_point_outputs(config, converged)

    if not (want_show or want_save):
        return

    fig = None

    if hasattr(model, "make_solution_plot"):
        fig = model.make_solution_plot(field, config, params)

    if fig is None:
        return

    save_path = None
    if want_save and output_paths is not None:
        stem = build_filename_stem(config, "solution_plot", point_index=point_index)
        save_path = output_paths["plots"] / f"{stem}.{out_cfg['plot_format']}"

    handle_figure(
        fig,
        save=want_save,
        show=want_show,
        save_path=save_path,
        dpi=out_cfg["plot_dpi"],
    )
    maybe_print_output_path(config, "solution plot", save_path)


def _print_run_summary(result, seed_label: str) -> None:
    print(
        f"[RunSingle] seed={seed_label}  "
        f"converged={result.converged}  "
        f"||F||_inf={result.residual_inf_norm:.6e}  "
        f"iterations={result.iterations}"
    )
    print(f"[RunSingle] message: {result.message}")

    metadata = result.metadata or {}

    phases = metadata.get("phases")
    if phases:
        print(f"[RunSingle] phases: {' -> '.join(phases)}")

    timings = metadata.get("timings")
    if isinstance(timings, dict) and timings:
        total = timings.get("total")
        if total is not None:
            print(f"[RunSingle] total wall time: {total:.3f} s")

        phase_timings = [
            f"{name}={dt:.3f}s"
            for name, dt in timings.items()
            if name != "total"
        ]
        if phase_timings:
            print(f"[RunSingle] phase timings: {', '.join(phase_timings)}")

    best_residual = metadata.get("best_residual")
    if best_residual is not None:
        print(f"[RunSingle] best retained ||F||_inf: {best_residual:.6e}")

    final_solver = metadata.get("final_solver")
    final_message = metadata.get("final_message")

    if final_solver is not None:
        print(f"[RunSingle] final solver: {final_solver}")
    if final_message is not None:
        print(f"[RunSingle] final phase message: {final_message}")

    if metadata.get("accepted_near_solution", False):
        print("[RunSingle] accepted best near-solution iterate")
