# run/run_scan.py

from __future__ import annotations

import numpy as np

from .run_model import run_single
from .output import (
    build_filename_stem,
    build_output_paths,
    get_output_config,
    handle_figure,
    make_default_scan_plot,
    maybe_print_output_path,
    save_observables_table,
    save_scan_summary,
)
from core.parameter_scan import build_parameter_points
from strategy.continuation import ContinuationManager, ContinuationOptions


def run_scan(model):
    """
    Run all parameter points defined by the model.

    Returns
    -------
    param_points : list[dict]
        Expanded parameter dictionaries used in the scan.
    observable_vals : ndarray
        Observable evaluated at each scan point.
    results : list[SolverResult]
        Structured solver results for each point.
    """
    config = model.get_model_config()
    out_cfg = get_output_config(config)
    output_paths = build_output_paths(config)

    finalize_parameters = getattr(model, "finalize_parameters", None)
    param_points = build_parameter_points(config, finalize_parameters)

    n_points = len(param_points)
    observable_vals = np.full(n_points, np.nan, dtype=float)
    results = []
    point_summaries = []

    continuation = ContinuationManager(
        ContinuationOptions(
            enabled=True,
            require_converged_previous=True,
            copy_on_store=True,
        )
    )

    for k, params in enumerate(param_points):
        print(f"\nPoint {k + 1}/{n_points}")

        if out_cfg["print_params_each_point"]:
            print("params =", params)

        initial_field = continuation.previous_seed()

        field, result, problem, observable_val = run_single(
            model=model,
            config=config,
            params=params,
            initial_field=initial_field,
            point_index=k,
            output_paths=output_paths,
        )

        # A finite energy from a failed solve is not a physical scan point.
        observable_vals[k] = (
            float(observable_val)
            if result.converged and observable_val is not None
            else np.nan
        )

        continuation.update(
            field=np.asarray(field).reshape(-1),
            converged=bool(result.converged),
        )

        results.append(result)
        point_summaries.append(
            _build_point_summary(
                point_index=k,
                params=params,
                result=result,
                observable_val=observable_vals[k],
            )
        )

    _handle_scan_plot(
        model=model,
        config=config,
        param_points=param_points,
        observable_vals=observable_vals,
        output_paths=output_paths,
    )

    scan_summary_path = save_scan_summary(
        config=config,
        point_summaries=point_summaries,
        paths=output_paths,
    )
    maybe_print_output_path(config, "scan summary", scan_summary_path)

    observables_path = save_observables_table(
        config=config,
        param_points=param_points,
        observable_vals=observable_vals,
        paths=output_paths,
    )
    maybe_print_output_path(config, "observables", observables_path)

    if out_cfg["print_scan_summary"]:
        _print_scan_summary(
            point_summaries=point_summaries,
            observable_vals=observable_vals,
        )

    return param_points, observable_vals, results


def _build_point_summary(
    point_index: int,
    params: dict,
    result,
    observable_val: float,
) -> dict:
    metadata = result.metadata or {}
    timings = metadata.get("timings", {}) if isinstance(metadata, dict) else {}

    return {
        "point_index": int(point_index),
        "params": dict(params),
        "converged": bool(result.converged),
        "residual_inf_norm": float(result.residual_inf_norm),
        "iterations": int(result.iterations),
        "observable": float(observable_val) if np.isfinite(observable_val) else np.nan,
        "phases": list(metadata.get("phases", [])),
        "best_residual": float(metadata.get("best_residual", result.residual_inf_norm)),
        "final_solver": metadata.get("final_solver"),
        "accepted_near_solution": bool(metadata.get("accepted_near_solution", False)),
        "total_time": float(timings.get("total", np.nan)),
    }


def _handle_scan_plot(
    *,
    model,
    config: dict,
    param_points: list[dict],
    observable_vals,
    output_paths: dict,
) -> None:
    out_cfg = get_output_config(config)

    want_show = bool(out_cfg["show_scan_plot"])
    want_save = bool(out_cfg["save_scan_plot"])

    if not (want_show or want_save):
        return

    if hasattr(model, "make_scan_plot"):
        fig = model.make_scan_plot(param_points, observable_vals, config)
    else:
        fig = make_default_scan_plot(config, param_points, observable_vals)

    save_path = None
    if want_save:
        stem = build_filename_stem(config, "scan_plot")
        save_path = output_paths["scan"] / f"{stem}.{out_cfg['plot_format']}"

    handle_figure(
        fig,
        save=want_save,
        show=want_show,
        save_path=save_path,
        dpi=out_cfg["plot_dpi"],
    )
    maybe_print_output_path(config, "scan plot", save_path)


def _print_scan_summary(point_summaries: list[dict], observable_vals: np.ndarray) -> None:
    n_points = len(point_summaries)
    n_converged = sum(1 for s in point_summaries if s["converged"])
    n_failed = n_points - n_converged
    n_near = sum(1 for s in point_summaries if s["accepted_near_solution"])

    residuals = np.array([s["residual_inf_norm"] for s in point_summaries], dtype=float)
    best_residuals = np.array([s["best_residual"] for s in point_summaries], dtype=float)
    times = np.array([s["total_time"] for s in point_summaries], dtype=float)

    print("\n" + "=" * 72)
    print("[RunScan] scan summary")
    print("=" * 72)
    print(f"[RunScan] points                : {n_points}")
    print(f"[RunScan] converged             : {n_converged}")
    print(f"[RunScan] failed                : {n_failed}")
    print(f"[RunScan] near-solution accepts : {n_near}")

    if residuals.size > 0:
        print(f"[RunScan] best final residual   : {np.nanmin(residuals):.6e}")
        print(f"[RunScan] worst final residual  : {np.nanmax(residuals):.6e}")

    if best_residuals.size > 0:
        print(f"[RunScan] best retained residual: {np.nanmin(best_residuals):.6e}")
        print(f"[RunScan] worst best-residual   : {np.nanmax(best_residuals):.6e}")

    finite_times = times[np.isfinite(times)]
    if finite_times.size > 0:
        print(f"[RunScan] total wall time       : {np.sum(finite_times):.3f} s")
        print(f"[RunScan] mean wall time/point  : {np.mean(finite_times):.3f} s")

    finite_obs = observable_vals[np.isfinite(observable_vals)]
    if finite_obs.size > 0:
        print(f"[RunScan] observable min        : {np.min(finite_obs):.6e}")
        print(f"[RunScan] observable max        : {np.max(finite_obs):.6e}")

    failed_points = [s for s in point_summaries if not s["converged"]]
    if failed_points:
        print("-" * 72)
        print("[RunScan] failed points:")
        for s in failed_points:
            print(
                f"  point={s['point_index'] + 1:03d}  "
                f"||F||_inf={s['residual_inf_norm']:.6e}  "
                f"best={s['best_residual']:.6e}  "
                f"solver={s['final_solver']}  "
                f"phases={' -> '.join(s['phases']) if s['phases'] else 'n/a'}"
            )

    print("=" * 72)
