"""Headless subprocess adapter; original runners and solvers stay unchanged."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np


PREFIX = "[KINK_UI] "


def emit(event, **data):
    print(PREFIX + json.dumps({"event": event, **data}, allow_nan=False), flush=True)


def load_model(path):
    name = "_kink_ui_model"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load this Python model.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def render_model_plot(problem, field, job_dir, index):
    """Render the handwritten model's Figure inside the worker process."""
    if problem.plot_solution_fn is None:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    figure = None
    existing_figures = set(plt.get_fignums())
    try:
        with plt.rc_context():
            figure = problem.plot_solution(field)
            if figure is None:
                return None
            if not isinstance(figure, Figure):
                raise TypeError("make_solution_plot() must return a Matplotlib Figure or None")
            path = job_dir / f"point_{index:04d}_model_plot.png"
            dpi = max(200, int(problem.config.get("output", {}).get("plot_dpi", 160)))
            figure.savefig(path, dpi=dpi)
        return str(path)
    except Exception as exc:
        print(f"[UI plot] Could not render make_solution_plot(): {exc}. "
              "Displaying raw field profiles instead.", flush=True)
        return None
    finally:
        if isinstance(figure, Figure):
            plt.close(figure)
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)


def run_request(request):
    from core.parameter_scan import build_parameter_points
    from run.output import build_output_paths
    from run.run_model import run_single
    import run.run_scan as scan_runner
    model = load_model(request["model_path"])
    config = model.get_model_config()
    config["output"] = {**config.get("output", {}), "show_solution_plots": False, "show_scan_plot": False}
    if request.get("solver_override") is not None:
        config["solver"] = request["solver_override"]
        config["jacobian_autodiff"] = request.get("jacobian_autodiff", "auto")
    # Keep every UI run in its own directory; CLI output behaviour is untouched.
    run_name = config["output"].get("run_name", "model")
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    config["output"]["run_name"] = f"{run_name}_{suffix}"
    paths = build_output_paths(config)
    job_dir = Path(request["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(request["model_path"], paths["root"] / "model.py")
    (paths["root"] / "ui_run_config.json").write_text(json.dumps(config, indent=2, default=str))
    model.get_model_config = lambda: config
    finalize = getattr(model, "finalize_parameters", None)
    points = build_parameter_points(config, finalize)
    emit("started", output_dir=str(paths["root"].resolve()), points=len(points),
         field_names=config.get("field_names", [f"field {a}" for a in range(config["n_fields"])]))
    completed = []

    def report_single(**kwargs):
        k = kwargs.get("point_index")
        k = 0 if k is None else int(k)
        emit("point_started", index=k, total=len(points))
        field, result, problem, obs = run_single(**kwargs)
        path = job_dir / f"point_{k:04d}.npz"
        np.savez_compressed(path, field=field, coordinate=problem.grid.build_coordinates()[0],
                            history=np.asarray(result.residual_history, dtype=float))
        finite_obs = float(obs) if obs is not None and np.isfinite(obs) else None
        status = "converged" if result.converged else "near solution" if result.metadata.get("accepted_near_solution") else "failed"
        item = dict(index=k, params={name: float(value) for name, value in problem.params.items()},
                    converged=bool(result.converged), status=status, residual=float(result.residual_inf_norm),
                    iterations=int(result.iterations), observable=finite_obs, message=result.message,
                    data_path=str(path), geometry=problem.geometry, spacing=problem.grid.grid_spacing[0])
        if not np.isfinite(item["residual"]):
            item["residual"] = None
        if request.get("use_model_plot", False):
            model_plot_path = render_model_plot(problem, field, job_dir, k)
            if model_plot_path is not None:
                item["model_plot_path"] = model_plot_path
        completed.append(item)
        emit("point_result", **item)
        return field, result, problem, obs

    original = scan_runner.run_single
    try:
        if config.get("variation_mode", "none") == "none":
            report_single(model=model, config=config, params=points[0], output_paths=paths)
        else:
            scan_runner.run_single = report_single
            scan_runner.run_scan(model)
    finally:
        scan_runner.run_single = original
    summary = {"points": completed, "output_dir": str(paths["root"].resolve())}
    (job_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit("finished", converged=sum(p["converged"] for p in completed), total=len(completed))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    try:
        run_request(json.loads(args.request.read_text(encoding="utf-8")))
        return 0
    except Exception as exc:
        traceback.print_exc()
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
