# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:27:50 2026

@author: Adam
"""

# run/output.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_CONFIG = {
    # General output location / naming
    "base_dir": "outputs",
    "run_name": "default_run",
    "file_tag": None,

    # Console verbosity
    "print_point_summary": True,
    "print_scan_summary": True,
    "print_params_each_point": True,
    "print_output_paths": True,

    # Plot control
    "show_solution_plots": True,
    "save_solution_plots": False,
    "show_scan_plot": True,
    "save_scan_plot": False,
    "plot_format": "png",
    "plot_dpi": 200,

    # Data saving
    "save_point_summary": False,
    "save_scan_summary": False,
    "save_observables": False,
    "save_final_field": False,

    # Save style
    "concise": True,
    "save_only_converged": True,
}


def get_output_config(config: dict) -> dict:
    """
    Merge the model output config with defaults.
    """
    user_cfg = dict(config.get("output", {}))
    merged = dict(DEFAULT_OUTPUT_CONFIG)
    merged.update(user_cfg)
    return merged


def build_output_paths(config: dict) -> dict[str, Path]:
    """
    Build and create the standard output directories for a run.
    """
    out_cfg = get_output_config(config)

    root = Path(out_cfg["base_dir"]) / out_cfg["run_name"]
    points_dir = root / "points"
    plots_dir = root / "plots"
    scan_dir = root / "scan"

    for path in (root, points_dir, plots_dir, scan_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "points": points_dir,
        "plots": plots_dir,
        "scan": scan_dir,
    }


def point_label(point_index: int | None) -> str:
    """
    Stable point label for filenames.
    """
    if point_index is None:
        return "single"
    return f"point_{point_index + 1:04d}"


def build_filename_stem(
    config: dict,
    kind: str,
    point_index: int | None = None,
) -> str:
    """
    Build a consistent filename stem.
    """
    out_cfg = get_output_config(config)
    parts = [kind]

    if point_index is not None:
        parts.append(point_label(point_index))

    file_tag = out_cfg.get("file_tag")
    if file_tag:
        parts.append(str(file_tag))

    return "__".join(parts)


def should_save_point_outputs(config: dict, converged: bool) -> bool:
    out_cfg = get_output_config(config)
    if not out_cfg["save_only_converged"]:
        return True
    return bool(converged)


def handle_figure(
    fig,
    *,
    save: bool,
    show: bool,
    save_path: Path | None = None,
    dpi: int = 200,
) -> None:
    """
    Save and/or show a matplotlib figure, then close it.
    """
    if fig is None:
        return

    if save and save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def save_point_summary(
    *,
    config: dict,
    point_index: int | None,
    params: dict,
    result,
    observable_val: float | None,
    seed_label: str,
    paths: dict[str, Path],
) -> Path | None:
    """
    Save a structured per-point summary as JSON.
    """
    out_cfg = get_output_config(config)
    if not out_cfg["save_point_summary"]:
        return None

    if not should_save_point_outputs(config, result.converged):
        return None

    metadata = result.metadata or {}
    concise = out_cfg["concise"]

    data = {
        "point_index": point_index,
        "seed_label": seed_label,
        "params": dict(params),
        "converged": bool(result.converged),
        "residual_inf_norm": float(result.residual_inf_norm),
        "iterations": int(result.iterations),
        "message": str(result.message),
        "observable": None if observable_val is None or not np.isfinite(observable_val) else float(observable_val),
    }

    if concise:
        data.update(
            {
                "best_residual": float(metadata.get("best_residual", result.residual_inf_norm)),
                "phases": list(metadata.get("phases", [])),
                "final_solver": metadata.get("final_solver"),
                "accepted_near_solution": bool(metadata.get("accepted_near_solution", False)),
                "total_time": float((metadata.get("timings") or {}).get("total", np.nan)),
            }
        )
    else:
        data["metadata"] = metadata
        data["residual_history"] = [float(x) for x in result.residual_history]

    stem = build_filename_stem(config, "summary", point_index=point_index)
    path = paths["points"] / f"{stem}.json"
    save_json(data, path)
    return path


def save_final_field(
    *,
    config: dict,
    point_index: int | None,
    field,
    result,
    paths: dict[str, Path],
) -> Path | None:
    """
    Save final field array as a space-delimited text file.
    """
    out_cfg = get_output_config(config)

    if not out_cfg["save_final_field"]:
        return None

    if not should_save_point_outputs(config, result.converged):
        return None

    stem = build_filename_stem(config, "field", point_index=point_index)
    path = paths["points"] / f"{stem}.txt"

    np.savetxt(
        path,
        np.asarray(field, dtype=float).T,
        fmt="%.16e",
        delimiter=" "
    )

    return path


def save_observables_table(
    *,
    config: dict,
    param_points: list[dict],
    observable_vals,
    paths: dict[str, Path],
) -> Path | None:
    """
    Save scan observables as CSV.
    """
    out_cfg = get_output_config(config)
    if not out_cfg["save_observables"]:
        return None

    all_param_names = []
    seen = set()
    for params in param_points:
        for key in params.keys():
            if key not in seen:
                seen.add(key)
                all_param_names.append(key)

    stem = build_filename_stem(config, "observables")
    path = paths["scan"] / f"{stem}.csv"

    header = ["point_index", *all_param_names, "observable"]
    lines = [",".join(header)]

    for i, (params, obs) in enumerate(zip(param_points, observable_vals)):
        row = [str(i)]
        for key in all_param_names:
            row.append(str(params.get(key, "")))
        row.append("" if not np.isfinite(obs) else f"{float(obs):.16e}")
        lines.append(",".join(row))

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_scan_summary(
    *,
    config: dict,
    point_summaries: list[dict],
    paths: dict[str, Path],
) -> Path | None:
    """
    Save scan-level point summary table as CSV or JSON depending on concise mode.
    """
    out_cfg = get_output_config(config)
    if not out_cfg["save_scan_summary"]:
        return None

    stem = build_filename_stem(config, "scan_summary")

    if out_cfg["concise"]:
        path = paths["scan"] / f"{stem}.csv"
        header = [
            "point_index",
            "converged",
            "residual_inf_norm",
            "best_residual",
            "iterations",
            "observable",
            "total_time",
            "final_solver",
            "accepted_near_solution",
        ]
        lines = [",".join(header)]

        for s in point_summaries:
            row = [
                str(s["point_index"]),
                str(s["converged"]),
                f'{float(s["residual_inf_norm"]):.16e}',
                f'{float(s["best_residual"]):.16e}',
                str(s["iterations"]),
                "" if not np.isfinite(s["observable"]) else f'{float(s["observable"]):.16e}',
                "" if not np.isfinite(s["total_time"]) else f'{float(s["total_time"]):.16e}',
                "" if s["final_solver"] is None else str(s["final_solver"]),
                str(bool(s["accepted_near_solution"])),
            ]
            lines.append(",".join(row))

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    path = paths["scan"] / f"{stem}.json"
    save_json({"points": point_summaries}, path)
    return path


def maybe_print_output_path(config: dict, label: str, path: Path | None) -> None:
    out_cfg = get_output_config(config)
    if out_cfg["print_output_paths"] and path is not None:
        print(f"[Output] {label}: {path}")


def _build_variation_axis(spec: dict):
    """
    Build the coordinate axis for one variation specification.
    """
    import numpy as np

    vmin = spec["min"]
    vmax = spec["max"]
    num = spec["num"]
    is_log = spec.get("is_log", False)

    if is_log:
        if vmin <= 0 or vmax <= 0:
            raise ValueError("Log variation requires min > 0 and max > 0")
        return np.logspace(np.log10(vmin), np.log10(vmax), num)

    return np.linspace(vmin, vmax, num)

def make_default_scan_plot(
    config: dict,
    param_points: list[dict],
    observable_vals,
):
    """
    Default scan observable plot builder.

    Behaviour:
        - single variation -> line plot against varied parameter
        - two-parameter multi variation -> imshow heatmap
        - otherwise -> line plot against point index

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt

    variation_mode = str(config.get("variation_mode", "none")).strip().lower()

    # --------------------------------------------------
    # 1D single-parameter scan
    # --------------------------------------------------
    if variation_mode == "single":
        fig, ax = plt.subplots()

        var_name = config["single_variation"]["name"]
        xvals = np.array([params[var_name] for params in param_points], dtype=float)

        ax.plot(xvals, observable_vals, marker="o")
        ax.set_xlabel(var_name)
        ax.set_ylabel(config.get("observable_name", "observable"))

        if config["single_variation"].get("is_log", False):
            ax.set_xscale("log")

        fig.tight_layout()
        return fig

    # --------------------------------------------------
    # 2D multi-parameter scan -> imshow
    # --------------------------------------------------
    if variation_mode == "multi":
        specs = config.get("multi_variation", [])

        if len(specs) == 2:
            spec_x = specs[0]
            spec_y = specs[1]

            x_name = spec_x["name"]
            y_name = spec_y["name"]

            x_vals = _build_variation_axis(spec_x)
            y_vals = _build_variation_axis(spec_y)

            nx = len(x_vals)
            ny = len(y_vals)

            z = np.asarray(observable_vals, dtype=float).reshape(nx, ny)

            fig, ax = plt.subplots()

            im = ax.imshow(
                z.T,
                origin="lower",
                aspect="auto",
                extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
            )

            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(config.get("observable_name", "observable"))

            if spec_x.get("is_log", False):
                ax.set_xscale("log")
            if spec_y.get("is_log", False):
                ax.set_yscale("log")

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(config.get("observable_name", "observable"))

            fig.tight_layout()
            return fig

    # --------------------------------------------------
    # Fallback -> plot against point index
    # --------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(observable_vals)), observable_vals, marker="o")
    ax.set_xlabel("parameter point index")
    ax.set_ylabel(config.get("observable_name", "observable"))
    fig.tight_layout()
    return fig
