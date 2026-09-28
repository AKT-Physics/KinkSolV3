# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:16:42 2026

@author: Adam Thomasson
"""

# core/problem.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .grid import Grid, build_grid_from_config
from .indexing import Indexer, build_indexer
from .row_system import compile_row_system


@dataclass(frozen=True)
class CompiledProblem:
    """
    Frozen problem definition passed to solvers and runners.

    This is the bridge between:
        - model authoring layer
        - numerical backend / solver layer

    The model still defines the physics. This object freezes all metadata and
    callbacks into one place so the solver no longer depends directly on the
    model module structure.
    """
    name: str
    model: Any
    config: dict
    params: dict

    grid: Grid
    indexer: Indexer

    n_dims: int
    geometry: str
    n_fields: int
    n_grid: int
    n_unknowns: int

    func_list: list
    func_index: np.ndarray
    deriv_dir: list

    build_initial_guess_fn: Callable[[dict, dict], Any]
    build_continuation_guess_fn: Optional[Callable[[dict, dict, Any], Any]]
    compute_observable_fn: Optional[Callable[[Any, dict, dict], Any]]
    plot_solution_fn: Optional[Callable[[Any, dict, dict], Any]]

    def build_initial_guess(self):
        """
        Build the initial guess and return it as a flat float array.
        """
        u0 = self.build_initial_guess_fn(self.config, self.params)
        return self._coerce_flat_real_array(u0)

    def build_continuation_guess(self, previous_field):
        """
        Build a continuation guess from a previous solution.
        Falls back to a flattened copy if no continuation callback is provided.
        """
        if self.build_continuation_guess_fn is None:
            return self._coerce_flat_real_array(previous_field)

        u = self.build_continuation_guess_fn(self.config, self.params, previous_field)
        return self._coerce_flat_real_array(u)

    def reshape_solution(self, u):
        """
        Reshape a flat unknown vector to (n_fields, n_grid).
        """
        arr = self._coerce_flat_real_array(u)
        return self.indexer.split_fields(arr)

    def flatten_solution(self, field_array):
        """
        Flatten a field array of shape (n_fields, n_grid) to 1D.
        """
        arr = np.asarray(field_array, dtype=float)
        return self.indexer.flatten_fields(arr)

    def compute_observable(self, field_array):
        """
        Evaluate the model observable if available.
        """
        if self.compute_observable_fn is None:
            raise NotImplementedError("This model does not define compute_observable().")
        return self.compute_observable_fn(field_array, self.config, self.params)

    def plot_solution(self, field_array):
        """
        Plot the solution if the model provides a plotting callback.
        """
        if self.plot_solution_fn is None:
            raise NotImplementedError("This model does not define plot_solution().")
        return self.plot_solution_fn(field_array, self.config, self.params)

    def _coerce_flat_real_array(self, x) -> np.ndarray:
        """
        Convert input to a flat float array of size n_unknowns.

        Handles:
        - Python lists
        - NumPy arrays
        - object arrays
        - arrays containing Dual-like values with a `.real` attribute
        """
        arr = np.asarray(x, dtype=object).reshape(-1)

        out = np.empty(arr.size, dtype=float)
        for i, v in enumerate(arr):
            if hasattr(v, "real"):
                out[i] = float(v.real)
            else:
                out[i] = float(v)

        if out.size != self.n_unknowns:
            raise ValueError(
                f"Expected vector of size {self.n_unknowns}, got {out.size}."
            )
        return out


def compile_problem(model: Any, config: dict, params: dict) -> CompiledProblem:
    grid = build_grid_from_config(config)
    if grid.n_dims != 1:
        raise NotImplementedError(
            "The row-system compiler currently supports only 1D grids. "
            "Flattened 2D/3D stencils need a separate dependency compiler."
        )
    if grid.n_grid < 3:
        raise ValueError("The 1D row-system compiler needs at least three grid sites.")

    n_fields = int(config["n_fields"])
    indexer = build_indexer(grid, n_fields)

    required = (
        "build_bulk_functions",
        "build_boundary_spec",
        "build_dependency_spec",
        "build_initial_guess",
    )

    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise AttributeError(
            "Model is missing required new row-system API functions: "
            + ", ".join(missing)
        )

    extra_rows = (
        model.build_extra_rows(config, params)
        if hasattr(model, "build_extra_rows")
        else []
    )

    row_system = compile_row_system(
        indexer=indexer,
        n_grid=grid.n_grid,
        n_fields=n_fields,
        bulk_functions=model.build_bulk_functions(config, params),
        boundary_spec=model.build_boundary_spec(config, params),
        dependency_spec=model.build_dependency_spec(config, params),
        extra_rows=extra_rows,
    )

    func_list = row_system.func_list
    func_index = np.asarray(row_system.func_index, dtype=int).reshape(-1)
    deriv_dir = row_system.deriv_dir

    n_unknowns = indexer.n_unknowns

    if func_index.size != n_unknowns:
        raise ValueError(
            f"func_index has size {func_index.size}, expected {n_unknowns}."
        )

    if len(deriv_dir) != n_unknowns:
        raise ValueError(
            f"deriv_dir has length {len(deriv_dir)}, expected {n_unknowns}."
        )

    if len(func_list) == 0:
        raise ValueError("func_list must not be empty.")

    if np.min(func_index) < 0 or np.max(func_index) >= len(func_list):
        raise ValueError("func_index contains out-of-range function codes.")

    name = getattr(model, "__name__", "unnamed_model")

    return CompiledProblem(
        name=name,
        model=model,
        config=dict(config),
        params=dict(params),
        grid=grid,
        indexer=indexer,
        n_dims=grid.n_dims,
        geometry=grid.geometry,
        n_fields=n_fields,
        n_grid=grid.n_grid,
        n_unknowns=n_unknowns,
        func_list=func_list,
        func_index=func_index,
        deriv_dir=deriv_dir,
        build_initial_guess_fn=model.build_initial_guess,
        build_continuation_guess_fn=getattr(model, "build_continuation_guess", None),
        compute_observable_fn=getattr(model, "compute_observable", None),
        plot_solution_fn=getattr(model, "make_solution_plot", None),
    )
