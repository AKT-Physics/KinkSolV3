#!/usr/bin/env python3
# Original blank template by Adam Thomasson (March 2026).
"""Blank template for a new one-dimensional KinkSol V2 model.

How to use this template
------------------------
Choose a 1D Cartesian or radial grid, then replace the bulk equations, the
illustrative boundary conditions, initial guess, observable, and plot with
those of your model. The placeholders deliberately raise NotImplementedError.

The V2 model interface uses get_model_config(), build_bulk_functions(),
build_boundary_spec(), build_dependency_spec(), and build_initial_guess().
Optional callbacks include finalize_parameters(), build_extra_rows(),
compute_observable(), and make_solution_plot(). See the worked examples for
complete equations. This version's row compiler supports only 1D grids.

Storage convention
------------------
The Newton solver holds one flattened vector with all sites of field 0,
followed by all sites of field 1, then field 2. A residual callback receives
that vector and the *flat row index* i; the spatial site is i % n_grid.
Use indexer.field_index(a, site) when indexing the flat vector. In the bulk
Jacobian specification, declare every value read by a residual as a
(field, relative_site_offset) pair.
"""

from functools import partial
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.boundary import dirichlet_zero, neumann_right_1d
from core.grid import build_grid_from_config
from core.indexing import build_indexer





# ======================================================================
# 1. MODEL CONFIGURATION
# ======================================================================
def get_model_config():
    """Set the grid, field count, scan, solver, and output options.

    Keep these consistent with the equations and the boundary conditions
    below. Changing the number of fields requires updating each builder.
    """
    return {
        # --------------------------------------------------
        # Grid / geometry metadata (this skeleton is 1D)
        # --------------------------------------------------
        "n_dims": 1,
        "geometry": "radial",          # Or "cartesian" with coord_names=("x",).
        "coord_names": ("r",),
        "grid_shape": (2500,),
        "grid_spacing": (0.01,),

        # --------------------------------------------------
        # Field content and fixed model parameters
        # --------------------------------------------------
        "n_fields": 3,
        "parameters": {"lambda_p": 0.1},

        # --------------------------------------------------
        # Parameter variation (the V2 runner constructs the points)
        # --------------------------------------------------
        "variation_mode": "none",     # "none", "single", or "multi".
        "single_variation": {
            "name": "lambda_p",
            "min": 0.1,
            "max": 10.0,
            "num": 10,
            "is_log": True,           #  min/max are physical values.
        },
        # "multi" uses a product of parameter *ranges*.
        "multi_variation": [],
        "observable_name": "E",

        # --------------------------------------------------
        # Solver: optional pointwise phase, then a sparse Newton phase.
        # Explicit values make all controls visible to new model authors.
        # --------------------------------------------------
        "solver": {
            # Phase selection (will slow down solving, but improve branch selection)
            "use_pointwise_phase": False,
            "use_trust_region": False,

            # Pointwise pre-phase
            "pointwise_tol": 1e-1,
            "pointwise_max_iter": 100,
            "pointwise_relaxation": 1.0,
            "pointwise_freq_out": 1,

            # Global sparse Newton phase
            "final_tol": 1e-9,
            "final_max_iter": 100,
            "final_freq_out": 1,
            "final_step_scale": 1.0,  # Used for plain Newton, without trust region.

            # Trust-region controls
            "trust_initial_radius": 1.0,
            "trust_min_radius": 1e-6,
            "trust_max_radius": 50.0,
            "trust_shrink": 0.75,
            "trust_grow": 1.5,
            "trust_max_trials": 10,
            "trust_accept_growth": 50.0,
            "trust_good_improve": 0.75,

            # Optional pointwise rescue after a failed global solve
            "enable_rescue": False,
            "rescue_pointwise_max_iter": 50,
            "rescue_pointwise_relaxation": 1.0,

            # Optional acceptance of a near solution
            "accept_best_near_tol": False,
            "near_tol_factor": 10.0,
        },

        # --------------------------------------------------
        # Output files and interactive plotting
        # --------------------------------------------------
        "output": {
            "base_dir": str(PROJECT_ROOT / "outputs"),
            "run_name": "model_template_BLANK",
            "show_solution_plots": True,
            "show_scan_plot": True,
            "save_point_summary": True,
            "save_scan_summary": True,
            "save_observables": True,
            "save_final_field": True,
            "save_only_converged": True,
        },
    }


# ======================================================================
# 2. PARAMETER CONSTRAINTS AND SHARED GRID HELPERS
# ======================================================================
def finalize_parameters(params):
    """Derive constrained parameters from the fixed and scanned parameters.

    For example, if beta determines lambda_1 and lambda_2, set those keys
    here and return the complete dictionary. The scan runner calls this
    separately for each scan point; the single-point main() does likewise.
    """
    return params


def _grid_and_indexer(config):
    """Use the core helpers for coordinates and field-major indexing.

    indexer.field_index(a, site) equals a * grid.n_grid + site, and
    indexer.field_slice(a) selects the complete block for field a.
    """
    grid = build_grid_from_config(config)
    indexer = build_indexer(grid, config["n_fields"])
    return grid, indexer


# ======================================================================
# 3. BLANK MODEL EQUATIONS
# ======================================================================
FIELD0, FIELD1, FIELD2 = range(3)
# For a bulk row i, use site = i % n_grid before accessing nearby values.
# For example, field[a * n_grid + site] is field a at this site; the site
# offsets +/- 1 are valid only for interior rows. Replace all three functions.
def blank_eq_field0(field, i, n_grid, d_x, params):
    raise NotImplementedError("Define the bulk equation for field 0.")


def blank_eq_field1(field, i, n_grid, d_x, params):
    raise NotImplementedError("Define the bulk equation for field 1.")


def blank_eq_field2(field, i, n_grid, d_x, params):
    raise NotImplementedError("Define the bulk equation for field 2.")


# ======================================================================
# 4. RESIDUAL AND BOUNDARY BUILDERS
# ======================================================================
# Define the equations you want to assign boundary conditions to
def build_bulk_functions(config, params):
    """Map each field to its bulk equation, with the grid and parameters bound.

    V2 creates the row-to-function mapping from this dictionary and the
    boundary specification. 
    """
    grid, _ = _grid_and_indexer(config)
    if config["n_fields"] != 3:
        raise ValueError("Update the field mapping when changing n_fields.")
    n_grid, d_x = grid.n_grid, grid.grid_spacing[0]
    return {
        FIELD0: partial(blank_eq_field0, n_grid=n_grid, d_x=d_x, params=params),
        FIELD1: partial(blank_eq_field1, n_grid=n_grid, d_x=d_x, params=params),
        FIELD2: partial(blank_eq_field2, n_grid=n_grid, d_x=d_x, params=params),
    }


def build_boundary_spec(config, params):
    """Illustrative radial BCs: zero at the origin and Neumann at the edge.

    For each field, the left row enforces u(0)=0; the right row enforces
    u(N-1)-u(N-2)=0 (a first-order zero-derivative condition). Replace these
    field by field. Cartesian walls often need different conditions. If a
    boundary residual uses other fields or distant sites, declare its
    (field, absolute_site) pairs under left_deps or right_deps.
    """
    return {
        a: {"left": dirichlet_zero, "right": neumann_right_1d}
        for a in range(config["n_fields"])
    }


# ======================================================================
# 5. SPARSE JACOBIAN DEPENDENCIES AND SPECIAL ROWS
# ======================================================================
def build_dependency_spec(config, params):
    """Conservative example: each bulk row may use all fields at i-1, i, i+1.

    Every value actually read by each residual must be declared, even if its
    derivative happens to vanish at one particular solution. An omitted
    dependency is absent from the sparse Jacobian. You may replace this broad
    default with bulk_by_field rules, e.g. {FIELD0: {"deps":
    [(FIELD0, -1), (FIELD0, 0), (FIELD0, +1), (FIELD1, 0)]}}.
    Compare the resulting Jacobian against finite differences when you add
    real equations.
    """
    return {"default": {"fields": list(range(config["n_fields"])), "stencil_radius": 1}}


def build_extra_rows(config, params):
    """Optional row overrides for origin conditions or other constraints.

    Each entry needs field, absolute site, func, and deps (pairs of field
    and absolute site). Return [] when ordinary boundary/bulk rows suffice.
    """
    return []


# ======================================================================
# 6. INITIAL GUESS
# ======================================================================
def build_initial_guess(config, params):
    """Return a flat real array: all sites of field 0, then field 1, etc.

    For example, use grid, indexer = _grid_and_indexer(config), then
    x = grid.build_coordinates()[0]. Allocate
    np.zeros(indexer.n_unknowns) and fill each indexer.field_slice(a).
    A physically informed initial profile can be crucial for convergence.
    """
    raise NotImplementedError("Define an initial guess for your model.")


# ======================================================================
# 7. OBSERVABLE AND PLOTTING
# ======================================================================
def compute_observable(field, config, params):
    """Return a scalar observable from field of shape (n_fields, n_grid).

    For example: energy, action, charge, wall tension, or string tension.
    Define the measure and whether boundary sites contribute; alternatively
    remove this optional callback if the model has no observable.
    """
    raise NotImplementedError("Define the observable, or remove this callback.")


def make_solution_plot(field, config, params):
    """Return a Matplotlib Figure for the solved field profiles.

    Plot against the coordinates from grid.build_coordinates()[0]. The
    runner handles saving, showing, and closing the returned Figure.
    Remove this optional callback if no solution plot is needed.
    """
    raise NotImplementedError("Define a solution plot, or disable plot output.")


# ======================================================================
# 8. SINGLE POINT OR PARAMETER SCAN
# ======================================================================
def main():
    from run.run_model import run_single
    from run.run_scan import run_scan

    model = sys.modules[__name__]
    config = get_model_config()
    mode = str(config.get("variation_mode", "none")).strip().lower()
    if mode == "none":
        params = finalize_parameters(dict(config["parameters"]))
        run_single(model, config, params)
    elif mode in ("single", "multi"):
        run_scan(model)
    else:
        raise ValueError(f"Unknown variation_mode: {mode}")


if __name__ == "__main__":
    main()
