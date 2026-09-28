#!/usr/bin/env python3
# Original radial-string template by Adam Thomasson (March 2026).
"""Worked example: 1D radial-string ansatz, ported to KinkSol V2.

Purpose and conventions
-----------------------
This is a working counterpart to model_template_BLANK.py. It has a radial
grid, three fields f, h, and gamma, a lambda_p scan, an energy-like radial
integral, and profile plots. The field blocks are flattened consecutively:
all f sites, then all h sites, then all gamma sites. At a bulk residual row
i, the radial lattice index is site = i % n_grid and r = site * dr.
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


F, H, GAMMA = range(3)


# ======================================================================
# 1. MODEL CONFIGURATION
# ======================================================================
def get_model_config():
    """Set the radial grid, parameter scan, solver, and output options.
    """
    return {
        # --------------------------------------------------
        # Radial grid: r=0 at site 0
        # --------------------------------------------------
        "n_dims": 1,
        "geometry": "radial",
        "coord_names": ("r",),
        "grid_shape": (10000,),    # Legacy resolution; reduce for a quick check.
        "grid_spacing": (0.0025,),

        # --------------------------------------------------
        # Fields and fixed parameters
        # --------------------------------------------------
        "n_fields": 3,
        "parameters": {"lambda_p": 1.0},

        # --------------------------------------------------
        # V2 scan: positive endpoint values, including for log spacing
        # --------------------------------------------------
        "variation_mode": "none",
        "single_variation": {
            "name": "lambda_p", "min": 0.1, "max": 10.0,
            "num": 3, "is_log": True,
        },
        "multi_variation": [],
        "observable_name": "E",

        # --------------------------------------------------
        # Solver phases: all settings exposed.
        # --------------------------------------------------
        "solver": {
            # Phase selection
            "use_pointwise_phase": False,
            "use_trust_region": False,

            # Pointwise pre-phase (inactive while use_pointwise_phase=False)
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
        # Outputs and interactive plots
        # --------------------------------------------------
        "output": {
            "base_dir": str(PROJECT_ROOT / "outputs"),
            "run_name": "model_template_SPHALERON",
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
# 2. GRID AND FLAT FIELD INDEXING
# ======================================================================
def _grid_and_indexer(config):
    """Obtain the radial coordinates and field-major indexing from core."""
    grid = build_grid_from_config(config)
    indexer = build_indexer(grid, config["n_fields"])
    return grid, indexer


def _u(field, a, site, n_grid):
    """Read field a at a radial lattice site from the flat unknown vector."""
    return field[a * n_grid + site]


def _d2(field, a, site, n_grid, dx):
    """Centered second derivative at an interior radial site."""
    return (
        _u(field, a, site + 1, n_grid)
        - 2.0 * _u(field, a, site, n_grid)
        + _u(field, a, site - 1, n_grid)
    ) / (dx * dx)


# ======================================================================
# 3. MODEL EQUATIONS
# ======================================================================
def f_eq(field, i, n_grid, d_x, params):
    """Residual for f at an interior site with r > 0.

    The radial Laplacian is f_rr + 2*f_r/r. The other terms couple f to
    (1-h)^2, sin^2(gamma/2)/r^2, gamma_r^2/8, and the potential term
    lambda_p*f*(f^2-1). The r=0 row is supplied separately as a boundary
    condition to avoid division by zero.
    """
    site = i % n_grid
    r = site * d_x     # Bulk rows begin at site 1; r is strictly positive.
    f = _u(field, F, site, n_grid)
    h = _u(field, H, site, n_grid)
    gamma = _u(field, GAMMA, site, n_grid)
    gamma_minus = _u(field, GAMMA, site - 1, n_grid)
    gamma_plus = _u(field, GAMMA, site + 1, n_grid)
    radial_laplacian = (
        _d2(field, F, site, n_grid, d_x)
        + (_u(field, F, site + 1, n_grid) - _u(field, F, site - 1, n_grid))
        / (r * d_x)
    )
    gamma_gradient_sq = (gamma_plus - gamma_minus) ** 2 / (4.0 * d_x ** 2)
    omega8_over_r2 = np.sin(0.5 * gamma) ** 2 / r ** 2 + 0.125 * gamma_gradient_sq
    return (
        radial_laplacian
        - 2.0 * omega8_over_r2 * f * (1.0 - h) ** 2
        - params["lambda_p"] * f * (f ** 2 - 1.0)
    )


def h_eq(field, i, n_grid, d_x):
    """Residual for h at an interior site.

    h_rr + f^2*(1-h)/4 - 2*h*(1-h)*(1-2*h)/r^2 = 0.
    The site 0 row is Dirichlet, avoiding the singular radial terms.
    """
    site = i % n_grid
    r = site * d_x
    h = _u(field, H, site, n_grid)
    f = _u(field, F, site, n_grid)
    return (
        _d2(field, H, site, n_grid, d_x)
        + 0.25 * f * f * (1.0 - h)
        - 2.0 * h * (1.0 - h) * (1.0 - 2.0 * h) / (r * r)
    )


def gamma_eq(field, i):
    """Algebraic row gamma-pi=0 at the origin and bulk sites."""
    return field[i] - np.pi


# ======================================================================
# 4. RESIDUAL, BOUNDARY, AND JACOBIAN BUILDERS
# ======================================================================
def build_bulk_functions(config, params):
    """Bind grid and parameter values to the three interior residuals."""
    grid, _ = _grid_and_indexer(config)
    n_grid, d_x = grid.n_grid, grid.grid_spacing[0]
    return {
        F: partial(f_eq, n_grid=n_grid, d_x=d_x, params=params),
        H: partial(h_eq, n_grid=n_grid, d_x=d_x),
        GAMMA: gamma_eq,
    }


def build_boundary_spec(config, params):
    """
    At r=0: f=0, h=0, gamma=pi. At the final site: each field has a
    first-order Neumann condition u(N-1)-u(N-2)=0. In particular the last
    gamma row is Neumann rather than the algebraic gamma-pi residual.
    """
    return {
        F: {"left": dirichlet_zero, "right": neumann_right_1d},
        H: {"left": dirichlet_zero, "right": neumann_right_1d},
        GAMMA: {"left": gamma_eq, "right": neumann_right_1d},
    }


def build_dependency_spec(config, params):
    """List all bulk couplings for sparse automatic differentiation.

    The f equation reads neighboring f and gamma, plus local h and gamma.
    The h equation reads neighboring h and local f. The gamma row reads
    gamma only at the same site. Boundary rows have their own dependencies.
    These are the actual values read by the equations, not a generic stencil
    for all three fields; missing entries would corrupt the Jacobian.
    """
    return {
        "bulk_by_field": {
            F: {"deps": [
                (F, -1), (F, 0), (F, +1),
                (H, 0),
                (GAMMA, -1), (GAMMA, 0), (GAMMA, +1),
            ]},
            H: {"deps": [
                (F, 0), (H, -1), (H, 0), (H, +1),
            ]},
            GAMMA: {"deps": [(GAMMA, 0)]},
        }
    }


# ======================================================================
# 5. INITIAL GUESS
# ======================================================================
def build_initial_guess(config, params):
    """Use the seed f=tanh(r), h=r^2/(1+r^2), gamma=pi.

    Return a single flat vector with one complete block per field.
    """
    grid, indexer = _grid_and_indexer(config)
    r = grid.build_coordinates()[0]
    field = np.empty(indexer.n_unknowns, dtype=float)
    field[indexer.field_slice(F)] = np.tanh(r)
    field[indexer.field_slice(H)] = r ** 2 / (1.0 + r ** 2)
    field[indexer.field_slice(GAMMA)] = np.pi
    return field


# ======================================================================
# 6. ENERGY-LIKE OBSERVABLE
# ======================================================================
def compute_observable(field, config, params):
    """Evaluate the original energy-like sum over interior radial sites.

    The observable uses the radial measure r^2 dr and centered derivatives:
    r^2 dr * [f_r^2/2 + f^2(1-h)^2/r^2
             + lambda_p*(f^2-1)^2/4
             + 4*(h_r^2 + 2*h^2*(1-h)^2/r^2)/r^2].
    The runner passes field reshaped to (n_fields, n_grid). Boundary sites
    are excluded.
    """
    grid = build_grid_from_config(config)
    dr = grid.grid_spacing[0]
    r = grid.build_coordinates()[0][1:-1]
    f, h, _ = field
    df = (f[2:] - f[:-2]) / (2.0 * dr)
    dh = (h[2:] - h[:-2]) / (2.0 * dr)
    f_i, h_i = f[1:-1], h[1:-1]
    density = (
        0.5 * df ** 2
        + f_i ** 2 * (1.0 - h_i) ** 2 / r ** 2
        + 0.25 * params["lambda_p"] * (f_i ** 2 - 1.0) ** 2
        + 4.0 * (
            dh ** 2 + 2.0 * h_i ** 2 * (1.0 - h_i) ** 2 / r ** 2
        ) / r ** 2
    )
    return float(np.sum(r ** 2 * dr * density))


# ======================================================================
# 7. SOLUTION PLOT
# ======================================================================
def make_solution_plot(field, config, params):
    """Return a Figure showing f, h, and gamma on the radial grid."""
    r = build_grid_from_config(config).build_coordinates()[0]
    fig, ax = plt.subplots()
    ax.plot(r, field[F], label="f")
    ax.plot(r, field[H], label="h")
    ax.plot(r, field[GAMMA], label=r"$\gamma$")
    ax.set_xlim(-r[-1] / 100.0, r[-1] * 1.01)
    ax.set_ylim(-0.1, np.pi + 0.1)
    ax.set(xlabel="r", ylabel="field value")
    ax.legend()
    fig.tight_layout()
    return fig


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
        run_single(model, config, dict(config["parameters"]))
    elif mode in ("single", "multi"):
        run_scan(model)
    else:
        raise ValueError(f"Unknown variation_mode: {mode}")


if __name__ == "__main__":
    main()