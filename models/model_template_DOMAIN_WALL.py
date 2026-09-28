#!/usr/bin/env python3
"""Worked example: 1D Cartesian Z2 x U(1) domain wall.

Fields and bulk equations
-------------------------
The unknowns are phi, sigma1, and sigma2, in consecutive field blocks. The
bulk residuals are the finite-difference versions of

    phi_xx    - lambda_1/2 * (phi^2 - eta_1^2) * phi
              - beta * phi * (sigma1^2 + sigma2^2) = 0,
    sigma1_xx - lambda_2/2 * (sigma1^2 + sigma2^2 - eta_2^2) * sigma1
              - beta * phi^2 * sigma1 = 0,
    sigma2_xx - lambda_2/2 * (sigma1^2 + sigma2^2 - eta_2^2) * sigma2
              - beta * phi^2 * sigma2 = 0.

Both ends use Neumann conditions for every field.
"""

from functools import partial
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.boundary import neumann_left_1d, neumann_right_1d
from core.grid import build_grid_from_config
from core.indexing import build_indexer



# ======================================================================
# 1. MODEL CONFIGURATION
# ======================================================================
def get_model_config():
    """Set the physical box, fixed parameters, solver stages, and scan."""
    return {
        # --------------------------------------------------
        # Centered Cartesian grid; an odd size puts x=0 at the middle site
        # --------------------------------------------------
        "n_dims": 1,
        "geometry": "cartesian",
        "coord_names": ("x",),
        "grid_shape": (10001,),
        "grid_spacing": (0.005,),

        # --------------------------------------------------
        # Fields and independent model parameters
        # --------------------------------------------------
        "n_fields": 3,
        "parameters": {
            "eta_1": 1.0,
            "eta_2": np.sqrt(3) / 2.0,
            "beta": 1.0,
            "wall_width": 1.0,     # Initial guess only.
        },

        # --------------------------------------------------
        # A single beta scan changes lambda_1 and lambda_2 as well
        # --------------------------------------------------
        "variation_mode": "none",
        "single_variation": {
            "name": "beta", "min": 1.0, "max": 10.0,
            "num": 10, "is_log": False,
        },
        "multi_variation": [],
        "observable_name": "E",

        # --------------------------------------------------
        # Solver phases and settings
        # --------------------------------------------------
        "solver": {
            # Phase selection
            "use_pointwise_phase": False,
            "use_trust_region": False,

            # Pointwise pre-phase (inactive while use_pointwise_phase=False)
            "pointwise_tol": 1e-2,
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
        # Files, summaries, and interactive plots
        # --------------------------------------------------
        "output": {
            "base_dir": str(PROJECT_ROOT / "outputs"),
            "run_name": "model_template_DOMAIN_WALL",
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
# 2. PARAMETER CONSTRAINTS AND INDEXING
# ======================================================================
def finalize_parameters(params):
    """
    The scan runner applies this independently to every varied beta value.
    """
    beta = params["beta"]
    params["lambda_1"] = 2.0 * beta
    params["lambda_2"] = 2.0 * beta
    return params


def _grid_and_indexer(config):
    """Grid helpers."""
    grid = build_grid_from_config(config)
    indexer = build_indexer(grid, config["n_fields"])
    return grid, indexer


def _u(field, a, site, n_grid):
    """Read field a at spatial site from the flattened solver vector."""
    return field[a * n_grid + site]


def _d2(field, a, site, n_grid, dx):
    """Centered second spatial derivative at an interior site."""
    return (
        _u(field, a, site + 1, n_grid)
        - 2.0 * _u(field, a, site, n_grid)
        + _u(field, a, site - 1, n_grid)
    ) / (dx * dx)


# ======================================================================
# 3. MODEL EQUATIONS (INTERIOR ROWS)
# ======================================================================
PHI, SIGMA1, SIGMA2 = range(3)

def phi_eq(field, i, n_grid, d_x, params):
    """phi_xx - lambda_1/2*(phi^2-eta_1^2)*phi - beta*phi*|sigma|^2.

    i is the flat residual-row index; i % n_grid is the spatial site. This
    function is called only for interior rows, where site +/- 1 exists.
    """
    site = i % n_grid
    phi = _u(field, PHI, site, n_grid)
    sigma1 = _u(field, SIGMA1, site, n_grid)
    sigma2 = _u(field, SIGMA2, site, n_grid)
    return (
        _d2(field, PHI, site, n_grid, d_x)
        - 0.5 * params["lambda_1"] * (phi * phi - params["eta_1"] ** 2) * phi
        - params["beta"] * phi * (sigma1 * sigma1 + sigma2 * sigma2)
    )


def sigma1_eq(field, i, n_grid, d_x, params):
    """sigma1_xx - lambda_2/2*(|sigma|^2-eta_2^2)*sigma1
    - beta*phi^2*sigma1. The two sigma components couple locally.
    """
    site = i % n_grid
    phi = _u(field, PHI, site, n_grid)
    sigma1 = _u(field, SIGMA1, site, n_grid)
    sigma2 = _u(field, SIGMA2, site, n_grid)
    return (
        _d2(field, SIGMA1, site, n_grid, d_x)
        - 0.5 * params["lambda_2"]
        * (sigma1 * sigma1 + sigma2 * sigma2 - params["eta_2"] ** 2) * sigma1
        - params["beta"] * phi * phi * sigma1
    )


def sigma2_eq(field, i, n_grid, d_x, params):
    """sigma2_xx - lambda_2/2*(|sigma|^2-eta_2^2)*sigma2
    - beta*phi^2*sigma2. The two sigma components couple locally.
    """
    site = i % n_grid
    phi = _u(field, PHI, site, n_grid)
    sigma1 = _u(field, SIGMA1, site, n_grid)
    sigma2 = _u(field, SIGMA2, site, n_grid)
    return (
        _d2(field, SIGMA2, site, n_grid, d_x)
        - 0.5 * params["lambda_2"]
        * (sigma1 * sigma1 + sigma2 * sigma2 - params["eta_2"] ** 2) * sigma2
        - params["beta"] * phi * phi * sigma2
    )


# ======================================================================
# 4. V2 ROW, BOUNDARY, AND JACOBIAN BUILDERS
# ======================================================================
def build_bulk_functions(config, params):
    """Map each field to its equation, binding n_grid, dx, and parameters.
    """
    grid, _ = _grid_and_indexer(config)
    n_grid, d_x = grid.n_grid, grid.grid_spacing[0]
    return {
        PHI: partial(phi_eq, n_grid=n_grid, d_x=d_x, params=params),
        SIGMA1: partial(sigma1_eq, n_grid=n_grid, d_x=d_x, params=params),
        SIGMA2: partial(sigma2_eq, n_grid=n_grid, d_x=d_x, params=params),
    }


def build_boundary_spec(config, params):
    """Use the first-order zero-derivative rows on both ends.

    Left: u(1)-u(0)=0; right: u(N-1)-u(N-2)=0 for all three fields.
    The shared boundary callbacks come from core.boundary.
    """
    return {
        a: {"left": neumann_left_1d, "right": neumann_right_1d}
        for a in (PHI, SIGMA1, SIGMA2)
    }


def build_dependency_spec(config, params):
    """Declare all values each residual reads for the sparse Jacobian.

    Every interior field needs its own three-point stencil and both other
    fields at the same site. The boundary rows instead use their two local
    sites, as supplied by the ordinary 1D boundary compiler. A dependency
    missing here would make the sparse Newton Jacobian incorrect.
    """
    return {
        "bulk_by_field": {
            PHI: {"deps": [
                (PHI, -1), (PHI, 0), (PHI, +1),
                (SIGMA1, 0), (SIGMA2, 0),
            ]},
            SIGMA1: {"deps": [
                (PHI, 0),
                (SIGMA1, -1), (SIGMA1, 0), (SIGMA1, +1),
                (SIGMA2, 0),
            ]},
            SIGMA2: {"deps": [
                (PHI, 0), (SIGMA1, 0),
                (SIGMA2, -1), (SIGMA2, 0), (SIGMA2, +1),
            ]},
        }
    }


# ======================================================================
# 5. INITIAL PROFILES
# ======================================================================
def build_initial_guess(config, params):
    """Use phi=tanh(x/w), sigma1=sech(x/w), sigma2=-sech(x/w).
    """
    grid, indexer = _grid_and_indexer(config)
    x = grid.build_coordinates()[0]
    width = float(params.get("wall_width", 1.0))
    if width <= 0:
        raise ValueError("wall_width must be positive.")
    profile = x / width
    bump = 1.0 / np.cosh(profile)
    field = np.empty(indexer.n_unknowns, dtype=float)
    field[indexer.field_slice(PHI)] = np.tanh(profile)
    field[indexer.field_slice(SIGMA1)] = bump
    field[indexer.field_slice(SIGMA2)] = -bump
    return field


# ======================================================================
# 6. INTEGRATED OBSERVABLE
# ======================================================================
def compute_observable(field, config, params):
    """Compute the older example's integrated 1D energy.

    E = integral dx [(phi_x)^2 + (sigma1_x)^2 + (sigma2_x)^2
         + lambda_1/4*(phi^2-eta_1^2)^2
         + lambda_2/4*(sigma1^2+sigma2^2-eta_2^2)^2
         + beta*phi^2*(sigma1^2+sigma2^2)].

    Centered first derivatives and fields are sampled at interior sites;
    the two outer sites do not enter the sum. The runner passes field with
    shape (n_fields, n_grid), not the flat residual vector.
    """
    dx = build_grid_from_config(config).grid_spacing[0]
    phi, sigma1, sigma2 = field
    dphi = (phi[2:] - phi[:-2]) / (2.0 * dx)
    ds1 = (sigma1[2:] - sigma1[:-2]) / (2.0 * dx)
    ds2 = (sigma2[2:] - sigma2[:-2]) / (2.0 * dx)
    p, s1, s2 = phi[1:-1], sigma1[1:-1], sigma2[1:-1]
    density = (
        dphi ** 2 + ds1 ** 2 + ds2 ** 2
        + 0.25 * params["lambda_1"] * (p ** 2 - params["eta_1"] ** 2) ** 2
        + 0.25 * params["lambda_2"]
        * (s1 ** 2 + s2 ** 2 - params["eta_2"] ** 2) ** 2
        + params["beta"] * p ** 2 * (s1 ** 2 + s2 ** 2)
    )
    return float(dx * np.sum(density))


# ======================================================================
# 7. SOLUTION PLOT
# ======================================================================
def make_solution_plot(field, config, params):
    """Return a Figure of three profiles and sigma1^2+sigma2^2 vs x."""
    x = build_grid_from_config(config).build_coordinates()[0]
    fig, ax = plt.subplots()
    ax.plot(x, field[PHI], label=r"$\phi$")
    ax.plot(x, field[SIGMA1], label=r"$\sigma_1$")
    ax.plot(x, field[SIGMA2], label=r"$\sigma_2$")
    ax.plot(x, field[SIGMA1] ** 2 + field[SIGMA2] ** 2, label=r"$|\sigma|^2$")
    ax.set(xlabel="x", ylabel="field value")
    ax.set_xlim(left=np.min(x), right=np.max(x))
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
        params = finalize_parameters(dict(config["parameters"]))
        run_single(model, config, params)
    elif mode in ("single", "multi"):
        run_scan(model)
    else:
        raise ValueError(f"Unknown variation_mode: {mode}")


if __name__ == "__main__":
    main()
