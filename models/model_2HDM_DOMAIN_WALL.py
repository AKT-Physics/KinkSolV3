#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Worked Example: 2HDM Z2 symmetric Domain Walls Global
6 field highly coupled model
Understanding of solutions were actually my first discovery
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import partial
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.grid import build_grid_from_config
from core.indexing import build_indexer
from core.boundary import (
    neumann_left_1d,
    neumann_right_1d,
    dirichlet_zero,
)


# =============================================================================
# 1. MODEL CONFIGURATION
# =============================================================================
def get_model_config():
    return {
        # --------------------------------------------------
        # Grid / geometry metadata
        # --------------------------------------------------
        "n_dims": 1,
        "geometry": "cartesian",
        "coord_names": ("x",),
        "grid_shape": (501,),       # odd -> central site is x = 0
        "grid_spacing": (0.05,),

        # --------------------------------------------------
        # Field content
        # --------------------------------------------------
        "n_fields": 6,

        # --------------------------------------------------
        # Independant model parameters
        # --------------------------------------------------
        "parameters": { 
            "m_h": 125, # FIXED --> DO NOT CHANGE
            "m_H": 500,  # }
            "m_A": 125,  #  } -->     Free Parameters
            "m_H_pm": 125,#  } --> MASSES MUST REMAIN >= m_h
            "t_b": 0.85,  #}
        },
        
        # --------------------------------------------------
        # Parameter variation options
        # --------------------------------------------------
        "variation_mode": "none",   # "none", "single", "multi"

        "single_variation": {
            "name": "m_H",
            "min": 500,
            "max": 2000,
            "num": 15,
            "is_log": False,
        },
        
        "multi_variation": [
                {
                    "name": "m_H",
                    "min": 250,
                    "max": 500,
                    "num": 4,
                    "is_log": False,
                },
                {
                    "name": "t_b",
                    "min": 0.5,
                    "max": 2.0,
                    "num": 4,
                    "is_log": False,
                },
        ],

        "observable_name": "E",
        
        "continuation_seed_eps": 1.0,
        "continuation_condensate_threshold": 1e-3,
        
        
        # --------------------------------------------------
        # Solver options
        # --------------------------------------------------
        "solver": {
            "use_pointwise_phase": True,
            "use_trust_region": False,

            "pointwise_tol": 1e-1,
            "pointwise_max_iter": 10,
            "pointwise_relaxation": 1.5,
            "pointwise_freq_out": 10,

            "final_tol": 1e-4,
            "final_max_iter": 100,
            "final_freq_out": 1,
            "final_step_scale": 1.0,

            "trust_initial_radius": 2.0,
            "trust_min_radius": 1e-6,
            "trust_max_radius": 50.0,
            "trust_shrink": 0.7,
            "trust_grow": 2.0,
            "trust_max_trials": 10,
            "trust_accept_growth": 50.0,
            "trust_good_improve": 0.8,

            "enable_rescue": True,
            "rescue_pointwise_max_iter": 100,
            "rescue_pointwise_relaxation": 1.5,

            "accept_best_near_tol": False,
            "near_tol_factor": 10.0,
        },
        
        # --------------------------------------------------
        # Output options
        # --------------------------------------------------
        "output": {
            # General output location / naming
            # --------------------------------------------------
            "base_dir": str(PROJECT_ROOT / "outputs"),
            "run_name": "model_2HDM_DOMAIN_WALL_GS",
            "file_tag": None,  # optional extra tag in filenames      
            
            # Console verbosity
            # --------------------------------------------------
            "print_point_summary": True,
            "print_scan_summary": False,
            "print_params_each_point": True,
            "print_output_paths": False,        
            
            # Plot control
            # --------------------------------------------------
            "show_solution_plots": True,
            "save_solution_plots": False,
            "show_scan_plot": True,
            "save_scan_plot": False,
            "plot_format": "pdf",
            "plot_dpi": 100,        
            
            # Data saving
            # --------------------------------------------------
            "save_point_summary": False,
            "save_scan_summary": False,
            "save_observables": False,
            "save_final_field": True,        
            
            # Save style
            # --------------------------------------------------
            "concise": True,
            "save_only_converged": True,
}
        
    }


def finalize_parameters(params):
    if params["m_H"] < params["m_h"]:
        raise ValueError("m_H must remain >= m_h")
    if params["m_A"] < params["m_h"]:
        raise ValueError("m_A must remain >= m_h")
    if params["m_H_pm"] < params["m_h"]:
        raise ValueError("m_H_pm must remain >= m_h")
    if params["t_b"] <= 0:
        raise ValueError("t_b must be positive")
    return params


# =============================================================================
# GENERIC GRID / INDEXING HELPERS
# =============================================================================
def _grid_and_indexer(config):

    grid = build_grid_from_config(config)
    indexer = build_indexer(grid, config["n_fields"])
    return grid, indexer



# =============================================================================
# MODEL EQUATIONS
# =============================================================================
F1, FP, XI1, XI2, GAMMA, ETA = range(6)

def _u(field, a, site, n_grid):
    return field[a * n_grid + site]

def _d1_central(field, a, site, n_grid, dx):
    return (_u(field, a, site + 1, n_grid) - _u(field, a, site - 1, n_grid)) / (2.0 * dx)

def _d2_central(field, a, site, n_grid, dx):
    return (
        _u(field, a, site + 1, n_grid)
        - 2.0 * _u(field, a, site, n_grid)
        + _u(field, a, site - 1, n_grid)
    ) / (dx * dx)



def f1_eq(field, i, n_grid, d_x, params, *args):
    site = i % n_grid
    
    m_h = params["m_h"]
    m_H = params["m_H"]
    m_A = params["m_A"]
    m_H_pm = params["m_H_pm"]
    t_b = params["t_b"]
    
    rH   = (m_H * m_H) / (m_h * m_h)
    rA   = (m_A * m_A) / (m_h * m_h)
    rHpm = (m_H_pm * m_H_pm) / (m_h * m_h)


    f1_i    = _u(field, F1,    site, n_grid)
    fp_i    = _u(field, FP,    site, n_grid)
    xi1_i   = _u(field, XI1,   site, n_grid)
    xi2_i   = _u(field, XI2,   site, n_grid)
    gamma_i = _u(field, GAMMA, site, n_grid)

    d2f1_dx2 = _d2_central(field, F1,    site, n_grid, d_x)
    dgamma_dx = _d1_central(field, GAMMA, site, n_grid, d_x)
    deta_dx   = _d1_central(field, ETA,   site, n_grid, d_x)

    cos_gamma_half_sq = np.cos(gamma_i / 2.0) ** 2

    return (
        d2f1_dx2
        - f1_i * (
            -0.5
            + 0.5 * (1.0 + rH * t_b * t_b) * f1_i**2
            + 0.5 * (1.0 - rH + 2.0 * rHpm) * fp_i**2
            + 0.5 * (1.0 - rH) * xi1_i**2
            + 0.5 * (1.0 + 2.0 * rA - rH) * xi2_i**2
            + 0.25 * dgamma_dx**2
            + 0.25 * cos_gamma_half_sq * deta_dx**2
        )
    )


def fp_eq(field, i, n_grid, d_x, params, *args):
    site = i % n_grid
    
    m_h = params["m_h"]
    m_H = params["m_H"]
    m_H_pm = params["m_H_pm"]
    t_b = params["t_b"]

    rH   = (m_H * m_H) / (m_h * m_h)
    rHpm = (m_H_pm * m_H_pm) / (m_h * m_h)

    f1_i    = _u(field, F1,    site, n_grid)
    fp_i    = _u(field, FP,    site, n_grid)
    xi1_i   = _u(field, XI1,   site, n_grid)
    xi2_i   = _u(field, XI2,   site, n_grid)
    gamma_i = _u(field, GAMMA, site, n_grid)
    eta_i   = _u(field, ETA,   site, n_grid)

    d2fp_dx2    = _d2_central(field, FP,    site, n_grid, d_x)
    dxi1_dx     = _d1_central(field, XI1,   site, n_grid, d_x)
    dxi2_dx     = _d1_central(field, XI2,   site, n_grid, d_x)
    dgamma_dx   = _d1_central(field, GAMMA, site, n_grid, d_x)
    d2gamma_dx2 = _d2_central(field, GAMMA, site, n_grid, d_x)
    deta_dx     = _d1_central(field, ETA,   site, n_grid, d_x)
    d2eta_dx2   = _d2_central(field, ETA,   site, n_grid, d_x)

    sin_gamma = np.sin(gamma_i)
    cos_gamma = np.cos(gamma_i)

    sin_eta_half = np.sin(eta_i / 2.0)
    cos_eta_half = np.cos(eta_i / 2.0)
    cos_gamma_half_sq = np.cos(gamma_i / 2.0) ** 2

    return (
        d2fp_dx2
        - fp_i * (
            -0.5
            + 0.5 * (1.0 - rH + 2.0 * rHpm) * f1_i**2
            + 0.5 * (1.0 + rH / (t_b * t_b)) * fp_i**2
            + 0.5 * (1.0 + rH / (t_b * t_b)) * xi1_i**2
            + 0.5 * (1.0 + rH / (t_b * t_b)) * xi2_i**2
            + 0.25 * dgamma_dx**2
            + 0.25 * cos_gamma_half_sq * deta_dx**2
        )
        + 0.25 * sin_gamma * (sin_eta_half * xi1_i - cos_eta_half * xi2_i) * d2eta_dx2
        + 0.5 * (cos_eta_half * xi1_i + sin_eta_half * xi2_i) * d2gamma_dx2
        + 0.125 * sin_gamma * (cos_eta_half * xi1_i + sin_eta_half * xi2_i) * deta_dx**2
        - 0.25 * (1.0 - cos_gamma) * (sin_eta_half * xi1_i - cos_eta_half * xi2_i) * deta_dx * dgamma_dx
        + 0.5 * sin_gamma * (sin_eta_half * dxi1_dx - cos_eta_half * dxi2_dx) * deta_dx
        + (cos_eta_half * dxi1_dx + sin_eta_half * dxi2_dx) * dgamma_dx
    )


def xi1_eq(field, i, n_grid, d_x, params, *args):
    site = i % n_grid
    
    m_h = params["m_h"]
    m_H = params["m_H"]
    t_b = params["t_b"]

    rH = (m_H * m_H) / (m_h * m_h)

    f1_i    = _u(field, F1,    site, n_grid)
    fp_i    = _u(field, FP,    site, n_grid)
    xi1_i   = _u(field, XI1,   site, n_grid)
    xi2_i   = _u(field, XI2,   site, n_grid)
    gamma_i = _u(field, GAMMA, site, n_grid)
    eta_i   = _u(field, ETA,   site, n_grid)

    d2xi1_dx2   = _d2_central(field, XI1,   site, n_grid, d_x)
    dfp_dx      = _d1_central(field, FP,    site, n_grid, d_x)
    dxi2_dx     = _d1_central(field, XI2,   site, n_grid, d_x)
    dgamma_dx   = _d1_central(field, GAMMA, site, n_grid, d_x)
    d2gamma_dx2 = _d2_central(field, GAMMA, site, n_grid, d_x)
    deta_dx     = _d1_central(field, ETA,   site, n_grid, d_x)
    d2eta_dx2   = _d2_central(field, ETA,   site, n_grid, d_x)

    sin_gamma = np.sin(gamma_i)
    cos_gamma = np.cos(gamma_i)

    sin_eta_half = np.sin(eta_i / 2.0)
    cos_eta_half = np.cos(eta_i / 2.0)

    cos_gamma_half = np.cos(gamma_i / 2.0)
    sin_gamma_half = np.sin(gamma_i / 2.0)
    cos_gamma_half_sq = cos_gamma_half**2


    return (
        d2xi1_dx2
        - xi1_i * (
            -0.5
            + 0.5 * (1.0 - rH) * f1_i**2
            + 0.5 * (1.0 + rH / (t_b * t_b)) * fp_i**2
            + 0.5 * (1.0 + rH / (t_b * t_b)) * xi1_i**2
            + 0.5 * (1.0 + rH / (t_b * t_b)) * xi2_i**2
            + 0.25 * cos_gamma_half_sq * deta_dx**2
            + 0.25 * dgamma_dx**2
        )
        - 0.5 * fp_i * cos_eta_half * d2gamma_dx2
        + (
            -0.25 * fp_i * sin_eta_half * sin_gamma
            + 0.5 * cos_gamma_half_sq * xi2_i
        ) * d2eta_dx2
        - 0.125 * cos_eta_half * fp_i * sin_gamma * deta_dx**2
        + 0.25 * (1.0 - cos_gamma) * sin_eta_half * fp_i * deta_dx * dgamma_dx
        - 0.5 * cos_gamma_half * sin_gamma_half * xi2_i * deta_dx * dgamma_dx
        - (
            0.5 * sin_eta_half * sin_gamma * deta_dx
            + cos_eta_half * dgamma_dx
        ) * dfp_dx
        + cos_gamma_half_sq * deta_dx * dxi2_dx
    )


def xi2_eq(field, i, n_grid, d_x, params, *args):
    site = i % n_grid
    
    m_h = params["m_h"]
    m_H = params["m_H"]
    m_A = params["m_A"]
    t_b = params["t_b"]
    
    rH = (m_H * m_H) / (m_h * m_h)
    rA = (m_A * m_A) / (m_h * m_h)    

    f1_i    = _u(field, F1,    site, n_grid)
    fp_i    = _u(field, FP,    site, n_grid)
    xi1_i   = _u(field, XI1,   site, n_grid)
    xi2_i   = _u(field, XI2,   site, n_grid)
    gamma_i = _u(field, GAMMA, site, n_grid)
    eta_i   = _u(field, ETA,   site, n_grid)

    d2xi2_dx2   = _d2_central(field, XI2,   site, n_grid, d_x)
    dfp_dx      = _d1_central(field, FP,    site, n_grid, d_x)
    dxi1_dx     = _d1_central(field, XI1,   site, n_grid, d_x)
    dgamma_dx   = _d1_central(field, GAMMA, site, n_grid, d_x)
    d2gamma_dx2 = _d2_central(field, GAMMA, site, n_grid, d_x)
    deta_dx     = _d1_central(field, ETA,   site, n_grid, d_x)
    d2eta_dx2   = _d2_central(field, ETA,   site, n_grid, d_x)

    sin_gamma = np.sin(gamma_i)
    cos_gamma = np.cos(gamma_i)

    sin_eta_half = np.sin(eta_i / 2.0)
    cos_eta_half = np.cos(eta_i / 2.0)

    cos_gamma_half = np.cos(gamma_i / 2.0)
    sin_gamma_half = np.sin(gamma_i / 2.0)
    cos_gamma_half_sq = cos_gamma_half**2

    return (
        d2xi2_dx2
        - xi2_i * (
            -0.5
            + 0.5 * (1.0 + 2.0 * rA - rH) * f1_i**2
            + 0.5 * (1.0 + rH / (t_b * t_b)) * fp_i**2
            + 0.5 * (1.0 + rH / (t_b * t_b)) * xi1_i**2
            + 0.5 * (1.0 + rH / (t_b * t_b)) * xi2_i**2
            + 0.25 * cos_gamma_half_sq * deta_dx**2
            + 0.25 * dgamma_dx**2
        )
        - 0.5 * fp_i * sin_eta_half * d2gamma_dx2
        + (
            0.25 * cos_eta_half * fp_i * sin_gamma
            - 0.5 * cos_gamma_half_sq * xi1_i
        ) * d2eta_dx2
        - 0.125 * fp_i * sin_eta_half * sin_gamma * deta_dx**2
        - 0.25 * (1.0 - cos_gamma) * cos_eta_half * fp_i * deta_dx * dgamma_dx
        + 0.5 * cos_gamma_half * sin_gamma_half * xi1_i * deta_dx * dgamma_dx
        + (
            0.5 * cos_eta_half * sin_gamma * deta_dx
            - sin_eta_half * dgamma_dx
        ) * dfp_dx
        - cos_gamma_half_sq * deta_dx * dxi1_dx
    )


def gamma_eq(field, i, n_grid, d_x, params, *args):
    site = i % n_grid

    f1_i    = _u(field, F1,    site, n_grid)
    fp_i    = _u(field, FP,    site, n_grid)
    xi1_i   = _u(field, XI1,   site, n_grid)
    xi2_i   = _u(field, XI2,   site, n_grid)
    gamma_i = _u(field, GAMMA, site, n_grid)
    eta_i   = _u(field, ETA,   site, n_grid)

    df1_dx      = _d1_central(field, F1,    site, n_grid, d_x)
    dfp_dx      = _d1_central(field, FP,    site, n_grid, d_x)
    dxi1_dx     = _d1_central(field, XI1,   site, n_grid, d_x)
    dxi2_dx     = _d1_central(field, XI2,   site, n_grid, d_x)
    dgamma_dx   = _d1_central(field, GAMMA, site, n_grid, d_x)
    deta_dx     = _d1_central(field, ETA,   site, n_grid, d_x)

    d2fp_dx2    = _d2_central(field, FP,    site, n_grid, d_x)
    d2xi1_dx2   = _d2_central(field, XI1,   site, n_grid, d_x)
    d2xi2_dx2   = _d2_central(field, XI2,   site, n_grid, d_x)
    d2gamma_dx2 = _d2_central(field, GAMMA, site, n_grid, d_x)

    sin_gamma = np.sin(gamma_i)
    cos_gamma = np.cos(gamma_i)

    sin_eta_half = np.sin(eta_i / 2.0)
    cos_eta_half = np.cos(eta_i / 2.0)

    sigma = f1_i**2 + fp_i**2 + xi1_i**2 + xi2_i**2

    return (
        0.25 * sigma * d2gamma_dx2
        + 0.5 * (cos_eta_half * xi1_i + sin_eta_half * xi2_i) * d2fp_dx2
        - 0.5 * fp_i * (cos_eta_half * d2xi1_dx2 + sin_eta_half * d2xi2_dx2)
        + 0.0625 * sin_gamma * sigma * deta_dx**2
        + 0.5 * (f1_i * df1_dx + fp_i * dfp_dx + xi1_i * dxi1_dx + xi2_i * dxi2_dx) * dgamma_dx
        - 0.25 * (1.0 + cos_gamma) * (sin_eta_half * xi1_i - cos_eta_half * xi2_i) * deta_dx * dfp_dx
        + 0.25 * (1.0 + cos_gamma) * fp_i * (sin_eta_half * dxi1_dx - cos_eta_half * dxi2_dx) * deta_dx
        + 0.25 * sin_gamma * (xi2_i * dxi1_dx - xi1_i * dxi2_dx) * deta_dx
    )


def eta_eq(field, i, n_grid, d_x, params, *args):
    site = i % n_grid

    f1_i    = _u(field, F1,    site, n_grid)
    fp_i    = _u(field, FP,    site, n_grid)
    xi1_i   = _u(field, XI1,   site, n_grid)
    xi2_i   = _u(field, XI2,   site, n_grid)
    gamma_i = _u(field, GAMMA, site, n_grid)
    eta_i   = _u(field, ETA,   site, n_grid)

    df1_dx      = _d1_central(field, F1,    site, n_grid, d_x)
    dfp_dx      = _d1_central(field, FP,    site, n_grid, d_x)
    dxi1_dx     = _d1_central(field, XI1,   site, n_grid, d_x)
    dxi2_dx     = _d1_central(field, XI2,   site, n_grid, d_x)
    dgamma_dx   = _d1_central(field, GAMMA, site, n_grid, d_x)
    deta_dx     = _d1_central(field, ETA,   site, n_grid, d_x)

    d2fp_dx2    = _d2_central(field, FP,    site, n_grid, d_x)
    d2xi1_dx2   = _d2_central(field, XI1,   site, n_grid, d_x)
    d2xi2_dx2   = _d2_central(field, XI2,   site, n_grid, d_x)
    d2eta_dx2   = _d2_central(field, ETA,   site, n_grid, d_x)

    sin_gamma = np.sin(gamma_i)
    cos_gamma = np.cos(gamma_i)

    sin_eta_half = np.sin(eta_i / 2.0)
    cos_eta_half = np.cos(eta_i / 2.0)

    cos_gamma_half = np.cos(gamma_i / 2.0)
    cos_gamma_half_sq = cos_gamma_half**2

    sigma = f1_i**2 + fp_i**2 + xi1_i**2 + xi2_i**2

    return (
        0.25 * sigma * cos_gamma_half_sq * d2eta_dx2
        + 0.25 * sin_gamma * (sin_eta_half * xi1_i - cos_eta_half * xi2_i) * d2fp_dx2
        + (
            -0.25 * fp_i * sin_eta_half * sin_gamma
            + 0.5 * cos_gamma_half_sq * xi2_i
        ) * d2xi1_dx2
        + (
            0.25 * cos_eta_half * fp_i * sin_gamma
            - 0.5 * cos_gamma_half_sq * xi1_i
        ) * d2xi2_dx2
        + 0.5 * cos_gamma_half_sq * (
            f1_i * df1_dx + fp_i * dfp_dx + xi1_i * dxi1_dx + xi2_i * dxi2_dx
        ) * deta_dx
        - 0.125 * sin_gamma * sigma * deta_dx * dgamma_dx
        + 0.25 * (1.0 + cos_gamma) * (
            sin_eta_half * xi1_i - cos_eta_half * xi2_i
        ) * dfp_dx * dgamma_dx
        - 0.25 * dgamma_dx * (
            (1.0 + cos_gamma) * fp_i * (sin_eta_half * dxi1_dx - cos_eta_half * dxi2_dx)
            + sin_gamma * (xi2_i * dxi1_dx - xi1_i * dxi2_dx)
        )
    )



# =============================================================================
# EXTRA BOUNDARY CONDITIONS
# =============================================================================
# Degenerate branch selector
def fp_eq_xi2_centre(field, i, n_grid, *args):
    """
    Degenerate-branch selector at the centre:
        f_+(0) - xi_2(0) = 0
    """
    centre_site = n_grid // 2
    return _u(field, FP, centre_site, n_grid) - _u(field, XI2, centre_site, n_grid)




# =============================================================================
# BOUNDARY CONDITIONS AND DEPENDENCIES
# =============================================================================

# Define the equations you want to assign boundary conditions to
def build_bulk_functions(config, params):
    grid, _ = _grid_and_indexer(config)
    n_grid = grid.n_grid
    d_x = grid.grid_spacing[0]

    return {
        F1:    partial(f1_eq,    n_grid=n_grid, d_x=d_x, params=params),
        FP:    partial(fp_eq,    n_grid=n_grid, d_x=d_x, params=params),
        XI1:   partial(xi1_eq,   n_grid=n_grid, d_x=d_x, params=params),
        XI2:   partial(xi2_eq,   n_grid=n_grid, d_x=d_x, params=params),
        GAMMA: partial(gamma_eq, n_grid=n_grid, d_x=d_x, params=params),
        ETA:   partial(eta_eq,   n_grid=n_grid, d_x=d_x, params=params),
    }


# Specify dependencies of each equation 
# If not specified then fully coupled bulk will be assumed
def build_dependency_spec(config, params):
    coupled_deps = [
        (F1, 0),
        (FP, -1), (FP, 0), (FP, +1),
        (XI1, -1), (XI1, 0), (XI1, +1),
        (XI2, -1), (XI2, 0), (XI2, +1),
        (GAMMA, -1), (GAMMA, 0), (GAMMA, +1),
        (ETA, -1), (ETA, 0), (ETA, +1),
    ]
    return {
        "default": {
            "fields": [F1, FP, XI1, XI2, GAMMA, ETA],
            "stencil_radius": 1,
        },
        "bulk_by_field": {
            F1: {
                "deps": [
                    (F1, -1), (F1, 0), (F1, +1),   # second derivative of f1
                    (FP, 0),                       # fp only algebraically
                    (XI1, 0),
                    (XI2, 0),
                    (GAMMA, -1), (GAMMA, 0), (GAMMA, +1),
                    (ETA, -1), (ETA, +1),         # eta through first derivative
                ]
            },
            # Keys in bulk_by_field may be either:
                # - a single field id, e.g. F1
                # - a tuple of field ids sharing the same dependency spec
            (FP, XI1, XI2): {"deps": coupled_deps},
            # The gamma and eta equations also contain the first derivative of f1.
            (GAMMA, ETA): {"deps": [(F1, -1), (F1, +1), *coupled_deps]},
        }
    }


def build_boundary_spec(config, params):
    bc = {"left": neumann_left_1d, "right": neumann_right_1d}
    return {field: dict(bc) for field in (F1, FP, XI1, XI2, GAMMA, ETA)}



def build_extra_rows(config, params):
    grid, _ = _grid_and_indexer(config)
    centre_site = grid.n_grid // 2

    rows = [
        {"field": XI1,   "site": centre_site, "func": dirichlet_zero, "deps": [(XI1, centre_site)]},
        {"field": GAMMA, "site": centre_site, "func": dirichlet_zero, "deps": [(GAMMA, centre_site)]},
        {"field": ETA,   "site": centre_site, "func": dirichlet_zero, "deps": [(ETA, centre_site)]},
    ]

    if params["m_A"] == params["m_H_pm"]:
        rows.append(
            {
                "field": FP,
                "site": centre_site,
                "func": partial(fp_eq_xi2_centre, n_grid=grid.n_grid),
                "deps": [(FP, centre_site), (XI2, centre_site)],
            }
        )

    return rows



# =============================================================================
# INITIAL GUESS
# =============================================================================
def build_initial_guess(config, params):
    grid, indexer = _grid_and_indexer(config)

    if not grid.is_1d:
        raise ValueError("This model currently only supports 1D.")

    x = grid.build_coordinates()[0]
    n_grid = grid.n_grid

    m_A = params["m_A"]
    m_H_pm = params["m_H_pm"]
    t_b = params["t_b"]
    
    v1 = np.cos(np.arctan(t_b))
    v2 = np.sin(np.arctan(t_b))

    field = np.zeros(indexer.n_unknowns, dtype=float)

    for site in range(n_grid):
        xi = x[site]
        field[indexer.field_index(F1, site)] = v1
        field[indexer.field_index(XI1, site)] = v2 * np.tanh(xi)

        if m_A == m_H_pm:
            field[indexer.field_index(FP, site)] = 0.25 / np.cosh(xi)
            field[indexer.field_index(XI2, site)] = 0.25 / np.cosh(xi)
            field[indexer.field_index(GAMMA, site)] = v2 * np.tanh(xi)
            field[indexer.field_index(ETA, site)] = -v2 * np.tanh(xi)

        elif m_A < m_H_pm:
            field[indexer.field_index(FP, site)] = 0.0
            field[indexer.field_index(XI2, site)] = 0.25 / np.cosh(xi)
            field[indexer.field_index(GAMMA, site)] = 0.0
            field[indexer.field_index(ETA, site)] = -v2 * np.tanh(xi)

        else:
            field[indexer.field_index(FP, site)] = 0.25 / np.cosh(xi)
            field[indexer.field_index(XI2, site)] = 0.0
            field[indexer.field_index(GAMMA, site)] = v2 * np.tanh(xi)
            field[indexer.field_index(ETA, site)] = 0.0

    return field

# Continuation guess if you expect one field to be active/inactive as you vary the parameters
def build_continuation_guess(config, params, previous_field):
    grid, indexer = _grid_and_indexer(config)
    n_grid = grid.n_grid

    prev = np.asarray(previous_field, dtype=object).reshape(-1)
    field = np.array(
        [float(v.real) if hasattr(v, "real") else float(v) for v in prev],
        dtype=float,
    )

    centre_site = n_grid // 2
    i_fp_centre = indexer.field_index(1, centre_site)
    i_xi2_centre = indexer.field_index(3, centre_site)

    fp_centre = abs(field[i_fp_centre])
    xi2_centre = abs(field[i_xi2_centre])

    threshold = config.get("continuation_condensate_threshold", 1e-6)

    if max(fp_centre, xi2_centre) > threshold:
        return field

    x = grid.build_coordinates()[0]
    eps = config.get("continuation_seed_eps", 1e-3)

    for site in range(n_grid):
        seed_profile = 0.25 / np.cosh(x[site])
        field[indexer.field_index(1, site)] += eps * seed_profile
        field[indexer.field_index(3, site)] += eps * seed_profile

    return field





# =============================================================================
# OBSERVABLES FOR PARAMETER SCANS
# =============================================================================
def compute_observable(field, config, params):
    """
    Compute the integrated 1D energy using the full 6-field domain-wall
    energy density, including the gamma/eta gradient and current-coupling terms.
    """
    grid, _ = _grid_and_indexer(config)
    d_x = grid.grid_spacing[0]
    
    m_h = params["m_h"]
    m_H = params["m_H"]
    m_A = params["m_A"]
    m_H_pm = params["m_H_pm"]
    t_b = params["t_b"]

    f1    = field[F1, :]
    fp    = field[FP, :]
    xi1   = field[XI1, :]
    xi2   = field[XI2, :]
    gamma = field[GAMMA, :]
    eta   = field[ETA, :]

    # Centered first derivatives on interior points
    df1_dx    = (f1[2:]    - f1[:-2])    / (2.0 * d_x)
    dfp_dx    = (fp[2:]    - fp[:-2])    / (2.0 * d_x)
    dxi1_dx   = (xi1[2:]   - xi1[:-2])   / (2.0 * d_x)
    dxi2_dx   = (xi2[2:]   - xi2[:-2])   / (2.0 * d_x)
    dgamma_dx = (gamma[2:] - gamma[:-2]) / (2.0 * d_x)
    deta_dx   = (eta[2:]   - eta[:-2])   / (2.0 * d_x)

    # Interior field values
    f1_i    = f1[1:-1]
    fp_i    = fp[1:-1]
    xi1_i   = xi1[1:-1]
    xi2_i   = xi2[1:-1]
    gamma_i = gamma[1:-1]
    eta_i   = eta[1:-1]

    # Useful combinations
    sigma = f1_i**2 + fp_i**2 + xi1_i**2 + xi2_i**2

    cos_eta_half   = np.cos(eta_i / 2.0)
    sin_eta_half   = np.sin(eta_i / 2.0)
    cos_gamma_half = np.cos(gamma_i / 2.0)
    cos_gamma_half_sq = cos_gamma_half**2
    sin_gamma = np.sin(gamma_i)

    # Currents
    J1p = xi1_i * dfp_dx - fp_i * dxi1_dx
    J2p = xi2_i * dfp_dx - fp_i * dxi2_dx
    J21 = xi2_i * dxi1_dx - xi1_i * dxi2_dx

    # Potential (unchanged from previous model)
    rH   = (m_H * m_H) / (m_h * m_h)
    rA   = (m_A * m_A) / (m_h * m_h)
    rHpm = (m_H_pm * m_H_pm) / (m_h * m_h)
    
    V = (
        -0.25 * f1_i**2
        -0.25 * (fp_i**2 + xi1_i**2 + xi2_i**2)
        +0.125 * (1.0 + rH * t_b * t_b) * f1_i**4
        +0.125 * (1.0 + rH / (t_b * t_b)) * (fp_i**2 + xi1_i**2 + xi2_i**2)**2
        +0.25 * (1.0 - rH + 2.0 * rHpm) * (fp_i**2 + xi1_i**2 + xi2_i**2) * f1_i**2
        +0.25 * (rA - 2.0 * rHpm) * (xi1_i**2 + xi2_i**2) * f1_i**2
        -0.25 * rA * (xi1_i**2 - xi2_i**2) * f1_i**2
        + 0.125
    )

    energy_density = (
        0.5 * df1_dx**2
        + 0.5 * dfp_dx**2
        + 0.5 * dxi1_dx**2
        + 0.5 * dxi2_dx**2
        + 0.125 * sigma * (dgamma_dx**2 + deta_dx**2 * cos_gamma_half_sq)
        + 0.5 * dgamma_dx * (J1p * cos_eta_half + J2p * sin_eta_half)
        + 0.5 * deta_dx * (
            J21 * cos_gamma_half_sq
            + 0.5 * sin_gamma * (J1p * sin_eta_half - J2p * cos_eta_half)
        )
        + V
    )

    E = d_x * np.sum(energy_density)
    return E



# =============================================================================
# SOLUTION PLOTTING
# =============================================================================
def make_solution_plot(field, config, params):
    
    grid, _ = _grid_and_indexer(config)
    x = grid.build_coordinates()[0]

    fig, ax = plt.subplots()
    ax.plot(x, field[0, :], label=r"$\mathrm{f_1}$", color="blue")
    ax.plot(x, field[1, :], label=r"$\mathrm{f_+}$", color="green", ls="--")
    ax.plot(x, field[2, :], label=r"$\mathrm{f_2}$", color="red", ls="-.")
    ax.plot(x, field[3, :], label=r"$\mathrm{\xi_2}$", color="black", ls=":")
    ax.plot(x, 0.5*field[4, :], label=r"$\mathrm{\frac{1}{2}\gamma_1}$", color="purple", ls="--")
    ax.plot(x, 0.5*field[5, :], label=r"$\mathrm{\frac{1}{2}\eta_1}$", color="olive", ls="-.")
    

    ax.set_xlabel(r"$\mathrm{x}$")
    ax.set_xlim(left=np.min(x), right=np.max(x))
    ax.set_ylim(bottom=-1.15, top=1.15)
    ax.legend(loc="lower right", ncols=2)
    fig.tight_layout()
    return fig



# =============================================================================
# Runs the model
#==============================================================================
def main():
    import multiprocessing as mp
    from run.run_scan import run_scan
    from run.run_model import run_single

    mp.freeze_support()

    model = sys.modules[__name__]
    config = get_model_config()

    variation_mode = str(config.get("variation_mode", "none")).strip().lower()

    if variation_mode in ("single", "multi"):
        print("[ModelMain] resolved mode: scan")
        run_scan(model)
        return

    if variation_mode == "none":
        print("[ModelMain] resolved mode: single")
        params = dict(config["parameters"])

        finalize_parameters = getattr(model, "finalize_parameters", None)
        if finalize_parameters is not None:
            params = finalize_parameters(dict(params))

        run_single(
            model=model,
            config=config,
            params=params,
            initial_field=None,
            point_index=None,
        )
        return

    raise ValueError(
        f"Unknown variation_mode={variation_mode!r}. "
        "Expected 'none', 'single', or 'multi'."
    )


if __name__ == "__main__":
    main()
