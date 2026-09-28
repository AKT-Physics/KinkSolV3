# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:09:31 2026

@author: Adam Thomasson
"""

# core/parameter_scan.py

from __future__ import annotations

import itertools
import numpy as np


def build_single_variation(name, vmin, vmax, num, is_log):
    vals = _build_variation_values(vmin, vmax, num, is_log)
    return [{name: val} for val in vals]


def _build_variation_values(vmin, vmax, num, is_log):
    """
    Build the 1D array of values for one variation spec.
    """
    if is_log:
        if vmin <= 0 or vmax <= 0:
            raise ValueError("Log variation requires min > 0 and max > 0")
        return np.logspace(np.log10(vmin), np.log10(vmax), num)

    return np.linspace(vmin, vmax, num)


def build_multi_variation(specs):
    """
    Build the full Cartesian product of multiple parameter variations.

    Parameters
    ----------
    specs : list[dict]
        Each dict must contain:
            - name
            - min
            - max
            - num
            - is_log

    Returns
    -------
    list[dict]
        One dict per parameter point in the Cartesian product.
    """
    if not specs:
        return []

    names = []
    value_lists = []

    for spec in specs:
        name = spec["name"]
        vals = _build_variation_values(
            spec["min"],
            spec["max"],
            spec["num"],
            spec["is_log"],
        )
        names.append(name)
        value_lists.append(vals)

    points = []
    for combo in itertools.product(*value_lists):
        points.append({name: val for name, val in zip(names, combo)})

    return points


def build_parameter_points(config, finalize_parameters=None):
    """
    Generic parameter-point builder.

    Parameters
    ----------
    config : dict
        Model config containing:
            - parameters
            - variation_mode
            - single_variation
            - multi_variation
    finalize_parameters : callable
        Model-specific callback that takes a parameter dict and returns the
        full constrained/derived parameter dict.

    Returns
    -------
    list[dict]
        Full parameter dictionaries for each scan point.
    """
    if finalize_parameters is None:
        finalize_parameters = lambda p: p

    base_params = dict(config.get("parameters", {}))
    mode = str(config.get("variation_mode", "none")).strip().lower()

    if mode == "none":
        return [finalize_parameters(base_params)]

    if mode == "single":
        v = config["single_variation"]
        varied_points = build_single_variation(
            v["name"], v["min"], v["max"], v["num"], v["is_log"]
        )

        full_points = []
        for vp in varied_points:
            p = dict(base_params)
            p.update(vp)
            full_points.append(finalize_parameters(p))
        return full_points

    if mode == "multi":
        specs = config.get("multi_variation", [])
        varied_points = build_multi_variation(specs)

        full_points = []
        for vp in varied_points:
            p = dict(base_params)
            p.update(vp)
            full_points.append(finalize_parameters(p))
        return full_points

    raise ValueError(f"Unknown variation_mode: {mode}")