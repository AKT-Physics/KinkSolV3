# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:57:08 2026

@author: Adam Thomasson
"""

# core/row_system.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompiledRowSystem:
    func_list: list
    func_index: np.ndarray
    deriv_dir: list


def _freeze_func_registry(funcs: list):
    """
    Build a stable registry of unique row functions, preserving first occurrence.
    """
    registry = []
    func_to_code = {}

    for f in funcs:
        key = id(f)
        if key not in func_to_code:
            func_to_code[key] = len(registry)
            registry.append(f)

    return registry, func_to_code


def _flat_dep_list(indexer, deps):
    """
    Convert dependency tuples like [(field, site), ...] to flat indices.
    """
    return [indexer.field_index(field, site) for field, site in deps]


def _get_bulk_dep_spec(dependency_spec, field):
    bulk_by_field = dependency_spec.get("bulk_by_field", {})

    if field in bulk_by_field:
        return bulk_by_field[field]
    
    for key, spec in bulk_by_field.items():
        if isinstance(key, tuple) and field in key:
            return spec

    default = dependency_spec.get("default", None)
    if default is None:
        raise ValueError(
            "dependency_spec must define either 'default' or field-specific entries in 'bulk_by_field'."
        )

    return default


def _build_bulk_deps_for_field(indexer, site, spec, n_grid):
    if "deps" in spec:
        deps = []
        for field, rel_offset in spec["deps"]:
            dep_site = site + rel_offset
            if dep_site < 0 or dep_site >= n_grid:
                raise ValueError(
                    f"Dependency site out of range: site={site}, offset={rel_offset}, n_grid={n_grid}"
                )
            deps.append(indexer.field_index(field, dep_site))
        return sorted(set(deps))

    if "fields" in spec:
        stencil_radius = int(spec.get("stencil_radius", 1))
        deps = []
        for field in spec["fields"]:
            for rel_offset in range(-stencil_radius, stencil_radius + 1):
                dep_site = site + rel_offset
                if dep_site < 0 or dep_site >= n_grid:
                    raise ValueError(
                        f"Dependency site out of range: site={site}, offset={rel_offset}, n_grid={n_grid}"
                    )
                deps.append(indexer.field_index(field, dep_site))
        return sorted(set(deps))

    raise ValueError(
        "Bulk dependency spec must contain either 'deps' or 'fields'."
    )


def compile_row_system(
    *,
    indexer,
    n_grid: int,
    n_fields: int,
    bulk_functions: dict[int, Any],
    boundary_spec: dict[int, dict[str, Any]],
    dependency_spec: dict[str, Any],
    extra_rows: list[dict] | None = None,
) -> CompiledRowSystem:
    """
    Compile a user-friendly row specification into:
        - func_list
        - func_index
        - deriv_dir

    Parameters
    ----------
    indexer:
        Core Indexer object.
    n_grid:
        Number of grid sites.
    n_fields:
        Number of fields.
    bulk_functions:
        Mapping field -> bulk row function.
    boundary_spec:
        Mapping field -> {"left": func, "right": func}. For boundary
        functions using other fields or distant sites, provide "left_deps"
        and/or "right_deps" as explicit (field, absolute site) pairs.
    dependency_spec:
        A "default" bulk dependency rule and/or "bulk_by_field" rules.
        Each rule lists (field, relative site offset) pairs under "deps", or
        uses "fields" with a "stencil_radius". Boundary rows use their own
        field at sites within the top-level "stencil_radius" (default 1).
    extra_rows:
        Optional list of special row overrides, each with keys:
            - "field"
            - "site"
            - "func"
            - "deps"
    """
    if extra_rows is None:
        extra_rows = []

    # --------------------------------------------------------------
    # Build registry of all functions that may appear
    # --------------------------------------------------------------
    ordered_funcs = []

    for a in range(n_fields):
        ordered_funcs.append(boundary_spec[a]["left"])
        ordered_funcs.append(boundary_spec[a]["right"])
        ordered_funcs.append(bulk_functions[a])

    for row in extra_rows:
        ordered_funcs.append(row["func"])

    func_list, func_to_code = _freeze_func_registry(ordered_funcs)

    # --------------------------------------------------------------
    # Build default func_index and deriv_dir
    # --------------------------------------------------------------
    func_index = np.zeros(indexer.n_unknowns, dtype=int)
    deriv_dir = [None] * indexer.n_unknowns

    stencil_radius = int(dependency_spec.get("stencil_radius", 1))

    # Left boundary rows
    for a in range(n_fields):
        row = indexer.field_index(a, 0)
        left_func = boundary_spec[a]["left"]
        func_index[row] = func_to_code[id(left_func)]

        explicit_deps = boundary_spec[a].get("left_deps")
        if explicit_deps is None:
            deps = [indexer.field_index(a, offset) for offset in range(stencil_radius + 1)]
        else:
            deps = _flat_dep_list(indexer, explicit_deps)
        deriv_dir[row] = deps

    # Right boundary rows
    for a in range(n_fields):
        row = indexer.field_index(a, n_grid - 1)
        right_func = boundary_spec[a]["right"]
        func_index[row] = func_to_code[id(right_func)]

        explicit_deps = boundary_spec[a].get("right_deps")
        if explicit_deps is None:
            deps = [indexer.field_index(a, offset) for offset in range(n_grid - 1 - stencil_radius, n_grid)]
        else:
            deps = _flat_dep_list(indexer, explicit_deps)
        deriv_dir[row] = deps

    # Interior rows
    for site in range(1, n_grid - 1):
        for a in range(n_fields):
            row = indexer.field_index(a, site)
            bulk_func = bulk_functions[a]
            func_index[row] = func_to_code[id(bulk_func)]
    
            spec = _get_bulk_dep_spec(dependency_spec, a)
            deps = _build_bulk_deps_for_field(indexer, site, spec, n_grid)
            deriv_dir[row] = deps

    # --------------------------------------------------------------
    # Apply special row overrides
    # --------------------------------------------------------------
    for row_spec in extra_rows:
        a = row_spec["field"]
        site = row_spec["site"]
        func = row_spec["func"]
        deps = row_spec["deps"]

        row = indexer.field_index(a, site)
        func_index[row] = func_to_code[id(func)]
        deriv_dir[row] = _flat_dep_list(indexer, deps)

    return CompiledRowSystem(
        func_list=func_list,
        func_index=func_index,
        deriv_dir=deriv_dir,
    )
