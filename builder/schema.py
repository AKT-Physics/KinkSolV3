"""Serializable editor state and validation independent of the desktop UI."""
from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict

from builder.expressions import Expression, ExpressionError, check_name
from strategy.hybrid import HybridOptions


SCHEMA_VERSION = 1
EQUATION_MODES = ("direct", "energy")


def solver_defaults():
    opts = asdict(HybridOptions())
    opts.update(use_pointwise_phase=False, use_trust_region=False, enable_rescue=False, final_tol=1e-9)
    return opts


def default_spec():
    """Return an independent blank project with common numerical defaults."""
    return {
        "version": SCHEMA_VERSION, "name": "NewModel", "geometry": "cartesian",
        "grid": {"n": 1001, "spacing": 0.05},
        "fields": [{"name": "u1", "label": "u1", "equation": "", "guess": "",
                    "left": {"type": "custom", "value": ""},
                    "right": {"type": "custom", "value": ""}}],
        "parameters": [],
        "equation_mode": "direct", "functional": {"density": "", "weight": "1"},
        "observable": {"enabled": False, "name": "E", "from_functional": False,
                       "density": "", "weight": "1", "prefactor": "1", "quadrature": "interior_sum"},
        "plots": [{"expression": "u1", "label": "u1"}], "extra_rows": [], "scans": [],
        "solver": solver_defaults(), "jacobian_autodiff": "auto",
        "output": {"base_dir": "outputs", "save_point_summary": True, "save_scan_summary": True,
                   "save_observables": True, "save_final_field": True, "save_solution_plots": True,
                   "save_scan_plot": True, "show_solution_plots": True, "show_scan_plot": True,
                   "save_only_converged": True, "concise": False, "plot_dpi": 160, "plot_format": "png"},
    }


def parameter_order(spec):
    entries = spec["parameters"]
    names = [p["name"] for p in entries]
    remaining, result = list(entries), []
    resolved = set()
    while remaining:
        progress = False
        for p in remaining[:]:
            refs = Expression(p["expression"], parameters=names).parameter_refs
            if p["kind"] == "fixed" and refs:
                raise ExpressionError(f"Fixed parameter '{p['name']}' references parameters. Mark it derived.")
            if refs <= resolved:
                result.append(p)
                resolved.add(p["name"])
                remaining.remove(p)
                progress = True
        if not progress:
            raise ExpressionError("Circular parameter definitions: " + ", ".join(p["name"] for p in remaining))
    return result


def parameter_values(spec, overrides=None):
    params = {}
    for p in parameter_order(spec):
        if p["kind"] == "fixed" and overrides and p["name"] in overrides:
            value = overrides[p["name"]]
        else:
            value = Expression(p["expression"], parameters=params).evaluate(parameters=params)
        value = float(value)
        if not math.isfinite(value):
            raise ExpressionError(f"Parameter '{p['name']}' is not finite.")
        params[p["name"]] = value
    return params


def boundary_expression(field, side):
    bc, name = field[side], field["name"]
    value = bc.get("value", "0") or "0"
    if bc["type"] == "dirichlet":
        return f"{name} - ({value})"
    if bc["type"] == "neumann":
        return f"D({name}) - ({value})"
    if bc["type"] == "robin":
        return f"({bc.get('alpha', '1')})*{name} + ({bc.get('beta', '1')})*D({name}) - ({value})"
    if bc["type"] == "custom":
        return value
    raise ExpressionError(f"Unknown boundary type '{bc['type']}'.")


def check_equation_mode(spec):
    mode = spec.get("equation_mode")
    if mode not in EQUATION_MODES:
        raise ExpressionError("Unknown equation input mode. Choose direct residual equations or static energy density.")
    return mode


def resolved_spec(spec):
    """Resolve functional equations and automatic energy without altering input."""
    mode = check_equation_mode(spec)
    spec = copy.deepcopy(spec)
    if mode == "energy":
        from builder.symbolic import derive_equations
        equations, _ = derive_equations(spec)
        for f, eq in zip(spec["fields"], equations):
            f["equation"] = eq
    if spec["observable"].get("from_functional"):
        if mode != "energy":
            raise ExpressionError("Automatic energy requires static energy density input. Define an observable explicitly for direct equations.")
        spec["observable"]["density"] = spec["functional"]["density"]
        spec["observable"]["weight"] = spec["functional"]["weight"]
    return spec


def validate_spec(spec):
    if spec.get("version") != SCHEMA_VERSION:
        raise ExpressionError("Unsupported model-builder file version.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", spec["name"]):
        raise ExpressionError("Model name must use letters, numbers, underscores, dots or hyphens, with no path separators.")
    if spec["geometry"] not in ("cartesian", "radial"):
        raise ExpressionError("Choose Cartesian or radial geometry.")
    n, dx = spec["grid"]["n"], spec["grid"]["spacing"]
    if not isinstance(n, int) or n < 3 or not math.isfinite(dx) or dx <= 0:
        raise ExpressionError("Use at least three sites and a positive finite spacing.")
    fields, params = [f["name"] for f in spec["fields"]], [p["name"] for p in spec["parameters"]]
    if not fields:
        raise ExpressionError("Declare at least one real field.")
    names = fields + params
    for name in names:
        check_name(name)
    if len(set(names)) != len(names):
        raise ExpressionError("Field and parameter names must be unique.")
    for p in spec["parameters"]:
        if p["kind"] not in ("fixed", "derived"):
            raise ExpressionError("Parameter kind must be fixed or derived.")
    parameter_values(spec)
    spec = resolved_spec(spec)
    q = "r" if spec["geometry"] == "radial" else "x"
    for f in spec["fields"]:
        if not f["equation"].strip():
            raise ExpressionError(f"Enter a residual equation for field '{f['name']}'.")
        equation = Expression(f["equation"], fields, params, q)
        if not equation.refs:
            raise ExpressionError(f"Equation for '{f['name']}' has no field dependence.")
        if not f["guess"].strip():
            raise ExpressionError(f"Enter an initial profile for field '{f['name']}'.")
        Expression(f["guess"], parameters=params, coordinate=q, max_order=0)
        for side in ("left", "right"):
            expression = boundary_expression(f, side)
            if not expression.strip():
                raise ExpressionError(f"Define the {side} boundary condition for field '{f['name']}'.")
            bc = Expression(expression, fields, params, q, max_order=1)
            if not bc.refs:
                raise ExpressionError(f"The {side} boundary for '{f['name']}' has no field dependence.")
    seen = set()
    for row in spec["extra_rows"]:
        if row["field"] not in fields:
            raise ExpressionError("Special row refers to an unknown field.")
        site = n // 2 if row["site"] == "centre" else int(row["site"])
        if not 0 <= site < n:
            raise ExpressionError("Special-row site lies outside the grid.")
        if row["site"] == "centre" and spec["geometry"] == "cartesian" and n % 2 == 0:
            raise ExpressionError("An exact centre row requires an odd Cartesian grid size.")
        if (row["field"], site) in seen:
            raise ExpressionError("Multiple special rows replace the same equation.")
        seen.add((row["field"], site))
        Expression(row["expression"], fields, params, q, max_order=1)
    if spec["observable"]["enabled"]:
        obs = spec["observable"]
        Expression(obs["density"], fields, params, q, max_order=1)
        Expression(obs["weight"], parameters=params, coordinate=q, max_order=0)
        Expression(obs["prefactor"], parameters=params, max_order=0)
        if obs["quadrature"] not in ("interior_sum", "trapezoid"):
            raise ExpressionError("Unknown integration rule.")
    for curve in spec["plots"]:
        Expression(curve["expression"], fields, params, q)
    fixed = {p["name"] for p in spec["parameters"] if p["kind"] == "fixed"}
    scan_names = set()
    for scan in spec["scans"]:
        if scan["name"] not in fixed or scan["name"] in scan_names:
            raise ExpressionError("Scan each fixed parameter at most once; derived parameters are recalculated.")
        scan_names.add(scan["name"])
        if scan["num"] < 1 or not all(math.isfinite(scan[k]) for k in ("min", "max")):
            raise ExpressionError("Scan ranges need finite endpoints and at least one point.")
        if scan["is_log"] and min(scan["min"], scan["max"]) <= 0:
            raise ExpressionError("Logarithmic scan endpoints must be positive.")
    validate_solver(spec["solver"])
    if spec["jacobian_autodiff"] not in ("auto", "scalar", "row"):
        raise ExpressionError("Unknown Jacobian mode.")
    return spec


def validate_solver(options):
    defaults = solver_defaults()
    if set(options) != set(defaults):
        raise ExpressionError("Solver settings do not match this KinkSol version.")
    for key, value in options.items():
        if type(defaults[key]) is bool:
            if type(value) is not bool:
                raise ExpressionError(f"{key} must be a boolean.")
        elif not math.isfinite(value) or value <= 0:
            raise ExpressionError(f"{key} must be positive and finite.")
        elif type(defaults[key]) is int and (type(value) is not int):
            raise ExpressionError(f"{key} must be an integer.")
    if not options["trust_min_radius"] <= options["trust_initial_radius"] <= options["trust_max_radius"]:
        raise ExpressionError("Trust radii must satisfy minimum ≤ initial ≤ maximum.")
    if not 0 < options["trust_shrink"] < 1 or options["trust_grow"] <= 1:
        raise ExpressionError("Trust shrink must be between 0 and 1; growth must exceed 1.")
    if options["trust_good_improve"] > 1:
        raise ExpressionError("Good improvement threshold must not exceed 1.")
