"""Generate portable model modules using only the original numerical backend."""
from __future__ import annotations

import ast
import hashlib
import json
import pprint
from pathlib import Path

from builder import __version__
from builder.expressions import Expression, ExpressionError
from builder.schema import SCHEMA_VERSION, boundary_expression, check_equation_mode, parameter_order, validate_spec


BODY_MARKER = "# BEGIN GENERATED MODEL\n"


class ModifiedModelError(ValueError):
    """A Python file must be preserved in the advanced editor."""


HELPERS = '''from functools import partial
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt


def _find_project_root():
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for candidate in (start, *start.parents):
            if (candidate / "core" / "problem.py").is_file():
                return candidate
    raise RuntimeError("Place this model inside KinkSol, or run it from the KinkSol project folder.")


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.grid import build_grid_from_config
from core.indexing import build_indexer


def _u(field, a, site, n_grid):
    return field[a * n_grid + site]


def _d1(field, a, site, n_grid, dx):
    if site == 0:
        return (_u(field, a, 1, n_grid) - _u(field, a, 0, n_grid)) / dx
    if site == n_grid - 1:
        return (_u(field, a, site, n_grid) - _u(field, a, site - 1, n_grid)) / dx
    return (_u(field, a, site + 1, n_grid) - _u(field, a, site - 1, n_grid)) / (2.0 * dx)


def _d2(field, a, site, n_grid, dx):
    return (_u(field, a, site + 1, n_grid) - 2.0 * _u(field, a, site, n_grid)
            + _u(field, a, site - 1, n_grid)) / dx**2


def _profile_d1(values, dx):
    return np.gradient(values, dx)


def _profile_d2(values, dx):
    out = np.empty_like(values)
    out[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / dx**2
    out[0], out[-1] = out[1], out[-2]
    return out


def _geometry(config):
    grid = build_grid_from_config(config)
    n_grid, dx = grid.n_grid, grid.grid_spacing[0]
    origin = 0.0 if grid.geometry == "radial" else -0.5 * (n_grid - 1) * dx
    return grid, n_grid, dx, origin


'''


def generate_model(original):
    spec = validate_spec(original)
    names = [f["name"] for f in spec["fields"]]
    params = [p["name"] for p in spec["parameters"]]
    q = "r" if spec["geometry"] == "radial" else "x"
    scans = spec["scans"]
    config = {"n_dims": 1, "geometry": spec["geometry"], "coord_names": (q,),
              "grid_shape": (spec["grid"]["n"],), "grid_spacing": (spec["grid"]["spacing"],),
              "n_fields": len(names), "field_names": names, "parameters": {},
              "variation_mode": "none" if not scans else "single" if len(scans) == 1 else "multi",
              "single_variation": scans[0] if len(scans) == 1 else {}, "multi_variation": scans if len(scans) > 1 else [],
              "observable_name": spec["observable"]["name"], "solver": spec["solver"],
              "jacobian_autodiff": spec["jacobian_autodiff"],
              "output": {**spec["output"], "run_name": spec["name"]}}
    chunks = [HELPERS, f"FIELD_NAMES = {names!r}\n\n\n", "def get_model_config():\n    params = {}\n"]
    ordered = parameter_order(spec)
    for p in ordered:
        if p["kind"] == "fixed":
            code = Expression(p["expression"]).emit(None, None)
            chunks.append(f"    params[{p['name']!r}] = float({code})\n")
    literal = pprint.pformat(config, sort_dicts=False, width=95).replace("\n", "\n    ")
    chunks.append(f"    config = {literal}\n    config['parameters'] = params\n")
    chunks.append("    config['output']['base_dir'] = str(PROJECT_ROOT / config['output']['base_dir'])\n    return config\n\n\n")
    chunks.append("def finalize_parameters(params):\n    params = dict(params)\n")
    for p in ordered:
        if p["kind"] == "derived":
            code = Expression(p["expression"], parameters=params).emit(None, None)
            chunks.append(f"    params[{p['name']!r}] = float({code})\n")
    chunks.append("    return params\n\n\n")

    def parse(text, max_order=2):
        return Expression(text, names, params, q, max_order=max_order)

    def scalar_code(expr):
        return expr.emit(lambda name: f"_u(field, {names.index(name)}, site, n_grid)",
                         lambda name, order: f"_d{order}(field, {names.index(name)}, site, n_grid, dx)", "q")

    def callback(name, expr):
        chunks.append(f"def {name}(field, i, n_grid, dx, origin, params):\n")
        chunks.append(f"    # Residual: {expr.text}\n    site = i % n_grid\n    q = origin + site * dx\n    return {scalar_code(expr)}\n\n\n")

    bulk = []
    for a, f in enumerate(spec["fields"]):
        expr = parse(f["equation"])
        callback(f"_eq_{a}", expr)
        bulk.append(expr.dependencies())
        for side in ("left", "right"):
            callback(f"_bc_{a}_{side}", parse(boundary_expression(f, side), max_order=1))
    for k, row in enumerate(spec["extra_rows"]):
        callback(f"_extra_{k}", parse(row["expression"], max_order=1))

    chunks.append("def build_bulk_functions(config, params):\n    _, n_grid, dx, origin = _geometry(config)\n    return {\n")
    for a in range(len(names)):
        chunks.append(f"        {a}: partial(_eq_{a}, n_grid=n_grid, dx=dx, origin=origin, params=params),\n")
    chunks.append("    }\n\n\n")
    chunks.append("def build_boundary_spec(config, params):\n    _, n_grid, dx, origin = _geometry(config)\n    return {\n")
    for a, f in enumerate(spec["fields"]):
        chunks.append(f"        {a}: {{\n")
        for side in ("left", "right"):
            deps = parse(boundary_expression(f, side), max_order=1).dependencies(side)
            pairs = ", ".join(f"({b}, {off if side == 'left' else f'n_grid - 1 + ({off})'})" for b, off in deps)
            chunks.append(f"            '{side}': partial(_bc_{a}_{side}, n_grid=n_grid, dx=dx, origin=origin, params=params),\n")
            chunks.append(f"            '{side}_deps': [{pairs}],\n")
        chunks.append("        },\n")
    chunks.append("    }\n\n\n")
    deps = {"bulk_by_field": {a: {"deps": ds} for a, ds in enumerate(bulk)}}
    chunks.append("def build_dependency_spec(config, params):\n    # Structural references, including couplings whose derivative is zero at a seed.\n")
    chunks.append("    return " + pprint.pformat(deps, width=90).replace("\n", "\n    ") + "\n\n\n")
    chunks.append("def build_extra_rows(config, params):\n    _, n_grid, dx, origin = _geometry(config)\n    rows = []\n")
    for k, row in enumerate(spec["extra_rows"]):
        site = spec["grid"]["n"] // 2 if row["site"] == "centre" else int(row["site"])
        site_code = "n_grid // 2" if row["site"] == "centre" else str(site)
        side = "left" if site == 0 else "right" if site == spec["grid"]["n"] - 1 else None
        deps = parse(row["expression"], max_order=1).dependencies(side)
        pairs = ", ".join(f"({a}, site + ({offset}))" for a, offset in deps)
        chunks.append(f"    site = {site_code}\n    rows.append({{'field': {names.index(row['field'])}, 'site': site,\n")
        chunks.append(f"                 'func': partial(_extra_{k}, n_grid=n_grid, dx=dx, origin=origin, params=params),\n                 'deps': [{pairs}]}})\n")
    chunks.append("    return rows\n\n\n")
    chunks.append("def build_initial_guess(config, params):\n    grid, _, _, _ = _geometry(config)\n    indexer = build_indexer(grid, config['n_fields'])\n    q = grid.build_coordinates()[0]\n    field = np.empty(indexer.n_unknowns, dtype=float)\n")
    for a, f in enumerate(spec["fields"]):
        expr = Expression(f["guess"], parameters=params, coordinate=q)
        chunks.append(f"    # {f['name']}: {expr.text}\n    field[indexer.field_slice({a})] = {expr.emit(None, None, 'q')}\n")
    chunks.append("    if not np.all(np.isfinite(field)):\n        raise ValueError('Initial guess contains non-finite values.')\n    return field\n\n\n")

    def vector_code(expr, select):
        return expr.emit(lambda name: f"field[{names.index(name)}]{select}",
                         lambda name, order: f"_profile_d{order}(field[{names.index(name)}], dx){select}", "q")

    obs = spec["observable"]
    if obs["enabled"]:
        interior = obs["quadrature"] == "interior_sum"
        select = "[1:-1]" if interior else ""
        chunks.append("def compute_observable(field, config, params):\n    grid, _, dx, _ = _geometry(config)\n")
        chunks.append(f"    q = grid.build_coordinates()[0]{select}\n")
        chunks.append(f"    density = {vector_code(parse(obs['density'], max_order=1), select)}\n")
        weight = Expression(obs["weight"], parameters=params, coordinate=q).emit(None, None, "q")
        pref = Expression(obs["prefactor"], parameters=params).emit(None, None)
        chunks.append(f"    integrand = np.broadcast_to(np.asarray(density * ({weight}), dtype=float), q.shape)\n")
        chunks.append("    if not np.all(np.isfinite(integrand)):\n        raise ValueError('Observable has non-finite values; check its endpoint treatment.')\n")
        if interior:
            chunks.append(f"    return float(({pref}) * dx * np.sum(integrand))\n\n\n")
        else:
            chunks.append(f"    integrate = getattr(np, 'trapezoid', None) or np.trapz\n    return float(({pref}) * integrate(integrand, q))\n\n\n")
    if spec["plots"]:
        chunks.append("def make_solution_plot(field, config, params):\n    grid, _, dx, _ = _geometry(config)\n    q = grid.build_coordinates()[0]\n    fig, ax = plt.subplots()\n")
        for curve in spec["plots"]:
            code = vector_code(parse(curve["expression"]), "")
            chunks.append(f"    values = np.broadcast_to(np.asarray({code}, dtype=float), q.shape)\n    ax.plot(q, values, label={curve['label']!r})\n")
        chunks.append(f"    ax.set(xlabel={q!r}, ylabel='field / derived quantity')\n    ax.set_xlim(q[0], q[-1])\n    ax.legend()\n    fig.tight_layout()\n    return fig\n\n\n")
    chunks.append("def main():\n    from run.run_model import run_single\n    from run.run_scan import run_scan\n    model = sys.modules[__name__]\n    config = get_model_config()\n    if config['variation_mode'] == 'none':\n        run_single(model, config, finalize_parameters(config['parameters']))\n    else:\n        run_scan(model)\n\n\nif __name__ == '__main__':\n    main()\n")
    body = "".join(chunks)
    metadata = json.dumps(original, sort_keys=True, ensure_ascii=True)
    metadata_digest = hashlib.sha256(metadata.encode()).hexdigest()
    digest = hashlib.sha256(body.encode()).hexdigest()
    return (f"#!/usr/bin/env python3\n# Generated by KinkSol Model Builder {__version__}.\n"
            "# This model uses the existing KinkSol core, NumPy and Matplotlib only.\n"
            f"BUILDER_SPEC_JSON = {metadata!r}\nBUILDER_SPEC_SHA256 = {metadata_digest!r}\nBUILDER_BODY_SHA256 = {digest!r}\n\n" + BODY_MARKER + body)


def _project_header(spec):
    if not isinstance(spec, dict) or spec.get("version") != SCHEMA_VERSION or "fields" not in spec:
        raise ExpressionError("This is not a supported builder project.")
    check_equation_mode(spec)
    return spec


def read_builder_file(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return _project_header(json.loads(text))
    try:
        module = ast.parse(text)
        constants = {}
        for node in module.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("BUILDER_SPEC_JSON", "BUILDER_SPEC_SHA256", "BUILDER_BODY_SHA256"):
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
        body = text.split(BODY_MARKER, 1)[1]
        digest = hashlib.sha256(body.encode()).hexdigest()
        if digest != constants["BUILDER_BODY_SHA256"]:
            raise ModifiedModelError("The generated Python was edited. Opened in the source editor to preserve those changes.")
        if hashlib.sha256(constants["BUILDER_SPEC_JSON"].encode()).hexdigest() != constants["BUILDER_SPEC_SHA256"]:
            raise ModifiedModelError("The builder metadata was edited separately from the Python. Opened in the source editor to preserve the saved equations.")
        spec = json.loads(constants["BUILDER_SPEC_JSON"])
    except (KeyError, IndexError, SyntaxError, ValueError, TypeError, AttributeError) as exc:
        if isinstance(exc, ModifiedModelError):
            raise
        raise ModifiedModelError("This Python model has no intact builder metadata. Opened in the source editor.") from exc
    return _project_header(spec)
