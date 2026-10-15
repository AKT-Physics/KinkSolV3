"""A small, explicit mathematical grammar shared by all builder inputs.

User input is walked as an AST; it is never passed to eval or sympify.
Derivatives refer to named real fields. Arbitrary Python is a separate editor.
"""
from __future__ import annotations

import ast
import keyword
import operator

import numpy as np


FUNCTIONS = ("sin", "cos", "exp", "tanh", "log", "sqrt", "sinh", "cosh", "sech")
CONSTANTS = {"pi": np.pi, "e": np.e}
RESERVED = set(FUNCTIONS) | set(CONSTANTS) | {
    "D", "D2", "x", "r", "field", "i", "site", "params", "dx", "n_grid",
    "origin", "grid", "config", "indexer", "np", "s", "values",
}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}


class ExpressionError(ValueError):
    pass


def check_name(name):
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ExpressionError(f"'{name}' is not a valid symbolic name.")
    if name.startswith("_") or name in RESERVED:
        raise ExpressionError(f"'{name}' is reserved. Choose another name.")


class Expression:
    def __init__(self, text, fields=(), parameters=(), coordinate=None, max_order=2):
        self.text = str(text).strip().replace("^", "**")
        self.fields = tuple(fields)
        self.parameters = tuple(parameters)
        self.coordinate = coordinate
        self.refs = set()
        self.parameter_refs = set()
        if not self.text:
            raise ExpressionError("Enter an expression.")
        try:
            self.tree = ast.parse(self.text, mode="eval").body
        except SyntaxError as exc:
            raise ExpressionError(f"Invalid expression: {exc.msg}") from exc
        self._check(self.tree, max_order)

    def _check(self, node, max_order):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or not np.isfinite(node.value):
                raise ExpressionError("Only finite real numeric constants are allowed.")
        elif isinstance(node, ast.Name):
            name = node.id
            if name in self.fields:
                self.refs.add((name, 0))
            elif name in self.parameters:
                self.parameter_refs.add(name)
            elif name not in CONSTANTS and name != self.coordinate:
                raise ExpressionError(f"Unknown symbol '{name}'.")
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            self._check(node.left, max_order)
            self._check(node.right, max_order)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._check(node.operand, max_order)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if len(node.args) != 1 or node.keywords:
                raise ExpressionError(f"{name} takes one argument.")
            if name in ("D", "D2"):
                order = 1 if name == "D" else 2
                arg = node.args[0]
                if not isinstance(arg, ast.Name) or arg.id not in self.fields:
                    raise ExpressionError(f"Use {name}(field_name), with a declared field.")
                if order > max_order:
                    raise ExpressionError(f"Derivatives of order {order} are not permitted here.")
                self.refs.add((arg.id, order))
            elif name in FUNCTIONS:
                self._check(node.args[0], max_order)
            else:
                raise ExpressionError(f"Unsupported function '{name}'.")
        else:
            raise ExpressionError("Use numbers, named symbols, arithmetic and supported functions.")

    def emit(self, value, derivative, coordinate=None, parameter=None):
        """Emit Python using caller-provided field/derivative references."""
        parameter = parameter or (lambda name: f"params[{name!r}]")

        def walk(node):
            if isinstance(node, ast.Constant):
                return repr(node.value)
            if isinstance(node, ast.Name):
                name = node.id
                if name in self.fields:
                    return value(name)
                if name in self.parameters:
                    return parameter(name)
                if name in CONSTANTS:
                    return f"np.{name}"
                return coordinate or "q"
            if isinstance(node, ast.BinOp):
                op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}[type(node.op)]
                return f"({walk(node.left)} {op} {walk(node.right)})"
            if isinstance(node, ast.UnaryOp):
                return f"({'-' if isinstance(node.op, ast.USub) else '+'}{walk(node.operand)})"
            name = node.func.id
            if name in ("D", "D2"):
                return derivative(node.args[0].id, 1 if name == "D" else 2)
            arg = walk(node.args[0])
            # These rewrites also work with KinkSol's Dual objects.
            if name == "sinh":
                return f"((np.exp({arg}) - np.exp(-({arg}))) / 2.0)"
            if name == "cosh":
                return f"((np.exp({arg}) + np.exp(-({arg}))) / 2.0)"
            if name == "sech":
                return f"(2.0 / (np.exp({arg}) + np.exp(-({arg}))))"
            return f"np.{name}({arg})"
        return walk(self.tree)

    def evaluate(self, values=None, derivatives=None, parameters=None, coordinate=None):
        values, derivatives, parameters = values or {}, derivatives or {}, parameters or {}
        funcs = {name: getattr(np, name) for name in FUNCTIONS if name != "sech"}
        funcs["sech"] = lambda arg: 1.0 / np.cosh(arg)

        def walk(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                if node.id in self.fields:
                    return values[node.id]
                if node.id in self.parameters:
                    return parameters[node.id]
                return CONSTANTS[node.id] if node.id in CONSTANTS else coordinate
            if isinstance(node, ast.BinOp):
                return OPS[type(node.op)](walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp):
                return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
            name = node.func.id
            if name in ("D", "D2"):
                return derivatives[(node.args[0].id, 1 if name == "D" else 2)]
            return funcs[name](walk(node.args[0]))
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            try:
                return walk(self.tree)
            except (ArithmeticError, ValueError, TypeError) as exc:
                raise ExpressionError(f"Cannot evaluate '{self.text}': {exc}") from exc

    def to_sympy(self, symbols, jets):
        import sympy as sp
        funcs = {name: getattr(sp, name) for name in FUNCTIONS}

        def walk(node):
            if isinstance(node, ast.Constant):
                return sp.Rational(str(node.value))
            if isinstance(node, ast.Name):
                return {"pi": sp.pi, "e": sp.E}.get(node.id, symbols.get(node.id))
            if isinstance(node, ast.BinOp):
                return OPS[type(node.op)](walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp):
                return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
            name = node.func.id
            if name in ("D", "D2"):
                return jets[(node.args[0].id, 1 if name == "D" else 2)]
            return funcs[name](walk(node.args[0]))
        return walk(self.tree)

    def dependencies(self, boundary=None):
        deps = set()
        for name, order in self.refs:
            offsets = (0,) if order == 0 else (-1, 1) if order == 1 else (-1, 0, 1)
            if boundary == "left" and order:
                offsets = (0, 1) if order == 1 else (0, 1, 2)
            if boundary == "right" and order:
                offsets = (-1, 0) if order == 1 else (-2, -1, 0)
            deps.update((self.fields.index(name), offset) for offset in offsets)
        return sorted(deps)


def profile_values(expression, spec, params, fields, coordinates, interior=False):
    names = [entry["name"] for entry in spec["fields"]]
    dx = spec["grid"]["spacing"]
    select = slice(1, -1) if interior else slice(None)
    vals, derivs = {}, {}
    for name, arr in zip(names, fields):
        vals[name] = arr[select]
        derivs[name, 1] = np.gradient(arr, dx)[select]
        d2 = np.empty_like(arr)
        d2[1:-1] = (arr[2:] - 2 * arr[1:-1] + arr[:-2]) / dx**2
        d2[0], d2[-1] = d2[1], d2[-2]
        derivs[name, 2] = d2[select]
    q = coordinates[select]
    parser = Expression(expression, names, params, "r" if spec["geometry"] == "radial" else "x")
    value = parser.evaluate(vals, derivs, params, q)
    return np.broadcast_to(np.asarray(value, dtype=float), q.shape)
