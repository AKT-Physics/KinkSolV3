"""Continuum Euler-Lagrange derivation for reduced first-derivative functionals."""
from __future__ import annotations

import sympy as sp

from builder.expressions import Expression, ExpressionError


def symbolic_context(spec):
    names = [f["name"] for f in spec["fields"]]
    params = [p["name"] for p in spec["parameters"]]
    q = "r" if spec["geometry"] == "radial" else "x"
    symbols = {name: sp.Symbol(name, real=True) for name in names + params + [q]}
    jets = {(name, order): sp.Symbol(f"__d{order}_{name}", real=True)
            for name in names for order in (1, 2)}
    return names, params, q, symbols, jets


def derive_equations(spec):
    names, params, q, symbols, jets = symbolic_context(spec)
    density = Expression(spec["functional"]["density"], names, params, q, max_order=1)
    weight = Expression(spec["functional"]["weight"], parameters=params, coordinate=q, max_order=0)
    integrand = density.to_sympy(symbols, jets) * weight.to_sympy(symbols, jets)
    equations, latex = [], []
    for name in names:
        momentum = sp.diff(integrand, jets[name, 1])
        total = sp.diff(momentum, symbols[q])
        for other in names:
            total += sp.diff(momentum, symbols[other]) * jets[other, 1]
            total += sp.diff(momentum, jets[other, 1]) * jets[other, 2]
        # Use this residual sign consistently, retaining the weight and coefficients.
        residual = sp.expand(total - sp.diff(integrand, symbols[name]))
        if residual == 0:
            raise ExpressionError(f"The functional gives no equation for '{name}'. Add a constraint or use direct equations.")
        printable = {jet: sp.Function("D" if order == 1 else "D2")(symbols[field])
                     for (field, order), jet in jets.items()}
        printable[sp.E] = sp.Symbol("e")
        text = sp.sstr(residual.xreplace(printable))
        # Check that symbolic differentiation produced only supported syntax.
        Expression(text, names, params, q)
        pretty = {symbols[field]: sp.Function(field)(symbols[q]) for field in names}
        pretty.update({jet: sp.diff(sp.Function(field)(symbols[q]), symbols[q], order)
                       for (field, order), jet in jets.items()})
        equations.append(text)
        latex.append(sp.latex(residual.xreplace(pretty)) + " = 0")
    return equations, latex


def equation_latex(spec):
    names, params, q, symbols, jets = symbolic_context(spec)
    pretty = {symbols[field]: sp.Function(field)(symbols[q]) for field in names}
    pretty.update({jet: sp.diff(sp.Function(field)(symbols[q]), symbols[q], order)
                   for (field, order), jet in jets.items()})
    return [sp.latex(Expression(f["equation"], names, params, q).to_sympy(symbols, jets).xreplace(pretty)) + " = 0"
            for f in spec["fields"]]
