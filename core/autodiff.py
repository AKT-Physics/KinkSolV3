# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:14:09 2026

@author: Adam Thomasson
"""

# core/autodiff.py

from __future__ import annotations

import numpy as np


class Dual:
    """
    Minimal forward-mode dual number.

    A Dual represents:
        x = real + dual * eps
    with eps^2 = 0.

    This implementation is intentionally small and only supports the operations
    currently needed by the model equations and Jacobian assembly.
    """

    __slots__ = ("real", "dual")

    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    # ------------------------------------------------------------------
    # Basic arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        return Dual(self.real + other, self.dual)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        return Dual(self.real - other, self.dual)

    def __rsub__(self, other):
        if isinstance(other, Dual):
            return Dual(other.real - self.real, other.dual - self.dual)
        return Dual(other - self.real, -self.dual)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(
                self.real * other.real,
                self.real * other.dual + self.dual * other.real,
            )
        return Dual(self.real * other, self.dual * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Dual):
            real = self.real / other.real
            dual = self.dual / other.real - self.real * other.dual / (other.real ** 2)
            return Dual(real, dual)
        return Dual(self.real / other, self.dual / other)

    def __rtruediv__(self, other):
        if isinstance(other, Dual):
            real = other.real / self.real
            dual = other.dual / self.real - other.real * self.dual / (self.real ** 2)
            return Dual(real, dual)
        return Dual(other / self.real, -other * self.dual / (self.real ** 2))

    def __pow__(self, other):
        if isinstance(other, Dual):
            if np.any(self.real <= 0):
                raise ValueError("Dual base must be positive when exponent is Dual.")
            real = self.real ** other.real
            dual = real * (
                other.dual * np.log(self.real) + other.real * self.dual / self.real
            )
            return Dual(real, dual)

        real = self.real ** other
        dual = other * (self.real ** (other - 1)) * self.dual
        return Dual(real, dual)

    def __rpow__(self, other):
        if isinstance(other, Dual):
            return other.__pow__(self)

        if np.any(other <= 0):
            raise ValueError("Real base must be positive when exponent is Dual.")

        real = other ** self.real
        dual = real * np.log(other) * self.dual
        return Dual(real, dual)

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __pos__(self):
        return self

    # ------------------------------------------------------------------
    # Array indexing
    # ------------------------------------------------------------------

    def __getitem__(self, item):
        return Dual(self.real[item], self.dual[item])

    # ------------------------------------------------------------------
    # NumPy ufunc support
    # ------------------------------------------------------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        args = [x if isinstance(x, Dual) else Dual(x, 0.0) for x in inputs]

        if ufunc is np.sin:
            a = args[0]
            return Dual(np.sin(a.real), a.dual * np.cos(a.real))

        if ufunc is np.cos:
            a = args[0]
            return Dual(np.cos(a.real), -a.dual * np.sin(a.real))

        if ufunc is np.exp:
            a = args[0]
            real = np.exp(a.real)
            return Dual(real, a.dual * real)

        if ufunc is np.sqrt:
            a = args[0]
            if np.any(a.real <= 0):
                raise ValueError("sqrt(Dual) requires positive real part.")
            real = np.sqrt(a.real)
            return Dual(real, a.dual / (2.0 * real))

        if ufunc is np.tanh:
            a = args[0]
            real = np.tanh(a.real)
            return Dual(real, a.dual * (1.0 - real ** 2))

        if ufunc is np.log:
            a = args[0]
            if np.any(a.real <= 0):
                raise ValueError("log(Dual) requires strictly positive real part.")
            return Dual(np.log(a.real), a.dual / a.real)

        if ufunc is np.add:
            return args[0] + args[1]

        if ufunc is np.subtract:
            return args[0] - args[1]

        if ufunc is np.multiply:
            return args[0] * args[1]

        if ufunc is np.divide:
            return args[0] / args[1]

        if ufunc is np.power:
            return args[0] ** args[1]

        return NotImplemented


def deriv(func, x, direction: int, *args) -> float:
    """
    Differentiate a scalar callback `func(x, *args)` with respect to x[direction]
    using forward-mode automatic differentiation.

    Parameters
    ----------
    func:
        Function returning either a float-like value or a Dual.
    x:
        Flat unknown vector. This list/array is mutated temporarily in-place.
    direction:
        Flat index of the seeded variable.
    *args:
        Extra positional arguments forwarded to func.

    Returns
    -------
    float
        d func / d x[direction]
    """
    original = x[direction]
    x[direction] = Dual(original, 1.0)
    try:
        value = func(x, *args)
    finally:
        x[direction] = original

    if isinstance(value, Dual):
        return float(value.dual)
    return 0.0


def deriv_row(func, x, directions, row: int, basis: np.ndarray) -> np.ndarray:
    """Differentiate one residual with respect to all its declared columns.

    One call to the model callback propagates a vector of dual coefficients.
    The caller supplies a cached identity matrix to avoid allocating it for
    every row. A model callback must support vector dual coefficients as well
    as the ordinary scalar coefficients.
    """
    originals = [x[int(col)] for col in directions]
    try:
        for j, (col, original) in enumerate(zip(directions, originals)):
            x[int(col)] = Dual(original, basis[j])
        value = func(x, row)
    finally:
        for col, original in zip(directions, originals):
            x[int(col)] = original

    if not isinstance(value, Dual):
        return np.zeros(len(directions), dtype=float)

    derivatives = np.asarray(value.dual, dtype=float)
    if derivatives.shape != (len(directions),):
        raise ValueError(
            f"Row {row} returned dual coefficients of shape {derivatives.shape}; "
            f"expected ({len(directions)},). Use scalar Jacobian mode for "
            "callbacks without vector dual support."
        )
    return derivatives


def row_derivatives(func, x, directions, row, basis, mode, compatible):
    """Differentiate a row, checking vector support once per callback/width.

    In auto mode, an unsupported vector operation, unexpected derivative
    shape, or disagreement with scalar AD selects scalar AD for that callback.
    A callback can still opt out explicitly with ``jacobian_autodiff='scalar'``.
    """
    key = (id(func), len(directions))

    def scalar_derivatives():
        return np.fromiter(
            (deriv(func, x, int(col), row) for col in directions),
            dtype=float,
            count=len(directions),
        )

    if mode == "scalar" or (mode == "auto" and compatible.get(key) is False):
        return scalar_derivatives()

    if mode == "row" or (mode == "auto" and compatible.get(key) is True):
        # Also handle a callback whose support changes with the current row.
        try:
            return deriv_row(func, x, directions, row, basis)
        except (TypeError, ValueError, NotImplementedError):
            if mode == "row":
                raise
            scalar = scalar_derivatives()
            compatible[key] = False
            return scalar

    try:
        vector = deriv_row(func, x, directions, row, basis)
    except (TypeError, ValueError, NotImplementedError):
        scalar = scalar_derivatives()
        compatible[key] = False
        return scalar

    scalar = scalar_derivatives()
    if np.allclose(vector, scalar, rtol=1e-10, atol=1e-10, equal_nan=True):
        compatible[key] = True
        return vector

    compatible[key] = False
    return scalar


def to_real_1d(x) -> np.ndarray:
    """
    Convert a 1D iterable of float-like or Dual-like values to a flat float array.
    """
    arr = np.asarray(x).reshape(-1)
    if arr.dtype.kind in "biuf":
        return np.array(arr, dtype=float, copy=True)
    out = np.empty(arr.size, dtype=float)
    for i, v in enumerate(arr):
        out[i] = float(v.real) if hasattr(v, "real") else float(v)
    return out
