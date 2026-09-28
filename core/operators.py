# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:24:03 2026

@author: Adam Thomasson
"""

# core/operators.py

from __future__ import annotations

import numpy as np

from .autodiff import deriv, row_derivatives, to_real_1d
from .problem import CompiledProblem
from .sparse import (
    SparseStructure,
    build_sparse_structure,
    csr_from_data,
    empty_jacobian_data,
)


def eval_residual(problem: CompiledProblem, u, out: np.ndarray | None = None) -> np.ndarray:
    """
    Evaluate the full residual vector F(u).

    Parameters
    ----------
    problem:
        Frozen compiled problem.
    u:
        Flat unknown vector.
    out:
        Optional output array of shape (n_unknowns,).

    Returns
    -------
    ndarray
        Residual vector of shape (n_unknowns,).
    """
    x = np.asarray(u).reshape(-1)
    if x.size != problem.n_unknowns:
        raise ValueError(
            f"u has size {x.size}, expected {problem.n_unknowns}."
        )

    if out is None:
        F = np.empty(problem.n_unknowns, dtype=float)
    else:
        if out.shape != (problem.n_unknowns,):
            raise ValueError(
                f"out has shape {out.shape}, expected ({problem.n_unknowns},)."
            )
        F = out

    func_list = problem.func_list
    func_index = problem.func_index

    for i in range(problem.n_unknowns):
        fi = func_list[func_index[i]]
        value = fi(x, i)
        F[i] = float(value.real) if hasattr(value, "real") else float(value)

    return F


def eval_jacobian_data(
    problem: CompiledProblem,
    structure: SparseStructure,
    u,
    out: np.ndarray | None = None,
    compatible: dict | None = None,
) -> np.ndarray:
    """
    Evaluate sparse Jacobian data in the fixed structural order.

    Parameters
    ----------
    problem:
        Frozen compiled problem.
    structure:
        Fixed sparse pattern compatible with problem.deriv_dir.
    u:
        Flat unknown vector.
    out:
        Optional preallocated output array of shape (nnz,).

    Returns
    -------
    ndarray
        Jacobian data array aligned with structure.rows / structure.cols.
    """
    x = np.asarray(u, dtype=object).reshape(-1)
    if x.size != problem.n_unknowns:
        raise ValueError(
            f"u has size {x.size}, expected {problem.n_unknowns}."
        )

    if out is None:
        data = empty_jacobian_data(structure)
    else:
        if out.shape != (structure.nnz,):
            raise ValueError(
                f"out has shape {out.shape}, expected ({structure.nnz},)."
            )
        data = out

    func_list = problem.func_list
    func_index = problem.func_index

    mode = problem.config.get("jacobian_autodiff", "auto")
    if mode not in ("auto", "scalar", "row"):
        raise ValueError("jacobian_autodiff must be 'auto', 'scalar', or 'row'.")
    # Vector dual arithmetic costs more on short rows. Very wide rows can
    # consume substantial memory for their dense derivative vectors.
    min_row_size = 7
    max_auto_row_size = 64
    bases = {}
    if compatible is None:
        compatible = {}

    for row in range(problem.n_unknowns):
        fi = func_list[func_index[row]]
        start = structure.row_ptr[row]
        stop = structure.row_ptr[row + 1]

        if (mode != "scalar" and stop - start >= min_row_size
                and (mode == "row" or stop - start <= max_auto_row_size)):
            n_deps = stop - start
            if n_deps not in bases:
                bases[n_deps] = np.eye(n_deps)
            data[start:stop] = row_derivatives(
                fi, x, structure.cols[start:stop], row,
                bases[n_deps], mode, compatible,
            )
        else:
            for slot in range(start, stop):
                col = int(structure.cols[slot])
                data[slot] = deriv(fi, x, col, row)

    return data


def eval_jacobian_csr(
    problem: CompiledProblem,
    u,
    structure: SparseStructure | None = None,
):
    """
    Evaluate the Jacobian as a CSR sparse matrix.
    """
    if structure is None:
        structure = build_sparse_structure(problem)

    data = eval_jacobian_data(problem, structure, u)
    return csr_from_data(
        structure=structure,
        data=data,
        shape=(problem.n_unknowns, problem.n_unknowns),
    )


class ProblemOperators:
    """
    Small convenience wrapper bundling compiled operators for one problem.

    This avoids rebuilding sparse structure repeatedly and gives the solver a
    clean object-oriented access point without introducing heavy abstraction.
    """

    def __init__(self, problem: CompiledProblem):
        self.problem = problem
        self.structure = build_sparse_structure(problem)
        self._row_compatible = {}

    def residual(self, u, out: np.ndarray | None = None) -> np.ndarray:
        return eval_residual(self.problem, u, out=out)

    def jacobian_data(self, u, out: np.ndarray | None = None) -> np.ndarray:
        return eval_jacobian_data(
            self.problem, self.structure, u, out=out,
            compatible=self._row_compatible,
        )

    def jacobian_csr(self, u):
        data = self.jacobian_data(u)
        return csr_from_data(
            structure=self.structure,
            data=data,
            shape=(self.problem.n_unknowns, self.problem.n_unknowns),
        )

    def to_real(self, u) -> np.ndarray:
        return to_real_1d(u)
