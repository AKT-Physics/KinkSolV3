# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:15:53 2026

@author: Adam Thomasson
"""

# core/sparse.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .problem import CompiledProblem


@dataclass(frozen=True)
class SparseStructure:
    """
    Frozen sparse Jacobian structure.

    Attributes
    ----------
    rows, cols:
        Flat COO row/column arrays describing the nonzero pattern.
    row_ptr:
        Prefix pointers into rows/cols for each residual row. The entries for
        row i live in [row_ptr[i], row_ptr[i+1]).
    nnz:
        Number of structural nonzeros.
    """
    rows: np.ndarray
    cols: np.ndarray
    row_ptr: np.ndarray
    nnz: int


def normalize_deriv_dir(deriv_dir: Sequence, n_unknowns: int) -> List[np.ndarray]:
    """
    Normalize the model-provided deriv_dir into a list of 1D integer arrays.

    Each entry deriv_dir[row] is converted to a unique, sorted array of column
    indices on which that row structurally depends.
    """
    if len(deriv_dir) != n_unknowns:
        raise ValueError(
            f"deriv_dir has length {len(deriv_dir)}, expected {n_unknowns}."
        )

    normalized: List[np.ndarray] = []

    for row, deps in enumerate(deriv_dir):
        if isinstance(deps, np.ndarray):
            arr = deps.reshape(-1).astype(int, copy=False)
        elif isinstance(deps, (list, tuple)):
            arr = np.asarray(deps, dtype=int).reshape(-1)
        else:
            arr = np.asarray([deps], dtype=int)

        if arr.size == 0:
            raise ValueError(f"deriv_dir[{row}] is empty.")

        if np.any(arr < 0) or np.any(arr >= n_unknowns):
            raise ValueError(f"deriv_dir[{row}] contains out-of-range indices.")

        arr = np.unique(arr)
        normalized.append(arr)

    return normalized


def build_sparse_structure_from_deriv_dir(
    deriv_dir: Sequence,
    n_unknowns: int,
) -> SparseStructure:
    """
    Construct a fixed sparse pattern from normalized row dependencies.
    """
    deps_list = normalize_deriv_dir(deriv_dir, n_unknowns)

    row_counts = np.fromiter((len(deps) for deps in deps_list), dtype=int, count=n_unknowns)
    row_ptr = np.empty(n_unknowns + 1, dtype=int)
    row_ptr[0] = 0
    np.cumsum(row_counts, out=row_ptr[1:])

    nnz = int(row_ptr[-1])

    rows = np.empty(nnz, dtype=int)
    cols = np.empty(nnz, dtype=int)

    cursor = 0
    for row, deps in enumerate(deps_list):
        k = len(deps)
        rows[cursor:cursor + k] = row
        cols[cursor:cursor + k] = deps
        cursor += k

    return SparseStructure(
        rows=rows,
        cols=cols,
        row_ptr=row_ptr,
        nnz=nnz,
    )


def build_sparse_structure(problem: CompiledProblem) -> SparseStructure:
    """
    Build the frozen structural sparsity pattern for a compiled problem.
    """
    return build_sparse_structure_from_deriv_dir(
        deriv_dir=problem.deriv_dir,
        n_unknowns=problem.n_unknowns,
    )


def empty_jacobian_data(structure: SparseStructure) -> np.ndarray:
    """
    Allocate a data array compatible with the sparse structure.
    """
    return np.zeros(structure.nnz, dtype=float)


def csr_from_data(structure: SparseStructure, data: np.ndarray, shape: tuple[int, int]) -> csr_matrix:
    """
    Build a CSR sparse matrix from the fixed structure and a data array.
    """
    if data.shape != (structure.nnz,):
        raise ValueError(
            f"data has shape {data.shape}, expected ({structure.nnz},)."
        )

    # normalize_deriv_dir has already sorted and deduplicated each row.
    # Rebuilding a COO matrix would sort and copy this fixed pattern again.
    return csr_matrix(
        (data, structure.cols, structure.row_ptr), shape=shape, copy=False
    )
