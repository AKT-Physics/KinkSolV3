# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:56:10 2026

@author: Adam Thomasson
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .grid import Grid


@dataclass(frozen=True)
class Indexer:
    """
    Flat indexing helper for stacked field storage.

    Storage convention:
        u = [field_0(all sites), field_1(all sites), ..., field_{n_fields-1}(all sites)]

    so the unknown for field a at flattened spatial site s is

        idx = a * n_grid + s
    """
    n_fields: int
    n_grid: int
    grid_shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_fields <= 0:
            raise ValueError("n_fields must be positive.")
        if self.n_grid <= 0:
            raise ValueError("n_grid must be positive.")
        if len(self.grid_shape) == 0:
            raise ValueError("grid_shape must be non-empty.")

    @property
    def n_unknowns(self) -> int:
        return self.n_fields * self.n_grid

    def field_index(self, field: int, site: int) -> int:
        """
        Return the flat unknown index for (field, flattened spatial site).
        """
        if not (0 <= field < self.n_fields):
            raise IndexError(f"Field index {field} out of range [0, {self.n_fields}).")
        if not (0 <= site < self.n_grid):
            raise IndexError(f"Site index {site} out of range [0, {self.n_grid}).")
        return field * self.n_grid + site

    def field_slice(self, field: int) -> slice:
        """
        Slice selecting the full block for a given field.
        """
        start = self.field_index(field, 0)
        stop = start + self.n_grid
        return slice(start, stop)

    def split_fields(self, u: np.ndarray) -> np.ndarray:
        """
        Reshape a flat unknown vector to shape (n_fields, n_grid).
        """
        arr = np.asarray(u)
        if arr.size != self.n_unknowns:
            raise ValueError(
                f"Input size {arr.size} does not match n_unknowns={self.n_unknowns}."
            )
        return arr.reshape((self.n_fields, self.n_grid))

    def flatten_fields(self, field_array: np.ndarray) -> np.ndarray:
        """
        Flatten an array of shape (n_fields, n_grid) back to 1D.
        """
        arr = np.asarray(field_array)
        expected_shape = (self.n_fields, self.n_grid)
        if arr.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {arr.shape}.")
        return arr.reshape(-1)

    # ------------------------------------------------------------------
    # Spatial flatten / unflatten helpers
    # ------------------------------------------------------------------

    def flatten_site(self, *coords: int) -> int:
        """
        Flatten a spatial multi-index to a single site index.

        Examples:
            1D: flatten_site(i)
            2D: flatten_site(ix, iy)
            3D: flatten_site(ix, iy, iz)

        Row-major convention:
            2D: site = iy * nx + ix
            3D: site = iz * (ny * nx) + iy * nx + ix
        """
        if len(coords) != len(self.grid_shape):
            raise ValueError("Wrong number of coordinates for grid_shape.")

        if len(self.grid_shape) == 1:
            i = coords[0]
            nx = self.grid_shape[0]
            if not (0 <= i < nx):
                raise IndexError("1D site index out of range.")
            return i

        if len(self.grid_shape) == 2:
            ix, iy = coords
            nx, ny = self.grid_shape
            if not (0 <= ix < nx and 0 <= iy < ny):
                raise IndexError("2D site index out of range.")
            return iy * nx + ix

        if len(self.grid_shape) == 3:
            ix, iy, iz = coords
            nx, ny, nz = self.grid_shape
            if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                raise IndexError("3D site index out of range.")
            return iz * (nx * ny) + iy * nx + ix

        raise NotImplementedError("Only 1D/2D/3D flattening is supported.")

    def unflatten_site(self, site: int) -> Tuple[int, ...]:
        """
        Inverse of flatten_site().
        """
        if not (0 <= site < self.n_grid):
            raise IndexError(f"Site index {site} out of range [0, {self.n_grid}).")

        if len(self.grid_shape) == 1:
            return (site,)

        if len(self.grid_shape) == 2:
            nx, ny = self.grid_shape
            ix = site % nx
            iy = site // nx
            return (ix, iy)

        if len(self.grid_shape) == 3:
            nx, ny, nz = self.grid_shape
            plane = nx * ny
            iz = site // plane
            rem = site % plane
            iy = rem // nx
            ix = rem % nx
            return (ix, iy, iz)

        raise NotImplementedError("Only 1D/2D/3D unflattening is supported.")


def build_indexer(grid: Grid, n_fields: int) -> Indexer:
    """
    Construct an Indexer from a Grid and field count.
    """
    return Indexer(
        n_fields=int(n_fields),
        n_grid=grid.n_grid,
        grid_shape=tuple(grid.grid_shape),
    )