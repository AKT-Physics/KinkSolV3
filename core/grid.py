# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:53:41 2026

@author: Adam Thomasson
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Grid:
    """
    Immutable grid metadata.

    This object is deliberately lightweight. It stores only the information
    required to describe the computational grid, plus a few helper methods for
    coordinates and shape consistency.
    """
    n_dims: int
    geometry: str
    coord_names: Tuple[str, ...]
    grid_shape: Tuple[int, ...]
    grid_spacing: Tuple[float, ...]

    def __post_init__(self) -> None:
        if self.n_dims < 1:
            raise ValueError("n_dims must be >= 1.")

        if len(self.coord_names) != self.n_dims:
            raise ValueError("coord_names must have length n_dims.")

        if len(self.grid_shape) != self.n_dims:
            raise ValueError("grid_shape must have length n_dims.")

        if len(self.grid_spacing) != self.n_dims:
            raise ValueError("grid_spacing must have length n_dims.")

        if any(n <= 0 for n in self.grid_shape):
            raise ValueError("All entries of grid_shape must be positive.")

        if any(dx <= 0.0 for dx in self.grid_spacing):
            raise ValueError("All entries of grid_spacing must be positive.")

        if self.geometry not in ("cartesian", "radial"):
            raise ValueError("geometry must currently be 'cartesian' or 'radial'.")

    @property
    def n_grid(self) -> int:
        """Total number of spatial lattice sites."""
        return int(np.prod(self.grid_shape))

    @property
    def is_1d(self) -> bool:
        return self.n_dims == 1

    @property
    def is_2d(self) -> bool:
        return self.n_dims == 2

    @property
    def is_3d(self) -> bool:
        return self.n_dims == 3

    def build_coordinates(self) -> Tuple[np.ndarray, ...]:
        """
        Build centered coordinate arrays.

        Current first-pass convention:
        - Cartesian grids are centered around zero in each direction.
        - Radial grids currently return a uniform coordinate starting from zero.

        This can be extended later if you want more geometry-specific handling.
        """
        if self.geometry == "cartesian":
            coords = []
            for n, dx in zip(self.grid_shape, self.grid_spacing):
                x = (np.arange(n) - 0.5 * (n - 1)) * dx
                coords.append(x)
            return tuple(coords)

        if self.geometry == "radial":
            if self.n_dims != 1:
                raise ValueError("First-pass radial support is only implemented for 1D.")
            n_r = self.grid_shape[0]
            d_r = self.grid_spacing[0]
            r = np.arange(n_r) * d_r
            return (r,)

        raise ValueError(f"Unsupported geometry: {self.geometry}")


def build_grid_from_config(config: dict) -> Grid:
    """
    Construct a Grid object from a model config dictionary.
    """
    return Grid(
        n_dims=int(config["n_dims"]),
        geometry=str(config["geometry"]),
        coord_names=tuple(config["coord_names"]),
        grid_shape=tuple(config["grid_shape"]),
        grid_spacing=tuple(config["grid_spacing"]),
    )