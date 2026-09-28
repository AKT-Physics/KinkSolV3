# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:50:43 2026

@author: Adam
"""

# strategy/continuation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ContinuationOptions:
    """
    Policy controlling how continuation seeds are chosen during scans.
    """

    enabled: bool = True

    # If True, only converged solutions are allowed to update the continuation seed.
    # If False, the most recent field is always allowed to become the next seed.
    require_converged_previous: bool = True

    # If True, copy the previous field before storing it.
    # Keeping this True avoids aliasing bugs.
    copy_on_store: bool = True


@dataclass
class ContinuationState:
    """
    Mutable continuation state carried across a scan.
    """

    previous_field: np.ndarray | None = None
    previous_converged_field: np.ndarray | None = None
    point_index: int = 0


class ContinuationManager:
    """
    Encapsulates continuation behaviour for parameter scans.

    Workflow:
        - first point: build_initial_guess(...)
        - later points:
            * if continuation is enabled and a suitable previous field exists,
              use build_continuation_guess(...) if the model provides it
            * otherwise fall back to build_initial_guess(...)
    """

    def __init__(self, options: ContinuationOptions | None = None):
        self.options = options or ContinuationOptions()
        self.state = ContinuationState()

    def build_seed(
        self,
        model: Any,
        config: dict,
        params: dict,
    ) -> np.ndarray:
        """
        Build the field used to start the current solve.
        """
        previous_field = self._select_previous_field()

        if not self.options.enabled or previous_field is None:
            return self._build_initial_guess(model, config, params)

        if hasattr(model, "build_continuation_guess"):
            field = model.build_continuation_guess(config, params, previous_field)
            return self._as_real_1d_array(field)

        return self._as_real_1d_array(previous_field)

    def previous_seed(self) -> np.ndarray | None:
        """Return the previous solution to be adapted by the single-point runner.

        A missing seed means that the runner should build the analytic initial
        guess.  The runner calls the model's continuation callback once when a
        previous solution is available.
        """
        if not self.options.enabled:
            return None
        previous_field = self._select_previous_field()
        if previous_field is None:
            return None
        return np.array(previous_field, copy=True)

    def update(
        self,
        field: np.ndarray,
        converged: bool,
    ) -> None:
        """
        Update continuation state after a solve attempt.
        """
        field_to_store = self._prepare_store(field)

        self.state.previous_field = field_to_store

        if converged:
            self.state.previous_converged_field = field_to_store

        self.state.point_index += 1

    def reset(self) -> None:
        self.state = ContinuationState()

    def _select_previous_field(self) -> np.ndarray | None:
        if self.options.require_converged_previous:
            return self.state.previous_converged_field
        return self.state.previous_field

    def _build_initial_guess(
        self,
        model: Any,
        config: dict,
        params: dict,
    ) -> np.ndarray:
        if not hasattr(model, "build_initial_guess"):
            raise AttributeError(
                "Model must provide build_initial_guess(config, params)."
            )
        field = model.build_initial_guess(config, params)
        return self._as_real_1d_array(field)

    def _prepare_store(self, field: np.ndarray) -> np.ndarray:
        arr = self._as_real_1d_array(field)
        if self.options.copy_on_store:
            return np.array(arr, copy=True)
        return arr

    @staticmethod
    def _as_real_1d_array(field: Any) -> np.ndarray:
        arr = np.asarray(field)

        # Handles both float arrays and object arrays that may arise from dual-number
        # or mixed pipelines.
        if arr.dtype == object:
            arr = np.array(
                [float(v.real) if hasattr(v, "real") else float(v) for v in arr.reshape(-1)],
                dtype=float,
            )
        else:
            arr = np.asarray(arr, dtype=float).reshape(-1)

        return arr
