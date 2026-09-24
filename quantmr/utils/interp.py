from __future__ import annotations
from functools import cached_property
import numpy as np
from scipy.interpolate import interp1d


class Interp:

    def __init__(
        self,
        x: list | np.ndarray,
        y: list | np.ndarray,
        **interp1d_kwargs,
    ):
        self._x = x
        self._y = y
        self._interp1d_kwargs = interp1d_kwargs

    @cached_property
    def interpolator(self) -> interp1d:
        x = np.asarray(self._x, dtype=np.float64)
        y = np.asarray(self._y, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"x must be 1-d, got {x.ndim}-d")
        if y.ndim not in (1, 2):
            raise ValueError(f"y must be 1-d or 2-d, got {y.ndim}-d")
        if y.ndim == 1:
            if len(y) != len(x):
                raise ValueError(f"len(y)={len(y)} != len(x)={len(x)}")
            y = y[None, :]
        else:
            if y.shape[1] != len(x):
                raise ValueError(f"y.shape[1]={y.shape[1]} != len(x)={len(x)}")
        valid = (~np.isnan(x)) & (x > 0.0) & np.any(~np.isnan(y), axis=0)
        x, y = x[valid], y[:, valid]
        _, unique_idx = np.unique(x, return_index=True)
        x, y = x[unique_idx], y[:, unique_idx]
        return interp1d(x, y, **self._interp1d_kwargs)

    def interpolate(self, x: float | list | np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.squeeze(self.interpolator(x))
