from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
import numpy as np


@dataclass
class LatentState:

    t: object = field(default=None)
    a: object = field(default=None)
    rho: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "t",
            np.asarray([np.nan], dtype=np.float64)
            if self.t is None
            else np.asarray(self.t, dtype=np.float64).ravel(),
        )
        object.__setattr__(
            self,
            "a",
            np.asarray([np.nan], dtype=np.float64)
            if self.a is None
            else np.asarray(self.a, dtype=np.float64).ravel(),
        )
        n = self.a.size
        rho = (
            np.eye(n, dtype=np.float64)
            if self.rho is None
            else np.asarray(self.rho, dtype=np.float64)
        )
        object.__setattr__(self, "rho", rho)

    __hash__ = None

    @property
    def nfactors(self) -> int:
        return self.a.size

    def _corr_block(
        self,
        t: float | np.float64,
        s: float | np.float64,
    ) -> np.ndarray:
        a = np.where(np.isnan(self.a), 0.0, self.a)
        a_i, a_j = a[None, :], a[:, None]
        term1 = np.exp(-(a_i * t + a_j * s))
        term2 = np.divide(
            np.exp((a_i + a_j) * np.minimum(t, s)) - 1.0,
            a_i + a_j,
            where=~np.isclose(a_i + a_j, 0.0),
            out=np.full_like(a_i + a_j, np.minimum(t, s)),
        )
        term3_i = np.divide(
            2.0 * a_i,
            1 - np.exp(-2.0 * a_i * t),
            where=~np.isclose(a_i, 0.0),
            out=np.full_like(a_i, 1.0 / t),
        )
        term3_j = np.divide(
            2.0 * a_j,
            1 - np.exp(-2.0 * a_j * s),
            where=~np.isclose(a_j, 0.0),
            out=np.full_like(a_j, 1.0 / s),
        )
        term3 = np.sqrt(term3_i * term3_j)
        return self.rho * term1 * term2 * term3

    @cached_property
    def corr_matrix(self) -> np.ndarray:
        t = self.t
        k = self.nfactors
        size = t.size * k
        matrix = np.zeros((size, size), dtype=np.float64)
        for i, ti in enumerate(t):
            for j, tj in enumerate(t):
                block = self._corr_block(ti, tj)
                matrix[
                    i * k : (i + 1) * k,
                    j * k : (j + 1) * k,
                ] = block
        return matrix

    def generate_epsilon_dict(
        self,
        n_samples: int = 10_000,
        state: int | np.random.Generator | None = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        t = self.t
        k = self.nfactors
        n_t = t.size
        rng = (
            state
            if isinstance(state, np.random.Generator)
            else np.random.default_rng(state)
        )
        matL = np.linalg.cholesky(self.corr_matrix)
        total = n_t * k
        matZ = rng.standard_normal((n_samples, total))
        correlated = matZ @ matL.T
        return {
            (i, j): correlated[:, i * k + j]
            for i in range(n_t)
            for j in range(k)
        }
