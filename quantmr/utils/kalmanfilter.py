from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def _get(arr: np.ndarray, t: int, ndim_const: int) -> np.ndarray:
    if arr.ndim > ndim_const:
        return arr[t]
    return arr


def _update_scalar(
    x_p: np.ndarray,
    P_p: np.ndarray,
    y_t: np.ndarray,
    H_t: np.ndarray,
    d_t: np.ndarray,
    R_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    H = H_t.ravel()
    p = P_p[0, 0]
    v = y_t - d_t - H * x_p[0]

    R_diag = np.diag(R_t)
    R_inv_v = v / R_diag
    R_inv_H = H / R_diag
    HtRinvH = H @ R_inv_H
    HtRinvv = H @ R_inv_v

    denom = 1.0 / p + HtRinvH
    if denom <= 0:
        return x_p, P_p, -1e15

    log_det_S = np.sum(np.log(R_diag)) + np.log(p * denom)
    vtSinvv = v @ R_inv_v - p * HtRinvv**2 / (1.0 + p * HtRinvH)

    m = len(v)
    ll = -0.5 * (m * np.log(2.0 * np.pi) + log_det_S + vtSinvv)

    Kv = HtRinvv / denom
    KSKt = HtRinvH / denom * p

    x_u = np.array([x_p[0] + Kv])
    P_u = np.array([[p - KSKt]])
    return x_u, P_u, ll


def _update_scalar_partial(
    x_p: np.ndarray,
    P_p: np.ndarray,
    y_t: np.ndarray,
    H_t: np.ndarray,
    d_t: np.ndarray,
    R_t: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    H_full = H_t.ravel()
    idx = np.where(valid)[0]
    H = H_full[idx]
    y_v = y_t[idx]
    d_v = d_t[idx]
    R_diag = np.diag(R_t)[idx]
    p = P_p[0, 0]

    v = y_v - d_v - H * x_p[0]
    R_inv_v = v / R_diag
    R_inv_H = H / R_diag
    HtRinvH = H @ R_inv_H
    HtRinvv = H @ R_inv_v

    denom = 1.0 / p + HtRinvH
    if denom <= 0:
        v_full = np.full_like(y_t, np.nan)
        return x_p, P_p, -1e15, v_full

    log_det_S = np.sum(np.log(R_diag)) + np.log(p * denom)
    vtSinvv = v @ R_inv_v - p * HtRinvv**2 / (1.0 + p * HtRinvH)

    m = len(v)
    ll = -0.5 * (m * np.log(2.0 * np.pi) + log_det_S + vtSinvv)

    Kv = HtRinvv / denom
    KSKt = HtRinvH / denom * p

    x_u = np.array([x_p[0] + Kv])
    P_u = np.array([[p - KSKt]])

    v_full = np.full_like(y_t, np.nan)
    v_full[idx] = v
    return x_u, P_u, ll, v_full


def _update_general(
    x_p: np.ndarray,
    P_p: np.ndarray,
    y_t: np.ndarray,
    H_t: np.ndarray,
    d_t: np.ndarray,
    R_t: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    idx = np.where(valid)[0]
    H_v = H_t[idx]
    y_v = y_t[idx]
    d_v = d_t[idx]
    R_v = R_t[np.ix_(idx, idx)]

    v = y_v - d_v - H_v @ x_p
    S = H_v @ P_p @ H_v.T + R_v

    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        v_full = np.full_like(y_t, np.nan)
        return x_p, P_p, -1e15, v_full

    log_det_S = 2.0 * np.sum(np.log(np.diag(L)))
    alpha = np.linalg.solve(L, v)
    vtSinvv = alpha @ alpha

    m = len(v)
    ll = -0.5 * (m * np.log(2.0 * np.pi) + log_det_S + vtSinvv)

    K = P_p @ H_v.T @ np.linalg.inv(S)
    x_u = x_p + K @ v
    P_u = P_p - K @ S @ K.T
    P_u = 0.5 * (P_u + P_u.T)

    v_full = np.full_like(y_t, np.nan)
    v_full[idx] = v
    return x_u, P_u, ll, v_full


def kf_filter(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    y: np.ndarray,
    scalar: bool,
    n: int,
    m: int,
) -> dict:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    T = y.shape[0]

    x_filt = np.empty((T, n), dtype=np.float64)
    P_filt = np.empty((T, n, n), dtype=np.float64)
    x_pred_next = np.empty((T, n), dtype=np.float64)
    P_pred_next = np.empty((T, n, n), dtype=np.float64)
    F_used = np.empty((T, n, n), dtype=np.float64)
    v = np.full((T, m), np.nan, dtype=np.float64)

    loglik = 0.0
    x_p = x0.copy()
    P_p = P0.copy()

    for t in range(T):
        F_t = _get(F, t, 2)
        H_t = _get(H, t, 2)
        Q_t = _get(Q, t, 2)
        R_t = _get(R, t, 2)
        c_t = _get(c, t, 1)
        d_t = _get(d, t, 1)

        valid = ~np.isnan(y[t])
        n_valid = int(valid.sum())

        if n_valid == 0:
            x_filt[t] = x_p
            P_filt[t] = P_p
        elif scalar and n_valid == m:
            x_u, P_u, ll = _update_scalar(x_p, P_p, y[t], H_t, d_t, R_t)
            x_filt[t] = x_u
            P_filt[t] = P_u
            v[t] = y[t] - d_t - H_t.ravel() * x_p[0]
            loglik += ll
        elif scalar:
            x_u, P_u, ll, v_full = _update_scalar_partial(
                x_p, P_p, y[t], H_t, d_t, R_t, valid
            )
            x_filt[t] = x_u
            P_filt[t] = P_u
            v[t] = v_full
            loglik += ll
        else:
            x_u, P_u, ll, v_full = _update_general(
                x_p, P_p, y[t], H_t, d_t, R_t, valid
            )
            x_filt[t] = x_u
            P_filt[t] = P_u
            v[t] = v_full
            loglik += ll

        x_p = F_t @ x_filt[t] + c_t
        P_p = F_t @ P_filt[t] @ F_t.T + Q_t
        x_pred_next[t] = x_p
        P_pred_next[t] = P_p
        F_used[t] = F_t

    return {
        "x_filt": x_filt,
        "P_filt": P_filt,
        "x_pred_next": x_pred_next,
        "P_pred_next": P_pred_next,
        "F_used": F_used,
        "v": v,
        "loglik": loglik,
    }


@dataclass
class KalmanFilter:

    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    c: np.ndarray | None = None
    d: np.ndarray | None = None
    x0: np.ndarray | None = None
    P0: np.ndarray | None = None

    _n: int = field(init=False, repr=False, default=0)
    _m: int = field(init=False, repr=False, default=0)
    _scalar: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self.F = np.atleast_2d(np.asarray(self.F, dtype=np.float64))
        self.H = np.atleast_2d(np.asarray(self.H, dtype=np.float64))
        self.Q = np.atleast_2d(np.asarray(self.Q, dtype=np.float64))
        self.R = np.atleast_2d(np.asarray(self.R, dtype=np.float64))
        self._n = self.F.shape[0] if self.F.ndim == 2 else self.F.shape[1]
        self._m = self.H.shape[0] if self.H.ndim == 2 else self.H.shape[1]
        self.c = (
            np.zeros(self._n, dtype=np.float64)
            if self.c is None
            else np.asarray(self.c, dtype=np.float64)
        )
        self.d = (
            np.zeros(self._m, dtype=np.float64)
            if self.d is None
            else np.asarray(self.d, dtype=np.float64)
        )
        self.x0 = (
            np.zeros(self._n, dtype=np.float64)
            if self.x0 is None
            else np.asarray(self.x0, dtype=np.float64).ravel()
        )
        self.P0 = (
            np.eye(self._n, dtype=np.float64)
            if self.P0 is None
            else np.atleast_2d(np.asarray(self.P0, dtype=np.float64))
        )
        self._scalar = self._n == 1

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self):
        return id(self)

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    def filter(self, y: np.ndarray) -> dict:
        return kf_filter(
            self.F, self.H, self.Q, self.R,
            self.c, self.d, self.x0, self.P0,
            y, self._scalar, self._n, self._m,
        )

    def loglikelihood(self, y: np.ndarray) -> float:
        return self.filter(y)["loglik"]
