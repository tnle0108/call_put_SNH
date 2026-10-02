#%%

import numpy as np
import pandas as pd
import sys

from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[0]))

import quantmr
from quantmr.model.shortrate.hullwhite import HullWhite
from quantmr.utils import DayCount



def calc_rho(CURVE_NAMES: list, value_date, n_window: int | None = None, save: bool = False):
    """
    Estimate the correlation of the Hull-White factor shocks between curves (HW doc, Phụ lục 07).

    For each curve the Hull-White state x(t) is filtered on all curve data up to ``value_date``
    ("từ 2023 (thời điểm bắt đầu có dữ liệu) đến nay"), smoothed with a Rauch-Tung-Striebel pass,
    and rescaled as x~(t) = exp(a * t) * (x(t) - mu_t), with t in act365 years from the first
    filtered date. Since dx~ = sigma * exp(a t) dW, the Pearson correlation of the daily changes
    of x~ is the correlation of the Brownian drivers dW.

    Args:
        CURVE_NAMES (list): Curve names known to HullWhite.get, e.g. ["FI_ZYC_VND_VBMA_Bond_FI",
            "sob4"].
        value_date: Valuation date; no curve data after it is used.
        n_window (int | None): If given, correlate only the last ``n_window`` daily changes (a
            sensitivity option; the document uses the whole history, the default).
        save (bool): Also write the matrix to datasets/correlation/corr.csv.

    Returns:
        pd.DataFrame: Correlation matrix of the x~ daily changes, indexed and columned by curve
        name.
    """
    ROOT      = Path(quantmr.__file__).resolve().parent.parent
    CORR_FILE = ROOT / "datasets" / "correlation" / "corr.csv"
    as_of_range = ("1900-01-01", str(pd.Timestamp(value_date).date()))

    hw_by_curve = {}
    state_by_curve = {}
    for name in CURVE_NAMES:
        # CurveNode.CURVENODE_CACHE.clear()
        hw = HullWhite.get(name)
        print("a         =", repr(hw.a))
        print("sigma     =", repr(hw.sigma))
        print("sigma_eps =", repr(hw.sigma_eps))


        res = hw.filter_state(name, as_of_range=as_of_range)
        hw_by_curve[name] = hw
        state_by_curve[name] = res


    def rts_smooth_scalar(filt: dict) -> np.ndarray:
        """
        Rauch-Tung-Striebel backward smoother for a scalar Kalman filter state.

        Steps whose predicted variance is not positive are left at their filtered value.

        Args:
            filt (dict): Filter output with x_filt, P_filt, x_pred_next, P_pred_next and F_used.

        Returns:
            np.ndarray: Smoothed state, one value per filter date.
        """
        x_filt = filt["x_filt"]
        P_filt = filt["P_filt"]
        x_pred_next = filt["x_pred_next"]
        P_pred_next = filt["P_pred_next"]
        F_used = filt["F_used"]
        T = x_filt.shape[0]
        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()
        for t in range(T - 2, -1, -1):
            P_pred_t1 = P_pred_next[t, 0, 0]
            if P_pred_t1 <= 0:
                continue
            F_t = F_used[t, 0, 0]
            J_t = P_filt[t, 0, 0] * F_t / P_pred_t1
            x_smooth[t, 0] = x_filt[t, 0] + J_t * (
                x_smooth[t + 1, 0] - x_pred_next[t, 0]
            )
            P_smooth[t, 0, 0] = P_filt[t, 0, 0] + J_t**2 * (
                P_smooth[t + 1, 0, 0] - P_pred_t1
            )
        return x_smooth[:, 0]

    dc = DayCount.get("act365")
    tilde_x_by_curve = {}
    for name in CURVE_NAMES:
        r = state_by_curve[name]
        idx = pd.DatetimeIndex(r["dates"])
        x_smooth = rts_smooth_scalar(r["filter_result"])
        a = float(hw_by_curve[name].a)
        t0 = idx[0].to_numpy().astype("datetime64[D]")
        t_years = dc.yearfrac(
            np.full(len(idx), t0, dtype="datetime64[D]"),
            idx.values.astype("datetime64[D]"),
        )
        tilde_x_smooth = np.exp(a * t_years) * (x_smooth - r["mu_t"])
        tilde_x_by_curve[name] = pd.Series(tilde_x_smooth, index=idx, name=name)

    # Dates common to every curve, all on or before value_date (the filter only saw those).
    tilde_x_frame = pd.concat(tilde_x_by_curve.values(), axis=1, join="inner").dropna()
    eps_full = tilde_x_frame.diff().dropna()
    if n_window is not None:
        eps_full = eps_full.iloc[-n_window:]
    corr_eps = eps_full.corr()

    pd.set_option("display.float_format", lambda v: f"{v: .6f}")


    corr_out = corr_eps.copy()
    corr_out.index.name = "Corr"
    corr_out = corr_out.reset_index()

    if save:
        CORR_FILE.parent.mkdir(parents=True, exist_ok=True)
        corr_out.to_csv(CORR_FILE, index=False)
        print(f"Saved rigorous \u03b5-based correlation to {CORR_FILE}")

    return corr_eps
#%%