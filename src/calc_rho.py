#%%

import numpy as np
import pandas as pd
import sys

from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[0]))

import quantmr
from quantmr.schedule import BusCalendar
from quantmr.model.shortrate.hullwhite import HullWhite
from quantmr.utils import DayCount
from quantmr.curve.curvenode import CurveNode



def calc_rho(CURVE_NAMES: list):

    ROOT      = Path(quantmr.__file__).resolve().parent.parent
    CORR_FILE = ROOT / "datasets" / "correlation" / "corr.csv"

    REPORT_DATE = np.datetime64("2025-12-31")
    N_WINDOW    = 250
    HOLIDAYS    = "vnd"

    # CURVE_NAMES = ["zc usd sr", "zc vnd ccs sr", "zc vnd irs"]


    def get_business_dates(report_date: np.datetime64, n: int, calendar: str) -> np.ndarray:
        buscal = BusCalendar.get(calendar)
        candidate = pd.date_range(
            end=pd.Timestamp(report_date) - pd.Timedelta(days=1), periods=3 * n, freq="D"
        ).values.astype("datetime64[D]")
        busdays = candidate[buscal.is_busday(candidate)]
        return np.sort(busdays[-n:])

    DATES = get_business_dates(REPORT_DATE, N_WINDOW+1, HOLIDAYS)


    as_of_range = (str(DATES[0] - np.timedelta64(750, "D")), str(REPORT_DATE))

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

    tilde_x_frame = pd.concat(tilde_x_by_curve.values(), axis=1, join="inner")
    window_idx = pd.DatetimeIndex(DATES)
    tilde_x_frame = tilde_x_frame.reindex(window_idx, method=None).dropna()

    eps      = tilde_x_frame.diff()
    eps_full = eps.dropna()


    corr_eps = eps_full.corr()

    pd.set_option("display.float_format", lambda v: f"{v: .6f}")


    corr_out = corr_eps.copy()
    corr_out.index.name = "Corr"
    corr_out = corr_out.reset_index()

    CORR_FILE.parent.mkdir(parents=True, exist_ok=True)
    corr_out.to_csv(CORR_FILE, index=False)
    print(f"Saved rigorous \u03b5-based correlation to {CORR_FILE}")

    return corr_eps
#%%