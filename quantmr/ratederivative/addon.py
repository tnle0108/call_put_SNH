import os
from functools import partial
import numpy as np
import pandas as pd
from scipy import stats

from quantmr.config import load_spec
from quantmr.fx import FX
from quantmr.leg import FixedLeg, FloatingLeg
from quantmr.schedule import BusCalendar

try:
    from tqdm import tqdm as _tqdm

    _tqdm_bar = partial(_tqdm, ncols=100, leave=False)
except ImportError:

    def _tqdm_bar(it, **kw):
        return it

_ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_USD_SPREAD_FILE = os.path.join(_ROOT_DIR, "datasets", "spread", "spread.csv")
_USD_FIXED_RATE_FILE = os.path.join(
    _ROOT_DIR, "datasets", "fixed_rate", "fixed_usd.csv"
)

_USD_SPREAD_CACHE: pd.DataFrame | None = None
_USD_FIXED_RATE_CACHE: pd.DataFrame | None = None

TERMS = [
    "1m",
    "2m",
    "3m",
    "6m",
    "9m",
    "1y",
    "1y6m",
    "2y",
    "2y6m",
    "3y",
    "3y6m",
    "4y",
    "4y6m",
    "5y",
    "5y6m",
    "6y",
    "6y6m",
    "7y",
    "7y6m",
    "8y",
    "8y6m",
    "9y",
    "9y6m",
    "10y",
]


def _pv_long_to_rc(pv_long):
    pv_long = np.asarray(pv_long, dtype=np.float64)
    pv_short = -pv_long
    rc_long = np.maximum(pv_long, 0.0)
    rc_short = np.maximum(pv_short, 0.0)
    return pv_short, rc_long, rc_short


def _remaining_term(
    as_of,
    end_date,
    calendar=None,
    busadjust=None,
    eom=False,
):
    as_of_d = np.datetime64(as_of).astype("datetime64[D]")
    end_d = np.datetime64(end_date).astype("datetime64[D]")
    if end_d <= as_of_d:
        return "expired"
    cal = BusCalendar.get(calendar)
    for t in TERMS:
        t_date = cal.shift(as_of_d, term=t, busadjust=busadjust, eom=eom)
        if end_d <= t_date:
            return t
    return TERMS[-1]


def _enforce_monotonic(addon_table):
    addon_table = addon_table.copy()
    addon_table.index = pd.CategoricalIndex(
        addon_table.index,
        categories=TERMS,
        ordered=True,
    )
    addon_table = addon_table.sort_index()
    for col in addon_table.columns:
        addon_table[col] = np.maximum.accumulate(
            addon_table[col].values.astype(np.float64)
        )
    return addon_table


def _get_usd_spread(start, term: str) -> np.float64:
    global _USD_SPREAD_CACHE
    if _USD_SPREAD_CACHE is None:
        if os.path.exists(_USD_SPREAD_FILE):
            _USD_SPREAD_CACHE = pd.read_csv(
                _USD_SPREAD_FILE, index_col=0, parse_dates=True
            )
        else:
            _USD_SPREAD_CACHE = pd.DataFrame()
    df = _USD_SPREAD_CACHE
    if df.empty:
        return np.float64(0.0)
    start_ts = pd.Timestamp(start)
    col = term.upper()
    if start_ts in df.index and col in df.columns:
        return np.float64(df.loc[start_ts, col])
    idx = df.index.get_indexer([start_ts], method="ffill")
    if idx[0] >= 0 and col in df.columns:
        return np.float64(df.iloc[idx[0]][col])
    return np.float64(0.0)


def _get_usd_fixed_rate(start, term: str) -> np.float64:
    global _USD_FIXED_RATE_CACHE
    if _USD_FIXED_RATE_CACHE is None:
        if os.path.exists(_USD_FIXED_RATE_FILE):
            _USD_FIXED_RATE_CACHE = pd.read_csv(_USD_FIXED_RATE_FILE)
        else:
            _USD_FIXED_RATE_CACHE = pd.DataFrame()
    df = _USD_FIXED_RATE_CACHE
    if df.empty:
        return np.float64(np.nan)
    start_pd = pd.to_datetime(start)
    month, year = start_pd.month, start_pd.year
    mask = (df["month"] == month) & (df["year"] == year)
    matching = df[mask]
    if matching.empty:
        raise ValueError(
            f"No USD fixed rate data for month={month}, year={year}"
        )
    row = matching.iloc[0]
    col = term.upper()
    tenor_cols = [c for c in df.columns if c not in ("month", "year")]
    if col in tenor_cols:
        return np.float64(row[col])
    return np.float64(np.max(row[tenor_cols].values))


class AddOnRateDerivative:

    def __init__(
        self,
        name: str,
        sim_date: str | np.datetime64,
        report_date: str | np.datetime64,
        start_dates: list,
        terms: list[str] | None = None,
        freqs: list[str] | None = None,
        n_samples: int = 10_000,
        state: int | None = None,
    ):
        self.name = name.lower()
        self.sim_date = np.datetime64(sim_date).astype("datetime64[D]")
        self.report_date = np.datetime64(report_date).astype("datetime64[D]")
        self.start_dates = [
            np.datetime64(d).astype("datetime64[D]") for d in start_dates
        ]
        self.terms = terms or TERMS
        self.freqs = freqs or ["m", "q", "s"]
        self.n_samples = n_samples
        self.state = state
        self._spec = load_spec("ratederivative").get(self.name, {})
        self._rate_cache: dict = {}


    def _get_notional(self, side: str, start: np.datetime64) -> float:
        deri_type = self._spec.get("type", "irs")
        if deri_type == "irs":
            return 1.0
        domestic = self._spec.get("domestic", "").lower()
        ccy = self._spec.get(f"{side}_currency", "").lower()
        if ccy == domestic or not domestic:
            return 1.0
        fx_pair = self._spec.get("fx_pair")
        if not fx_pair:
            return 1.0
        fx = FX.get((fx_pair, start))
        raw = float(fx.spot)
        if fx_pair.lower().startswith(ccy):
            return 1.0 / raw
        return raw


    def _mtm_dates(self, start: np.datetime64, term: str) -> list:
        from quantmr.schedule import Schedule

        short_terms = ["1m", "2m"]
        if term.lower() in short_terms:
            mtm_freq = "w"
        else:
            mtm_freq = "m"
        sched = Schedule(start=start, term=term, freq=mtm_freq)
        return [d for d in sched.aends if d > self.sim_date]

    def _remaining_term(
        self, as_of: np.datetime64, end_date: np.datetime64
    ) -> str:
        return _remaining_term(
            as_of,
            end_date,
            calendar=self._spec.get("calendar"),
            busadjust=self._spec.get("busadjust"),
            eom=self._spec.get("eom", False),
        )

    def _get_foreign_pay_rates(self, start: np.datetime64, term: str) -> dict:
        receive_fixed_rate = None
        receive_spread = None
        deri_type = self._spec.get("type", "irs")
        if deri_type == "irs":
            return {"receive_fixed_rate": receive_fixed_rate, "receive_spread": receive_spread}

        ccs_type = self._spec.get("ccs_type", "")
        if ccs_type == "fixedfixed":
            if self._spec.get("receive_fixed_rate") is None:
                receive_fixed_rate = _get_usd_fixed_rate(start, term)
        elif ccs_type == "floatfixed":
            receive_spread = _get_usd_spread(start, term)

        return {"receive_fixed_rate": receive_fixed_rate, "receive_spread": receive_spread}

    def _get_inception_rates(
        self,
        start: np.datetime64,
        term: str,
        freq: str,
        pay_notional: float,
        rec_notional: float,
    ) -> dict:
        cache_key = (self.name, str(start), term, freq)
        if cache_key in self._rate_cache:
            return self._rate_cache[cache_key]

        from quantmr.ratederivative.ratederivative import RateDerivative

        foreign = self._get_foreign_pay_rates(start, term)

        tran = RateDerivative.get(
            self.name,
            start=start,
            term=term,
            freq=freq,
            disc_curve_as_of=start,
            pay_notional=pay_notional,
            rec_notional=rec_notional,
            fx_as_of=start,
            rec_fixed_rate=foreign["receive_fixed_rate"],
            rec_spread=foreign["receive_spread"],
        )
        rates = {
            "pay_fixed_rate": (
                tran.pay_leg._fixed_rate
                if isinstance(tran.pay_leg, FixedLeg)
                else None
            ),
            "rec_fixed_rate": (
                tran.rec_leg._fixed_rate
                if isinstance(tran.rec_leg, FixedLeg)
                else None
            ),
            "pay_spread": (
                tran.pay_leg._spread
                if isinstance(tran.pay_leg, FloatingLeg)
                else None
            ),
            "rec_spread": (
                tran.rec_leg._spread
                if isinstance(tran.rec_leg, FloatingLeg)
                else None
            ),
        }
        self._rate_cache[cache_key] = rates
        return rates


    def simulate_one(self, start: np.datetime64, term: str, freq: str) -> dict:
        from quantmr.ratederivative.ratederivative import RateDerivative

        mtm_dates = self._mtm_dates(start, term)
        if not mtm_dates:
            return {}

        pay_notional = self._get_notional("pay", start)
        rec_notional = self._get_notional("rec", start)
        rates = self._get_inception_rates(start, term, freq, pay_notional, rec_notional)

        tran_at_sim = RateDerivative.get(
            self.name,
            start=start,
            term=term,
            freq=freq,
            disc_curve_as_of=self.sim_date,
            pay_notional=pay_notional,
            rec_notional=rec_notional,
            fx_as_of=self.sim_date,
            **rates,
            solve=False,
        )
        end_date = tran_at_sim.rec_leg.schedule.aends[-1]
        remaining = self._remaining_term(self.sim_date, end_date)
        pv_at_sim = float(tran_at_sim.pv(self.sim_date).ravel()[0])

        tran = RateDerivative.get(
            self.name,
            start=start,
            term=term,
            freq=freq,
            disc_curve_as_of=self.sim_date,
            ref_curve_as_of=self.sim_date,
            disc_is_sim=True,
            ref_is_sim=True,
            pay_notional=pay_notional,
            rec_notional=rec_notional,
            **rates,
            fx_as_of=self.sim_date,
            solve=False,
        )

        mtm_arr = tran.mtm(
            mtm_dates, n_samples=self.n_samples, state=self.state
        )

        pv_long = mtm_arr
        pv_short = -pv_long
        rc_long = np.maximum(pv_long, 0.0)
        rc_short = np.maximum(pv_short, 0.0)
        pv_short_at_sim, rc_long_at_sim, rc_short_at_sim = _pv_long_to_rc(
            pv_at_sim
        )
        pcts = {
            "pv_long":      np.percentile(pv_long,                    [80, 95], axis=1),
            "pv_short":     np.percentile(pv_short,                   [80, 95], axis=1),
            "rc_long":      np.percentile(rc_long,                    [80, 95], axis=1),
            "rc_short":     np.percentile(rc_short,                   [80, 95], axis=1),
            "addon_pv_long":  np.percentile(pv_long  - pv_at_sim,       [80, 95], axis=1),
            "addon_pv_short": np.percentile(pv_short - pv_short_at_sim, [80, 95], axis=1),
            "addon_rc_long":  np.percentile(pv_long  - rc_long_at_sim,  [80, 95], axis=1),
            "addon_rc_short": np.percentile(pv_short - rc_short_at_sim, [80, 95], axis=1),
        }

        results: dict = {}
        for i, vd in enumerate(mtm_dates):
            results[(start, term, freq, vd)] = {
                "end": end_date,
                "sim": self.sim_date,
                "remaining_term": remaining,
                "pay_notional": pay_notional,
                "pv_at_sim": np.float64(pv_at_sim),
                "pv_long_95":        pcts["pv_long"][1, i],
                "pv_long_80":        pcts["pv_long"][0, i],
                "pv_short_95":       pcts["pv_short"][1, i],
                "pv_short_80":       pcts["pv_short"][0, i],
                "rc_long_95":        pcts["rc_long"][1, i],
                "rc_long_80":        pcts["rc_long"][0, i],
                "rc_short_95":       pcts["rc_short"][1, i],
                "rc_short_80":       pcts["rc_short"][0, i],
                "addon_pv_long_95":  pcts["addon_pv_long"][1, i],
                "addon_pv_long_80":  pcts["addon_pv_long"][0, i],
                "addon_pv_short_95": pcts["addon_pv_short"][1, i],
                "addon_pv_short_80": pcts["addon_pv_short"][0, i],
                "addon_rc_long_95":  pcts["addon_rc_long"][1, i],
                "addon_rc_long_80":  pcts["addon_rc_long"][0, i],
                "addon_rc_short_95": pcts["addon_rc_short"][1, i],
                "addon_rc_short_80": pcts["addon_rc_short"][0, i],
            }
        return results


    def simulate(self) -> pd.DataFrame:
        all_stats: dict = {}
        combos = [
            (start, term, freq)
            for start in self.start_dates
            for term in self.terms
            for freq in self.freqs
        ]
        for start, term, freq in _tqdm_bar(combos, desc="simulate"):
            all_stats.update(self.simulate_one(start, term, freq))
        df = pd.DataFrame.from_dict(all_stats, orient="index")
        df.index.names = ["start", "term", "freq", "value"]
        return df.reset_index()

    def addon_table(self, sim_df: pd.DataFrame) -> pd.DataFrame:
        addon_cols = [c for c in sim_df.columns if c.startswith("addon_")]
        tbl = (
            sim_df[addon_cols + ["remaining_term"]]
            .groupby("remaining_term")
            .agg("max")
        )
        tbl = tbl.clip(lower=0.0)
        return _enforce_monotonic(tbl)


    def backtest_one(
        self,
        start: np.datetime64,
        term: str,
        freq: str,
        backtest_dates: list,
    ) -> dict:
        from quantmr.ratederivative.ratederivative import RateDerivative

        results: dict = {}
        pay_notional = self._get_notional("pay", start)
        rec_notional = self._get_notional("rec", start)
        rates = self._get_inception_rates(start, term, freq, pay_notional, rec_notional)
        for vd in backtest_dates:
            tran = RateDerivative.get(
                self.name,
                start=start,
                term=term,
                freq=freq,
                disc_curve_as_of=vd,
                ref_curve_as_of=vd,
                pay_notional=pay_notional,
                rec_notional=rec_notional,
                fx_as_of=vd,
                **rates,
                solve=False,
            )
            end_date = tran.rec_leg.schedule.aends[-1]
            remaining = self._remaining_term(vd, end_date)
            pv_val = float(tran.pv(vd).ravel()[0])
            results[(start, term, freq, vd)] = {
                "end": end_date,
                "remaining_term": remaining,
                "pay_notional": pay_notional,
                "pv": pv_val,
            }
        return results

    def backtest(self, backtest_dates: list) -> pd.DataFrame:
        all_pv: dict = {}
        for start in self.start_dates:
            for term in self.terms:
                for freq in self.freqs:
                    all_pv.update(
                        self.backtest_one(start, term, freq, backtest_dates)
                    )
        df = pd.DataFrame.from_dict(all_pv, orient="index")
        df.index.names = ["start", "term", "freq", "value"]
        df = df.reset_index()
        df = df[df["value"] <= df["end"]]
        df = df[df["value"] > df["start"]]
        pv_short, rc_long, rc_short = _pv_long_to_rc(
            np.asarray(df["pv"].values)
        )
        df["pv_long"] = df["pv"]
        df["pv_short"] = pv_short
        df["rc_long"] = rc_long
        df["rc_short"] = rc_short
        df.drop(columns=["pv"], inplace=True)
        return df


    @staticmethod
    def binomial_calibrate(
        addon_table: pd.DataFrame,
        backtest_df: pd.DataFrame,
        alpha: float = 0.05,
        alpha_red: float = 0.0001,
        tol: float = 0.0001,
    ) -> pd.DataFrame:
        bt = pd.merge(
            backtest_df,
            addon_table,
            left_on="remaining_term",
            right_index=True,
            suffixes=("", "_addon"),
        )
        if "deal" not in bt.columns:
            bt["deal"] = (
                bt["start"].astype(str) + "_" + bt["term"] + "_" + bt["freq"]
            )
        bt = bt.sort_values(["deal", "value"])
        bucket_jump = bt.groupby(["deal", "remaining_term"])["value"].transform(
            "min"
        )
        bt["bucket_jump_date"] = bucket_jump

        pv_lookup = bt[["deal", "value", "pv_long"]].copy()
        pv_lookup = pv_lookup.rename(
            columns={
                "value": "bucket_jump_date",
                "pv_long": "pv_long_on_jump_date",
            }
        )
        bt = pd.merge(
            bt, pv_lookup, on=["deal", "bucket_jump_date"], how="left"
        )
        pv_s_jd, rc_l_jd, rc_s_jd = _pv_long_to_rc(
            np.asarray(bt["pv_long_on_jump_date"].values)
        )
        bt["pv_short_on_jump_date"] = pv_s_jd
        bt["rc_long_on_jump_date"] = rc_l_jd
        bt["rc_short_on_jump_date"] = rc_s_jd

        def _max_future(x):
            vals = x.values
            rev = np.maximum.accumulate(vals[::-1])[::-1]
            out = np.empty_like(rev, dtype=float)
            out[:-1] = rev[1:]
            out[-1] = np.nan
            return pd.Series(out, index=x.index)

        for col in ["pv_long", "pv_short"]:
            bt[f"max_future_{col}"] = bt.groupby("deal", group_keys=False)[
                col
            ].apply(_max_future)
            bt[f"delta_{col}"] = (
                bt[f"max_future_{col}"] - bt[f"{col}_on_jump_date"]
            )

        bt["max_future_rc_long"] = bt["max_future_pv_long"]
        bt["delta_rc_long"] = (
            bt["max_future_pv_long"] - bt["rc_long_on_jump_date"]
        )
        bt["max_future_rc_short"] = bt["max_future_pv_short"]
        bt["delta_rc_short"] = (
            bt["max_future_pv_short"] - bt["rc_short_on_jump_date"]
        )

        metrics = ["pv_long", "pv_short", "rc_long", "rc_short"]

        fail_cols = {f"addon_{m}_95": m for m in metrics}
        adjustments: dict[tuple[str, str], float] = {}

        tenors = addon_table.index.tolist()
        cells = [(col, m, tenor) for col, m in fail_cols.items() for tenor in tenors]

        for addon_col, metric, tenor in _tqdm_bar(
            cells, desc="buffer calibration"
        ):
            delta_col = f"delta_{metric}"
            tenor_bt = bt[bt["remaining_term"] == tenor]
            valid = tenor_bt.dropna(subset=[metric, f"max_future_{metric}"])
            n_obs = len(valid)
            if n_obs == 0:
                adjustments[(tenor, addon_col)] = 0.0
                continue

            addon_vals = valid[addon_col].values
            delta_vals = valid[delta_col].values

            def _is_red(adj: float) -> bool:
                k = int(np.sum(delta_vals > addon_vals + adj))
                result = stats.binomtest(
                    k, n_obs, p=alpha, alternative="greater"
                )
                return result.pvalue <= alpha_red

            if not _is_red(0.0):
                adjustments[(tenor, addon_col)] = 0.0
                continue

            adj = tol
            while _is_red(adj) and adj < 10.0:
                adj += tol
            adjustments[(tenor, addon_col)] = adj

        calibrated = addon_table.copy()
        addon_95_cols = [c for c in calibrated.columns if c.endswith("_95")]
        for col in addon_95_cols:
            for tenor in tenors:
                calibrated.loc[tenor, col] = calibrated.loc[
                    tenor, col
                ] + adjustments.get(
                    (tenor, col), 0.0
                )
        calibrated = _enforce_monotonic(calibrated)
        return calibrated
