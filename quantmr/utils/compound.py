import numpy as np
from quantmr.utils.helpers import align_arr, norm_str

COMPOUNDINGS = frozenset({"continuous", "simple", "annually"})


def _norm_compounding(compounding: str | None) -> str:
    c = norm_str(compounding or "continuous")
    if c not in COMPOUNDINGS:
        raise ValueError(f"Unsupported compounding method: {compounding!r}")
    return c


def df_from_rate(
    rate: float | np.ndarray,
    t: float | np.ndarray,
    compounding: str | None = None,
    **align_arr_kwargs,
) -> np.ndarray:
    c = _norm_compounding(compounding)
    rate = np.asarray(rate, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    rate, t = align_arr(rate, t, **align_arr_kwargs)
    match c:
        case "continuous":
            df = np.exp(-rate * t)
        case "simple":
            df = 1.0 / (1.0 + rate * t)
        case "annually":
            df = (1.0 + rate) ** (-t)
    df = np.where(t < 0, np.nan, df)
    df = np.where(t == 0, 1.0, df)
    return df


def rate_from_df(
    df: float | np.ndarray,
    t: float | np.ndarray,
    compounding: str | None = None,
    **align_arr_kwargs,
) -> np.ndarray:
    c = _norm_compounding(compounding)
    df = np.asarray(df, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    df, t = align_arr(df, t, **align_arr_kwargs)
    with np.errstate(divide="ignore", invalid="ignore"):
        match c:
            case "continuous":
                rate = -np.log(df) / t
            case "simple":
                rate = (1.0 / df - 1.0) / t
            case "annually":
                rate = df ** (-1.0 / t) - 1.0
    rate = np.where(t < 0, np.nan, rate)
    rate = np.where(np.isclose(t, 0), np.nan, rate)
    return rate


class Compound:

    COMPOUNDINGS = COMPOUNDINGS
    COMPOUND_CACHE: dict[str, "Compound"] = {}

    def __init__(self, compounding: str | None = None):
        self._compounding = compounding

    @property
    def compounding(self) -> str:
        return _norm_compounding(self._compounding)

    def df_from_rate(
        self,
        rate: float | np.ndarray,
        t: float | np.ndarray,
        **align_arr_kwargs,
    ) -> np.ndarray:
        return df_from_rate(rate, t, self._compounding, **align_arr_kwargs)

    def rate_from_df(
        self,
        df: float | np.ndarray,
        t: float | np.ndarray,
        **align_arr_kwargs,
    ) -> np.ndarray:
        return rate_from_df(df, t, self._compounding, **align_arr_kwargs)

    @classmethod
    def get(cls, compounding: str | None = None) -> "Compound":
        key = norm_str(compounding or "continuous")
        if key not in cls.COMPOUND_CACHE:
            cls.COMPOUND_CACHE[key] = cls(compounding)
        return cls.COMPOUND_CACHE[key]
