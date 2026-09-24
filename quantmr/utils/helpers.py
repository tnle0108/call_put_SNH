import re
import numpy as np


_DATE_PATTERNS: list[tuple[re.Pattern, str | None]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), None),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "slash_ymd"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "slash_mdy"),
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "dash_mdy"),
]


def norm_str(s: str) -> str:
    s = s.lower()
    return re.sub(r"[^a-z0-9+]", "", s)


def get_year(date: np.ndarray) -> np.ndarray:
    return date.astype("datetime64[Y]").astype(int) + 1970


def is_leap_year(date: np.ndarray) -> np.ndarray:
    year = get_year(date)
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


def day_of_year(date: np.ndarray) -> np.ndarray:
    year_start = date.astype("datetime64[Y]")
    return ((date - year_start) / np.timedelta64(1, "D")).astype(int) + 1


def to_roll(date: np.ndarray, roll: int = 31) -> np.ndarray:
    date = np.asarray(date, dtype="datetime64[D]")
    out = np.empty(date.shape, dtype="datetime64[D]")
    out[:] = np.datetime64("NaT")
    valid = ~np.isnat(date)
    if valid.any():
        d = date[valid]
        month = d.astype("datetime64[M]")
        month_start = month.astype("datetime64[D]")
        days_in_month = (
            (month + 1).astype("datetime64[D]") - month_start
        ).astype(int)
        roll_day = np.minimum(roll, days_in_month)
        out[valid] = month_start + (roll_day - 1).astype("timedelta64[D]")
    return out


def to_month_end(date: np.ndarray) -> np.ndarray:
    return to_roll(date, roll=31)


def is_month_end(date: np.ndarray) -> np.ndarray:
    return to_month_end(date) == date


def align_arr(
    x: np.ndarray, y: np.ndarray, axis: int = -1
) -> tuple[np.ndarray, np.ndarray]:

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError(
            f"align_arr supports at most 2-d arrays, got "
            f"x.ndim={x.ndim}, y.ndim={y.ndim}"
        )
    if x.ndim == y.ndim:
        bx, by = np.broadcast_arrays(x, y)
        return bx, by
    if x.ndim > y.ndim:
        larger, smaller, x_is_larger = x, y, True
    else:
        larger, smaller, x_is_larger = y, x, False
    l_ndim = larger.ndim
    s_ndim = smaller.ndim
    new_shape = [1] * l_ndim
    ax = axis % l_ndim
    first_ax = ax - s_ndim + 1
    if first_ax < 0:
        raise ValueError(
            f"Cannot align array with {s_ndim} dims along axis "
            f"{axis} of array with {l_ndim} dims"
        )
    for i in range(s_ndim):
        new_shape[first_ax + i] = smaller.shape[i]
    smaller = smaller.reshape(new_shape)
    if x_is_larger:
        bx, by = np.broadcast_arrays(x, smaller)
        return bx, by
    bx, by = np.broadcast_arrays(smaller, y)
    return bx, by


def parse_date_string(s: str) -> str | None:
    s = s.strip()
    for pat, kind in _DATE_PATTERNS:
        if pat.match(s):
            if not kind:
                return s
            if kind == "slash_ymd":
                return s.replace("/", "-")
            sep = "/" if "slash" in kind else "-"
            parts = s.split(sep)
            return f"{parts[2]}-{parts[0]}-{parts[1]}"
    return None


def to_array(
    x: float | np.float64 | str | np.datetime64 | list | np.ndarray,
) -> tuple[np.ndarray, str]:
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.integer
    ):
        return np.asarray(x, dtype=np.float64), "float"
    elif np.issubdtype(x.dtype, np.datetime64):
        return x.astype("datetime64[D]"), "date"
    elif np.issubdtype(x.dtype, np.str_):
        flat = x.flatten()
        parsed = (
            [parse_date_string(str(s)) for s in flat] if len(flat) > 0 else []
        )
        if parsed and all(p for p in parsed):
            return (
                np.array(parsed, dtype="datetime64[D]").reshape(x.shape),
                "date",
            )
        return x, "str"
    else:
        raise TypeError(f"Unsupported input type: {x.dtype}")
