import pandas as pd
import numpy as np
import sys
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path.cwd().parents[0]))

from callput import (
    BondSchedule,
    CouponPeriod,
)

EXTRA_WORKING_DAYS = {
    pd.Timestamp("2026-01-10"),
    pd.Timestamp("2024-05-04"),
    pd.Timestamp("2025-04-26"),
    pd.Timestamp("2026-08-22")
} # Vietnam compensatory working weekends

from src.map_curve import MapCurve


def tenor_to_years(tenor: str) -> float:
    """
    Convert a tenor string (e.g., '6M', '1Y') to a float representing the number of years.

    Assuming 1 year = 12 months = 52 weeks.

    Args:
        tenor (str): The tenor string to convert. Years (Y), Months (M), Weeks (W), and ON are
        supported.

    Returns:
        float: The equivalent number of years.
    """
    tenor = str(tenor).strip().upper()
    if tenor.endswith("Y"):
        return int(tenor[:-1])
    elif tenor.endswith("M"):
        return int(tenor[:-1]) / 12.0
    elif tenor.endswith("W"):
        return int(tenor[:-1]) / 52.0
    elif tenor == "ON":
        return 1 / 365.0
    else:
        raise ValueError(f"Unsupported tenor: {tenor}")


def get_known_rate(
    ref_tenor: str,
    fixing_date: pd.Timestamp,
    margin: float,
    ref_df: pd.DataFrame,
    convention: str,
) -> float:
    """
    Look up an already-fixed reference rate on its fixing date and add a margin.

    The reference curve row in effect on fixing_date is mapped to a YieldCurve (see MapCurve) and
    its zero rate is read at the reference tenor.

    Args:
        ref_tenor (str): Tenor of the reference rate, e.g. '6M' or '1Y' (see tenor_to_years).
        fixing_date (pd.Timestamp): Date on which the rate was fixed.
        margin (float): Margin added to the zero rate.
        ref_df (pd.DataFrame): Historical zero rates of the reference curve, one row per date and
            one column per tenor.
        convention (str): Day count convention used to turn tenors into year fractions.

    Returns:
        float: Zero rate of the reference curve at ref_tenor on fixing_date, plus margin.
    """
    hist_curve = MapCurve(
        rpd=fixing_date,
        df = ref_df,
        convention=convention,
    ).map_curve()

    t = tenor_to_years(ref_tenor)
    ref_rate = hist_curve.zero_rate(t)
    return float(ref_rate) + float(margin)

def load_holiday_calendar(folder: str | Path) -> dict[str, set[pd.Timestamp]]:
    """
    Helper function to load holiday calendars from CSV files in a specified folder.
    
    Args:
        folder (str): 
            - Path to the folder containing holiday CSV files. 
            - Each CSV file should have a "Date" column with holiday dates.
            - Each file name (without extension) will be used as the country name in the returned
              dictionary.

    Returns:
        dict: A dictionary where keys are country names (derived from CSV file names) and values are
        sets of holiday dates (as pd.Timestamp objects) for that country.
    """
    holiday_dict = {}

    for file in Path(folder).glob("*.csv"):
        country = file.stem

        df = pd.read_csv(file)

        holidays = set(
            pd.to_datetime(df["Date"]).dt.normalize()
        )

        holiday_dict[country] = holidays

    return holiday_dict




def adjust_following(date: pd.Timestamp | str, holidays: set[pd.Timestamp]) -> pd.Timestamp:
    """
    Find the following working day for a given date, considering weekends and holidays.

    If the date falls on a holiday or a weekend (Saturday or Sunday), the function will return the
    next working day.
    If the date is already a working day, it will return the same date.
    Weekend days listed in EXTRA_WORKING_DAYS (Vietnam compensatory working weekends) count as
    working days.

    Args:
        date (pd.Timestamp | str): The date to adjust.
        holidays (set[pd.Timestamp]): Holiday dates of the relevant calendar.

    Returns:
        pd.Timestamp: The normalised date, rolled forward to the first working day.
    """
    date = pd.Timestamp(date).normalize()

    while True:
        if date in holidays:
            date += timedelta(days=1)
            continue # If holiday, move to the next day and check again

        if date.weekday() >= 5 and date not in EXTRA_WORKING_DAYS:
            date += timedelta(days=1)
            continue # If normal weekend (Saturday=5, Sunday=6) and not an extra working day, move to the next day and check again

        break

    return date

def nan_to_none(x: object) -> float | None:
    """
    Convert a missing value to None and anything else to float.

    Args:
        x (object): Value to convert, e.g. a floor or cap read from the term sheet.

    Returns:
        float | None: None if x is NaN/NA, otherwise float(x).
    """
    return None if pd.isna(x) else float(x)


def roll_periods(
    issue_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    rolling_date: pd.Timestamp,
    months: int,
    holidays: set[pd.Timestamp],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Starts and ends of coupon periods, rolled both ways. Used to calculate coupon accruals and
    be the basis for payment dates.

    The grid runs from rolling_date backwards and forwards (k = ..., -1, 0, 1, ...), 
    each date counted from rolling_date itself so a month-end anchor does not drift 
    (31/08 -> 28/02 -> 31/08).

    If the last end date before the maturity date is not a working day and adjusted to the maturity
    date, it would be replaced by the adjusted maturity date. It is common Vietnamese practice.

    Args:
        - issue_date (pd.Timestamp): Start of the first period.
        - maturity_date (pd.Timestamp): Maturity as stored (adjusted or not).
        - rolling_date (pd.Timestamp): Anchor of the grid.
        - months (int): Coupon frequency in months (6 = semi-annual, 12 = annual).
        - holidays (set): Holiday dates, used only to recognise adjusted dates.

    Returns:
        - list[tuple[pd.Timestamp, pd.Timestamp]]: (start, end) per period, oldest first.
    """
    if maturity_date <= issue_date:
        raise ValueError(f"maturity_date {maturity_date.date()} is not after issue_date {issue_date.date()}")

    # Grid dates from one period before issue_date to one period after maturity_date.
    lo, hi = issue_date - relativedelta(months=months), maturity_date + relativedelta(months=months) # max range of grid dates to consider
    grid, k = [], 0
    while (d := rolling_date - relativedelta(months=k * months)) >= lo:
        if d <= hi:
            grid.append(d)
        k += 1
    k = 1
    while (d := rolling_date + relativedelta(months=k * months)) <= hi:
        if d >= lo:
            grid.append(d)
        k += 1
    grid = sorted(grid)

    # A grid date that only adjusts onto the adjusted maturity is maturity itself: drop it, so
    # the last period ends on the adjusted maturity instead of leaving a 1-2 day stub.
    adjusted_maturity = adjust_following(maturity_date, holidays)
    inner = [d for d in grid if issue_date < d < maturity_date and adjust_following(d, holidays) != adjusted_maturity]
    ends = inner + [adjusted_maturity]
    starts = [issue_date] + inner
    return list(zip(starts, ends))

@dataclass
class CouponSchedule:
    """
    A class to build a coupon schedule DataFrame for bonds based on their issue and maturity dates,
    coupon accrual, and other parameters.
    
    Attributes:
        df (pd.DataFrame): A DataFrame containing bond information, including issue date, maturity
        date, coupon accrual, and other relevant fields.
        holiday_calendar (dict): A dictionary mapping country names to sets of holiday dates. Used
        to adjust payment dates according to the "following" business day convention.
        country (str): The country for which the holiday calendar should be applied when adjusting
        payment dates.
    """
    df: pd.DataFrame
    holiday_calendar: dict[str, set[pd.Timestamp]]
    country: str

    def build_coupon_schedule_df(self) -> pd.DataFrame:
        """
        Builds a DataFrame containing the coupon schedule for each bond in the input DataFrame.

        Given a rolling date, the method calculates the coupon starts, ends, and payment dates for
        each bond:
            - Starts and ends are unadjusted, based on the rolling date and coupon accrual, with an
              exception for the last period, which ends on the adjusted maturity date.
            - Payment dates are adjusted to the next business day if they fall on a holiday, using
              the provided holiday calendar.
            - Rolling date is given by the bond's rolling_date field, or defaults to the maturity
              date if not provided.

        Returns:
            pd.DataFrame: A DataFrame with columns for bond ID, payment dates, and other relevant
            information.
        """
        rows = []
        for _, bond in self.df.iterrows(): # Loop through each bond in the DataFrame to build its coupon schedule
            accrual = float(bond["coupon_accrual"]) # Coupon accrual period in years (e.g., 0.5 for semi-annual, 1.0 for annual)

            holidays = self.holiday_calendar.get(self.country, set())
            periods = self._coupon_periods(bond) # Unadjusted (start, end) on the rolling_date grid
            # Pay on the next business day after an end that falls on a holiday; the
            # final pay date is the (adjusted) maturity itself and included in the pay_dates list.
            pay_dates = [adjust_following(end, holidays) for _, end in periods]

            # Margin date is the date on which the margin added upon reference rate to calculate float rate change takes effect.
            margin_dates_raw = str(bond["margin_date"]) if pd.notna(bond["margin_date"]) and str(bond["margin_date"]).strip() else None
            margin_values_raw = str(bond["margin"]) if pd.notna(bond["margin"]) and str(bond["margin"]).strip() else None

            # Type change date is the date on which the coupon type changes (from fixed to floating).
            has_type_change = pd.notna(bond["coupon_type_change_date"]) and str(bond["coupon_type_change_date"]).strip()
            type_change_date = (
                adjust_following(pd.to_datetime(bond["coupon_type_change_date"], format='%m/%d/%Y'), holidays)
                if has_type_change else None
            ) # Type change date must be working day
            has_fixed_rate = pd.notna(bond["annual_coupon_rate"]) and str(bond["annual_coupon_rate"]).strip()
            has_ref_curve = pd.notna(bond["ref_curve"]) and str(bond["ref_curve"]).strip()

            coupon_change_values = None
            coupon_change_dates = None

            if has_fixed_rate:
                coupon_change_values = [float(x) for x in str(bond["annual_coupon_rate"]).split(";")]
                if pd.notna(bond["coupon_change_date"]) and str(bond["coupon_change_date"]).strip():
                    coupon_change_dates = pd.to_datetime(str(bond["coupon_change_date"]).split(";"), format='%m/%d/%Y')
                    coupon_change_dates = [adjust_following(d, holidays) for d in coupon_change_dates]
                    if len(coupon_change_values) != len(coupon_change_dates) + 1:
                        raise ValueError(
                            f"Expected {len(coupon_change_dates) + 1} coupon rates, "
                            f"got {len(coupon_change_values)}"
                        )
                else:
                    coupon_change_dates = []


            n_periods = len(periods)
            for step, ((start_date, end_date), pay_date) in enumerate(zip(periods, pay_dates), start=1): # Loop over each period and its corresponding pay date.
                # Interest runs start -> end (unadjusted); paying late on a holiday earns
                # nothing extra, since the next period already starts at end_date. The
                # maturity period has no next period, so it accrues up to the actual pay date.
                accrual_end_date = pay_date if step == n_periods else end_date # Only maturity period accrues to the actual pay date.
                if has_type_change:
                    is_fixed_period = pay_date <= type_change_date # Period is fixed if it ends on or before the type change date.
                else:
                    is_fixed_period = bool(has_fixed_rate)

                if is_fixed_period and not has_fixed_rate:
                    raise ValueError(
                        f"Bond {bond['bond_id']} has fixed coupon type but no annual_coupon_rate specified"
                    )

                if not is_fixed_period and not has_ref_curve:
                    raise ValueError(
                        f"Bond {bond['bond_id']} has floating coupon type but no ref_curve specified"
                    )
                if is_fixed_period:
                    coupon = coupon_change_values[0]
                    for i, d in enumerate(coupon_change_dates):
                        if pay_date > d:
                            coupon = coupon_change_values[i + 1]
                        else:
                            break
                else:
                    coupon = np.nan

                rows.append(
                    {
                        "bond_id": bond["bond_id"],
                        "step": step,
                        "start_date": start_date,
                        "end_date": end_date,
                        "accrual_end_date": accrual_end_date,
                        "pay_date": pay_date,
                        "accrual": accrual,
                        "coupon_type": "fixed" if is_fixed_period else "float",
                        "fixed_rate": coupon if is_fixed_period else np.nan,
                        "ref_curve": bond["ref_curve"],
                        "ref_tenor": bond["ref_tenor"],
                        "margin_dates_raw": margin_dates_raw,
                        "margin_values_raw": margin_values_raw,
                        "floor": bond["floor"],
                        "cap": bond["cap"],
                    }
                )
        return pd.DataFrame(rows)

    def build_reset_schedule(self) -> list[pd.Timestamp]:
        """
        Return the reset (fixing) dates of the bond, which are the start dates of its coupon
        periods.

        Only the first bond in df is used. An empty list is returned if df has no rows.

        Returns:
            list[pd.Timestamp]: Unadjusted coupon period start dates, oldest first.
        """
        for _, bond in self.df.iterrows():
            # Reset (fixing) dates are the coupon period starts, unadjusted: the rate for
            # a period is fixed on the day it starts accruing.
            return [start for start, _ in self._coupon_periods(bond)]

        return []

    def _coupon_periods(self, bond: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Unadjusted (start, end) coupon periods of one bond, on the grid of its rolling date.

        Call roll_periods upon a given bond.

        Args:
            - bond (pd.Series): One row of the term sheet.

        Returns:
            - list[tuple[pd.Timestamp, pd.Timestamp]]: See roll_periods.
        """
        issue_date = pd.to_datetime(bond["issue_date"])
        maturity_date = pd.to_datetime(bond["maturity_date"])
        months = int(round(float(bond["coupon_accrual"]) * 12)) # Number of months between coupon payments (e.g., 6 for semi-annual, 12 for annual)
        # rolling_date anchors the period grid; defaults to maturity_date when blank.
        has_rolling = "rolling_date" in bond.index and pd.notna(bond["rolling_date"]) and str(bond["rolling_date"]).strip()
        rolling_date = pd.to_datetime(bond["rolling_date"]) if has_rolling else maturity_date
        holidays = self.holiday_calendar.get(self.country, set())
        return roll_periods(issue_date, maturity_date, rolling_date, months, holidays)

def resolve_margin(
    margin_dates_raw: str | None,
    margin_values_raw: str | None,
    ref_date: pd.Timestamp,
) -> float | None:
    """
    Return the margin in effect on a given date from a term-sheet margin step schedule.

    The schedule holds n + 1 margin values separated by ';' and n change dates (mm/dd/yyyy)
    separated by ';'. The first value applies before the first change date; value i + 1 applies
    from change date i onwards.

    Args:
        margin_dates_raw (str | None): Margin change dates, ';'-separated; blank for a constant
            margin.
        margin_values_raw (str | None): Margin values, ';'-separated.
        ref_date (pd.Timestamp): Date at which the margin is needed (the fixing date).

    Returns:
        float | None: The margin in effect on ref_date, or None if no margin is given.

    Raises:
        ValueError: If the number of values is not the number of change dates plus one.
    """
    if margin_values_raw is None or pd.isna(margin_values_raw) or not str(margin_values_raw).strip():
        return None
    values = [float(x) for x in str(margin_values_raw).split(";")]
    if margin_dates_raw is None or pd.isna(margin_dates_raw) or not str(margin_dates_raw).strip():
        return values[0]
    mdates = [pd.to_datetime(x.strip(), format="%m/%d/%Y") for x in str(margin_dates_raw).split(";")]
    if len(values) != len(mdates) + 1:
        raise ValueError(
            f"margin có {len(values)} giá trị nhưng margin_date có {len(mdates)} mốc"
        )
    margin = values[0]
    for i, d in enumerate(mdates):
        if ref_date >= d:
            margin = values[i + 1]
        else:
            break
    return margin

def days(d:date, start_date:date) -> int:
    """
    Return the number of calendar days from start_date to d.

    Args:
        d (date): End date.
        start_date (date): Start date.

    Returns:
        int: (d - start_date) in days; negative if d is before start_date.
    """
    return(d-start_date).days
def build(
        rpd: date,
        face: float,
        maturity_date: pd.Timestamp,
        coupon_schedule_df: pd.DataFrame,
        ref_dates: list[date] | None = None,
        fixing_lag_days: float |None = None,
        call_df: pd.DataFrame | None = None,
        put_df: pd.DataFrame | None = None,
        ref_curve_name: str | None = None,
        curve_folder: str | None = None,
        ref_convention: str | None = None,
        apply_floor: bool = True,
        apply_cap: bool = True,
        ref_df_raw: pd.DataFrame|None=None,
) -> BondSchedule:
    """
    Build the callput BondSchedule of one bond as seen from the valuation date rpd.

    All dates are converted to day counts from rpd. Periods paying on or before rpd are dropped.
    Each remaining period becomes a CouponPeriod of one of three kinds:
        - Fixed period: pays its fixed_rate.
        - Floating period already fixed (fixing date <= rpd): the reference rate is looked up in
          ref_df_raw on the fixing date, converted from continuous to simple compounding over the
          reference tenor, the margin is added and the floor/cap are applied; the result is
          emitted as a fixed rate.
        - Floating period fixed in the future: emitted with its fixing day, reference tenor,
          margin, floor and cap so that the tree projects the rate.
    Call/put dates on or before rpd are dropped with a printed warning; rows with a missing
    strike are skipped.

    Args:
        rpd (date): Valuation (report) date.
        face (float): Face value.
        maturity_date (pd.Timestamp): Maturity date.
        coupon_schedule_df (pd.DataFrame): Coupon schedule of the bond, as returned by
            CouponSchedule.build_coupon_schedule_df.
        ref_dates (list[date] | None): Reset dates (see CouponSchedule.build_reset_schedule). A
            floating period uses the latest reset on or before its unadjusted start date.
        fixing_lag_days (float | None): Days between the reset date and the fixing date; None or
            0 fixes on the reset date itself.
        call_df (pd.DataFrame | None): Call schedule with columns call_date and call_strike.
        put_df (pd.DataFrame | None): Put schedule with columns put_date and put_strike.
        ref_curve_name (str | None): Name of the reference curve; required when a period has
            already been fixed.
        curve_folder (str | None): Folder of the reference curve; required when a period has
            already been fixed.
        ref_convention (str | None): Day count convention of the reference curve; required when
            a period has already been fixed.
        apply_floor (bool): Whether to apply the coupon floor.
        apply_cap (bool): Whether to apply the coupon cap.
        ref_df_raw (pd.DataFrame | None): Historical reference curve zero rates, used to look up
            the rate of periods already fixed.

    Returns:
        BondSchedule: Face, maturity day, coupon periods and call/put strikes keyed by day
        from rpd.

    Raises:
        ValueError: If no coupon period pays after rpd, if a floating period has no reset date
            on or before its start, or if a past fixing needs a curve lookup but
            curve_folder/ref_curve_name/ref_convention is missing.
    """
    sched_all =coupon_schedule_df.sort_values("pay_date")
    pays_all = sched_all["pay_date"].tolist()
    # Coupon amount: days between the unadjusted start and accrual end (see
    # CouponSchedule.build_coupon_schedule_df).
    accruals_all = ((sched_all["accrual_end_date"] - sched_all["start_date"]).dt.days / 365.0).tolist()
    # Where the period sits in the tree: callput needs contiguous periods, so each
    # one accrues from the previous pay date (the first from its own start).
    starts_all = [sched_all["start_date"].iloc[0]] + pays_all[:-1]

    # The unadjusted start picks the fixing: the next period's start can fall before
    # this period's adjusted pay date, so comparing resets against pay_date would
    # hand this period the next period's fixing.
    unadj_starts_all = sched_all["start_date"].tolist()

    future = [row for row in zip(starts_all, pays_all, accruals_all, unadj_starts_all)
              if row[1] > rpd]
    if not future:
        raise ValueError(f"No future coupon periods after rpd={rpd}")
    starts, pays, accruals, unadj_starts = map(list, zip(*future))

    resets = ref_dates or []    
    type_map = coupon_schedule_df.set_index('pay_date')['coupon_type']
    fixed_rate_map = coupon_schedule_df.set_index('pay_date')['fixed_rate']
    margin_dates_map = coupon_schedule_df.set_index('pay_date')['margin_dates_raw']
    margin_values_map = coupon_schedule_df.set_index('pay_date')['margin_values_raw']
    floor_map = coupon_schedule_df.set_index('pay_date')['floor']
    cap_map = coupon_schedule_df.set_index('pay_date')['cap']
    tenor_map = coupon_schedule_df.set_index('pay_date')['ref_tenor']


    periods = []
    for pay, start, accrual, unadj_start in zip(pays, starts, accruals, unadj_starts):
        is_floating_period = type_map[pay] == 'float'

        if not is_floating_period:
            periods.append(
                CouponPeriod(
                    pay_day=days(pay, rpd),
                    accrual=accrual,
                    accrual_start_day=days(start, rpd),
                    fixed_rate=float(fixed_rate_map[pay]),
                    fixing_day=None,
                    margin=0.0,
                    # ref_tenor_years=tenor_to_years(tenor_map[pay]),
                    floor=None,
                    cap=None,
                )
            )
            continue

        candidates = [r for r in resets if r <= unadj_start] # Fixed when the period starts
        if not candidates:
            raise ValueError(f"No fixing date found for payment date {pay}")

        reset_selected = max(candidates)      # The actual reset date, correctly selected
        fixing = (reset_selected - pd.Timedelta(days=fixing_lag_days) if fixing_lag_days else reset_selected)

        if fixing <= rpd:
            if curve_folder is None or ref_curve_name is None or ref_convention is None:
                raise ValueError(
                    f"Period with fixing {fixing} <= rpd {rpd} requires "
                    f"curve_folder/ref_curve_name/ref_convention to look up the known rate"
                )
            margin_known = resolve_margin(margin_dates_map[pay], margin_values_map[pay], fixing)
            known_rate = get_known_rate(
                ref_tenor=tenor_map[pay],
                fixing_date=fixing,
                margin=0.0,  # Margin not added yet
                ref_df=ref_df_raw,
                convention=ref_convention,
            )

            ref_delta = tenor_to_years(tenor_map[pay])

            # Continuous reference rate -> simple reference rate
            known_rate = (
                np.exp(known_rate * ref_delta) - 1.0
            ) / ref_delta

            # Add margin after conversion
            known_rate += margin_known
            # The floor/cap must ALSO apply to periods already fixed: contractually the fixed
            # coupon is min(max(L + margin, floor), cap). This branch used to emit
            # floor=None, cap=None, so the floor was dropped, undervaluing the bond whenever
            # the reference rate fell below the floor, with no warning at all.
            known_floor = nan_to_none(floor_map[pay]) if apply_floor else None
            known_cap = nan_to_none(cap_map[pay]) if apply_cap else None
            if known_floor is not None:
                known_rate = max(known_rate, known_floor)
            if known_cap is not None:
                known_rate = min(known_rate, known_cap)
            periods.append(
                CouponPeriod(
                    pay_day=days(pay, rpd),
                    accrual=accrual,
                    accrual_start_day=days(start, rpd),
                    fixed_rate=known_rate,
                    fixing_day=None,
                    margin=0.0,
                    ref_tenor_years=tenor_to_years(tenor_map[pay]),
                    floor=None,
                    cap=None,
                )
            )
            continue

        margin = resolve_margin(margin_dates_map[pay], margin_values_map[pay], fixing)
        periods.append(
            CouponPeriod(
                pay_day=days(pay, rpd),
                accrual=accrual,
                accrual_start_day=days(start, rpd),
                fixing_day=days(fixing, rpd),
                ref_tenor_years=tenor_to_years(tenor_map[pay]),
                margin=margin if margin is not None else 0.0,
                floor=nan_to_none(floor_map[pay]) if apply_floor else None,
                cap=nan_to_none(cap_map[pay]) if apply_cap else None,
            )
        )

    dropped_calls = call_df[call_df['call_date'] <= rpd] if call_df is not None and not call_df.empty else None
    if dropped_calls is not None and not dropped_calls.empty:
        print(f"[{None}] Warning: {len(dropped_calls)} call date(s) already in the past, dropped: {dropped_calls['call_date'].tolist()}")

    dropped_puts = put_df[put_df['put_date'] <= rpd] if put_df is not None and not put_df.empty else None
    if dropped_puts is not None and not dropped_puts.empty:
        print(f"Warning: {len(dropped_puts)} put date(s) already in the past, dropped: {dropped_puts['put_date'].tolist()}")

    call = (
        {days(d, rpd): s for d, s in zip(call_df['call_date'], call_df['call_strike']) if d > rpd and pd.notna(s)}
        if call_df is not None and not call_df.empty
        else {}
    )
    put = (
        {days(d, rpd): s for d, s in zip(put_df['put_date'], put_df['put_strike']) if d > rpd and pd.notna(s)}
        if put_df is not None and not put_df.empty
        else {}
    )

    return BondSchedule(
        face=face,
        maturity_day=days(maturity_date, rpd),
        periods=periods,
        call=call,
        put=put,
    )