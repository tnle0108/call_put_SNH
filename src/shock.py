#%%
import pandas as pd
import numpy as np
import sys 
from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar
sys.path.insert(0, str(Path.cwd().parents[1]))


@dataclass
class ShockScenario:
    """
    Interest rate shock scenarios applied to a table of zero rates.

    Shock sizes scale with R_avg, the mean over the 3M, 6M, 1Y, 2Y, 5Y, 7Y, 10Y, 15Y and 20Y
    columns (those present) of each column's average rate, with t the tenor in years:
        - Parallel: clamp(0.6 * R_avg, 1%, 4%).
        - Short: clamp(0.85 * R_avg, 1%, 5%) * exp(-t / 4).
        - Long: clamp(0.4 * R_avg, 1%, 3%) * (1 - exp(-t / 4)).

    Attributes:
        shock_type (str): Scenario code: "1" parallel up, "2" parallel down, "3" steepener,
            "4" flattener, "5" short rates up, "6" short rates down.
        df (pd.DataFrame): Zero rates, one column per tenor. A column's tenor is the label after
            its last '_' ('5Y' and 'FI_ZYC_VND_LB_G1_5Y' are both 5Y). A "Date" column, if
            present, is left untouched.
        map_tenor (dict[str, float]): Tenor label to year fraction.
        R_AVG_TENORS (tuple[str, ...]): Tenors averaged into R_avg (TT83 / MB-01 §2.3.1).
        R_avg (float): Computed once from df when the scenario is created.
    """
    shock_type: str
    df: pd.DataFrame

    R_AVG_TENORS: ClassVar[tuple[str, ...]] = ("3M", "6M", "1Y", "2Y", "5Y", "7Y", "10Y", "15Y", "20Y")

    map_tenor: ClassVar[dict[str, float]] = {
        "ON": 1/365,
        "1W": 1/52, "2W": 2/52,
        "1M": 1/12, "2M": 2/12,
        "3M": 3/12, "6M": 6/12, "9M": 9/12,
        "1Y": 1.0, "15M": 15/12, "18M": 18/12, "21M": 21/12,
        "2Y": 2.0, "27M": 27/12, "30M": 30/12, "33M": 33/12,
        "3Y": 3.0, "5Y": 5.0,
        "4Y": 4.0,
        "7Y": 7.0, "10Y": 10.0, "15Y": 15.0, "20Y": 20.0, "30Y": 30.0
    }

    def __post_init__(self):
        """Compute R_avg once; it does not depend on the tenor being shocked."""
        self.R_avg = self.calc_R_avg()

    def calc_R_avg(self) -> float:
        """
        Average rate over the R_AVG_TENORS columns that df has.

        Tenors are matched by the label after the last '_', the same rule calc_delta_R and
        MapCurve use, so prefixed column names work. A missing tenor is skipped, as the rule
        allows ("if available"; e.g. SOB4 stops at 5Y), but finding none of them is an error:
        before, it silently gave R_avg = NaN and an all-NaN shocked curve.

        Returns:
            float: Mean of the per-tenor averages, in R_AVG_TENORS order.

        Raises:
            ValueError: If df has none of the R_AVG_TENORS.
        """
        column_of = {str(c).split("_")[-1]: c for c in self.df.columns if c != "Date"}
        means = [self.calc_R_average_for_each_tenor(column_of[tn])
                 for tn in self.R_AVG_TENORS if tn in column_of]
        if not means:
            raise ValueError(
                f"none of the R_avg tenors {self.R_AVG_TENORS} found in columns "
                f"{list(self.df.columns)}"
            )
        return float(np.mean(means))

    def calc_R_average_for_each_tenor(self, tenor: str):
        """
        Calculate the average rate of one tenor column of df.

        Args:
            tenor (str): Column name of the tenor.

        Returns:
            float: Mean of the column over all dates.

        Raises:
            ValueError: If the column is not in df.
        """
        if tenor not in self.df.columns:
            raise ValueError(f"Tenor '{tenor}' not found in DataFrame.")
        R_avg = float(self.df[tenor].mean())
        return R_avg

        
    def calc_delta_R (self, tenor:str):
        """
        Calculate the rate shock of one tenor under shock_type.

        Steepener: -0.65 * |short| + 0.9 * |long|. Flattener: 0.8 * |short| - 0.6 * |long|. The
        other scenarios use the parallel or short shock with the sign of the scenario (see the
        class docstring).

        Args:
            tenor (str): Column name; the tenor label after the last '_' must be in map_tenor.

        Returns:
            float: The shock added to the zero rate of that tenor.

        Raises:
            ValueError: If shock_type is not one of '1' to '6'.
        """
        tenor_label = tenor.split("_")[-1]
        t_k = self.map_tenor[tenor_label]
        R_avg = self.R_avg

        delta_R_short_horz = max(min(R_avg * 0.85,0.05), 0.01)
        delta_R_short = delta_R_short_horz * np.exp(-t_k/4)
        delta_R_long_horz = max(min(R_avg * 0.4, 0.03),0.01)
        delta_R_long = delta_R_long_horz * (1-np.exp(-t_k/4))

        if self.shock_type in ["1","2"]: #parallel
            if self.shock_type == "1": #parallel up
                delta_R = max(min(R_avg * 0.6, 0.04), 0.01)
            elif self.shock_type == "2": #parallel down
                delta_R = - max(min(R_avg * 0.6, 0.04), 0.01)

        elif self.shock_type in ["3"]: #steepener
            delta_R = -0.65 * abs(delta_R_short) + 0.9 * abs(delta_R_long)

        elif self.shock_type in ["4"]: #flattener
            delta_R = 0.8 * abs(delta_R_short) - 0.6 * abs(delta_R_long)

        elif self.shock_type in ["5"]: #short rates shock up
            delta_R = delta_R_short
        
        elif self.shock_type in ["6"]: #short rates shock down
            delta_R = - delta_R_short

        else:
            raise ValueError(f"Invalid shock_type: {self.shock_type}. Must be one of ['1', '2', '3', '4', '5', '6'].")

        return delta_R


    def create_shock_df(self):
        """
        Apply the shock to every tenor column of df.

        Returns:
            pd.DataFrame: A copy of df with each tenor column shifted by its calc_delta_R; a
            "Date" column is left as is.
        """
        shock_df = self.df.copy()
        for tenor in self.df.columns:
            if tenor == "Date":
                continue
            delta_R = self.calc_delta_R(tenor)
            shock_df[tenor] += delta_R
        return shock_df