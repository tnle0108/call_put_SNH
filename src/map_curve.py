import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from dataclasses import dataclass


sys.path.insert(0, str(Path.cwd().parents[0]))

from src.daycount import DayCount
from callput import YieldCurve


@dataclass
class MapCurve:
    # df: pd.DataFrame
    rpd: pd.Timestamp
    df: pd.DataFrame
    convention: str

    def __post_init__(self):
        self.rpd = pd.Timestamp(self.rpd)
    
    def map_curve(self):

        tenor_labels = [col.split("_")[-1] for col in self.df.columns]
        maturity_dates = []

        for tenor in tenor_labels:
            if tenor.endswith("Y"):
                n = int(tenor[:-1])
                maturity_dates.append(self.rpd + pd.DateOffset(years=n))
            elif tenor.endswith("M"):
                n = int(tenor[:-1])
                maturity_dates.append(self.rpd + pd.DateOffset(months=n))
            elif tenor.endswith("N"):
                n = 1
                maturity_dates.append(self.rpd + pd.DateOffset(days=n))
            elif tenor.endswith("W"):
                n = int(tenor[:-1])
                maturity_dates.append(self.rpd + pd.DateOffset(weeks=n))
            else:
                raise ValueError(f"Unsupported tenor: {tenor}")

        start = np.array([self.rpd] * len(maturity_dates), dtype="datetime64[D]")
        end = np.array(maturity_dates, dtype="datetime64[D]")
        maturities = DayCount.get(self.convention).yearfrac(start, end)

        # valid_idx = self.df.index[self.df.index <= self.rpd]
        # if len(valid_idx) == 0:
        #     raise ValueError(
        #         f"No curve date <= {self.rpd}"
        #     )

        # report_idx = valid_idx.max()
        valid_idx = self.df.index[self.df.index <= self.rpd]

        report_idx = (
            valid_idx.max()
            if len(valid_idx) > 0
            else self.df.index.min()
        )
        zero_rates = self.df.loc[report_idx].to_numpy(dtype=float)

        curve = YieldCurve.from_zero_rates(
            maturities = maturities,
            zero_rates = zero_rates,
        )

        return curve
