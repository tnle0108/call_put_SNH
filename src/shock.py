#%%
import pandas as pd
import numpy as np
import sys 
from pathlib import Path
from dataclasses import dataclass, field
from typing import ClassVar
sys.path.insert(0, str(Path.cwd().parents[1]))


from src.daycount import DayCount


@dataclass
class ShockScenario:
    shock_type: str
    df: pd.DataFrame

    map_tenor: ClassVar[dict[str, float]] = {
        "3M": 3/12, "6M": 6/12, "9M": 9/12,
        "1Y": 1.0, "15M": 15/12, "18M": 18/12, "21M": 21/12,
        "2Y": 2.0, "27M": 27/12, "30M": 30/12, "33M": 33/12,
        "3Y": 3.0, "5Y": 5.0,
    }

    def calc_R_average_for_each_tenor(self, tenor: str):
        '''calculate R average for each tenor of df'''
        if tenor not in self.df.columns:
            raise ValueError(f"Tenor '{tenor}' not found in DataFrame.")
        R_avg = float(self.df[tenor].mean())
        if R_avg < 0.01:
            R_avg = 0.01
        return R_avg
        

    def calc_delta_R (self, tenor:str):
        tenor_label = tenor.split("_")[-1]
        t_k = self.map_tenor[tenor_label]
        R_avg = self.calc_R_average_for_each_tenor(tenor)

        delta_R_short_horz = 0.85 * min(R_avg,0.05)
        delta_R_short = delta_R_short_horz * np.exp(-t_k/4)
        delta_R_long_horz = 0.4 * min(R_avg, 0.03)
        delta_R_long = delta_R_long_horz * (1-np.exp(-t_k/4))

        if self.shock_type in ["1","2"]: #parallel
            if self.shock_type == "1": #parallel up
                delta_R = 0.6 * min(R_avg, 0.04)
            elif self.shock_type == "2": #parallel down
                delta_R = -0.6 * min(R_avg, 0.04)

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
        shock_df = self.df.copy()
        for tenor in self.df.columns:
            if tenor == "Date":
                continue
            delta_R = self.calc_delta_R(tenor)
            shock_df[tenor] += delta_R
        return shock_df