#%%
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from Quant_Lib.curves import BenchmarkCurve

SOURCE_PATH = "../datasets/curve/vbma_bond_fi.csv"
# START_DATE = None
# END_DATE = None
START_DATE = pd.to_datetime("2026-03-02")
END_DATE = pd.to_datetime("2026-03-02")

vbma_bond_fi = pd.read_csv(SOURCE_PATH, index_col=0)
vbma_bond_fi.index = pd.to_datetime(vbma_bond_fi.index)

vbma_bond_fi = vbma_bond_fi.loc[vbma_bond_fi.index >= START_DATE] if START_DATE is not None else vbma_bond_fi
vbma_bond_fi = vbma_bond_fi.loc[vbma_bond_fi.index <= END_DATE] if END_DATE is not None else vbma_bond_fi

#%%
# vbma_bond_fi.loc["2026-03-31"]
#%%
fi_zyc = BenchmarkCurve(curve_name="FI ZYC VND", benchmark_price=vbma_bond_fi)
#%%
fi_zyc_vnd = fi_zyc.print_curve(np.datetime64("2026-03-02"))
print(fi_zyc_vnd['value'])
#%%
# fi_zyc_vnd_wide = fi_zyc_vnd["value"].to_frame().T
# fi_zyc_vnd_wide.index = [fi_zyc_vnd["used_date"].iloc[0]]
# fi_zyc_vnd_wide.index.name = "Date"
# tenor_order = list(TENOR_MONTHS.keys())

# fi_zyc_vnd_wide = (
#     fi_zyc_vnd
#     .loc[tenor_order, ["used_date", "value"]]
#     .set_index("used_date")["value"]
#     .to_frame()
#     .T
# )

# fi_zyc_vnd_wide.index.name = "Date"