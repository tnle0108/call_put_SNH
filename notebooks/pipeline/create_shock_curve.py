#%%
from pathlib import Path
from math import gcd
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, str(Path.cwd().parents[1]))
from src.shock import ShockScenario

groups = ['LB_G1', 'LB_G2', 'LB_G3', 'LB_G4','NBFI', 'FB']
shocks = ['1', '2','3','4','5','6']


root = Path.cwd().resolve().parent.parent
BOND_DATA_PATH = root / 'datasets' /'raw'/ 'bonds_placeholder.csv'
CURVE_DATA_PATH = root / 'datasets' / 'raw' / 'Histocopy_FI_ZYC_VND_GD2_family.xlsx'
START_DATE = pd.to_datetime('2024-06-24')
END_DATE = pd.to_datetime('2026-07-30')
REPORT_DATE = pd.to_datetime('2026-06-03')
DISC_CONVENTION = 'actactisda'
COUP_CONVENTION = 'act365'

for shock in shocks:

    output_file = root / 'datasets' / 'shock' / f"shock_{shock}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for group in groups:
            df = pd.read_excel(
                CURVE_DATA_PATH,
                sheet_name=f"FI ZYC VND_GD2_{group}",
                index_col=0,
            )
            shock_df = (
                ShockScenario(
                    shock_type=shock,
                    df=df,
                )
                .create_shock_df()
            )

            shock_df.to_excel(
                writer,
                sheet_name=group,
            )

    print(f"Saved: {output_file}")
#%%

for group in groups:
    df = pd.read_excel (CURVE_DATA_PATH, sheet_name=f'FI ZYC VND_GD2_{group}', index_col=0)
    for shock in shocks:
        shock_df = ShockScenario(shock_type=shock, df=df).create_shock_df()
        