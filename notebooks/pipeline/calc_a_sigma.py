#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))
from quantmr.model.shortrate.hullwhite import HullWhite

hw = HullWhite.get("zc usd sr")
result = hw.calibrate(
    "zc usd sr",
    method="kfmh",
    save=True,
)
print(f"Calibrated a: {result['a']:.4f}, sigma: {result['sigma']:.4f}")
#%%