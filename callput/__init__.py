"""VCB IRRBB call/put embedded-option pricer.

Two engines over one lattice:

``CallPutTree.single_curve``
    One Hull-White factor, one curve.  Fixed-rate bonds, step-ups included.

``CallPutTree.multi_curve``
    Two correlated factors -- discounting and the coupon index.  Any floating
    coupon, including a bond fixed for its first few years, and caps/floors on
    the coupon rate.

The time grid is expressed in **integer days** from the valuation date and is
anchored on the contract's own dates, so coupon, reset and exercise dates land
exactly on nodes.
"""

from .curve import YieldCurve
from .lattice import FactorLattice, Lattice
from .leg import ACT_FAMILY, CurveLeg, hw_B, hw_G, ou_variance
from .schedule import (
    BondSchedule,
    CompiledBond,
    CouponPeriod,
    FixingGroup,
    Period,
    build_day_grid,
    compile_bond,
)
from .tree import CallPutTree, PricingFlags

__all__ = [
    "YieldCurve",
    "CurveLeg",
    "ACT_FAMILY",
    "hw_B",
    "hw_G",
    "ou_variance",
    "CouponPeriod",
    "BondSchedule",
    "CompiledBond",
    "FixingGroup",
    "Period",
    "build_day_grid",
    "compile_bond",
    "Lattice",
    "FactorLattice",
    "CallPutTree",
    "PricingFlags",
]
