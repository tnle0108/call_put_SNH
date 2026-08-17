import pandas as pd
from Quant_Lib.curves.curve import Curve
from Quant_Lib.curves.benchmark_curve import BenchmarkCurve


class SpreadCurve(Curve):
    """Spread curve implementation that combines base curve with spread."""
    
    _curve_type = "SpreadCurve"

    def __init__(
        self, curve_name: str, base_curve: BenchmarkCurve, benchmark_price: pd.DataFrame = None
    ):
        self._base_curve_param = base_curve
        super().__init__(curve_name, benchmark_price)

    def _initialize_dependencies(self):
        """Initialize spread curve as an internal BenchmarkCurve."""
        # Restore the base_curve from parameter (config may have overwritten it)
        self._base_curve = self._base_curve_param
        
        # Create a lightweight BenchmarkCurve that shares benchmark objects
        self._spread_curve = BenchmarkCurve(curve_name=self._curve_name, _skip_init=True)
        self._spread_curve._curve_type = "SpreadCurve"
        
        # Share the same config and benchmark objects to avoid duplication
        self._spread_curve._benchmarks = self._benchmarks
        self._spread_curve._day_count = self._day_count
        self._spread_curve._extrapol_short_end = self._extrapol_short_end
        self._spread_curve._extrapol_long_end = self._extrapol_long_end
        self._spread_curve._benchmark_objects = self._benchmark_objects

    def _finalize_initialization(self):
        """Finalize initialization by calculating total curve factors."""
        # Share the same benchmark price with the internal spread curve
        self._spread_curve._benchmark_price = self._benchmark_price
        # No need to call input_benchmark_price_2_benchmark_objects since we share objects
        self._spread_curve.get_curve_factors()
        self.get_total_curve_factors()

    def get_total_curve_factors(self):
        if self._base_curve.day_count != self._spread_curve.day_count:
            raise NotImplementedError(
                "Case where base curve and spread curve have different day count conventions is not implemented yet."
            )
        curve_total_factors = {}
        for rpd in self._benchmark_price.index:
            if rpd in self._base_curve._curve_factors:

                def curve_total_factor(
                    t: float,
                    base_curve_factor=self._base_curve._curve_factors[rpd],
                    spread_curve_factor=self._spread_curve._curve_factors[rpd],
                ):
                    return base_curve_factor(t) + spread_curve_factor(t)

                curve_total_factors[rpd] = curve_total_factor

        self._curve_factors = curve_total_factors
        return curve_total_factors
