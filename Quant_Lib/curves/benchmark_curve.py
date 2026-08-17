from Quant_Lib.curves.curve import Curve
import pandas as pd


class BenchmarkCurve(Curve):
    """Benchmark curve implementation that inherits common functionality from Curve."""

    _curve_type = "BenchmarkCurve"

    def __init__(
        self,
        curve_name: str,
        benchmark_price: pd.DataFrame = None,
        dependency_curve: Curve | None = None,
        _skip_init: bool = False,
    ):
        self._dependency_curve_param = dependency_curve
        super().__init__(curve_name, benchmark_price, _skip_init=_skip_init)

    def _initialize_dependencies(self):
        """Initialize dependency curve if provided."""
        if self._dependency_curve_param is not None:
            if self._dependency_curve_param._curve_factors is None:
                raise ValueError(
                    "Dependency's curve factors must be calculated before using it as a dependency."
                )
            self._dependency_curve = self._dependency_curve_param

    def _finalize_initialization(self):
        """Finalize initialization by calculating curve factors."""
        self.get_curve_factors()
