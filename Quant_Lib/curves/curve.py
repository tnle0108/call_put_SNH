import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from Quant_Lib.instrus import create_instru
from Quant_Lib.configs import CURVES
from Quant_Lib.curves.curve_bootstrap import CurveBootstrapMixin
from Quant_Lib.curves.curve_operations import CurveOperationsMixin


class Curve(CurveBootstrapMixin, CurveOperationsMixin, ABC):
    """Abstract base class for all curve types with common functionality."""

    _curve_type: str = None  # Must be set by subclasses
    _benchmark_objects_cache = {}  # Class-level cache for benchmark objects by curve_name

    def __init__(
        self, curve_name: str, benchmark_price: pd.DataFrame = None, _skip_init: bool = False
    ):
        """Initialize curve with configuration from CURVES.

        Args:
            curve_name: Name of the curve to load configuration for
            benchmark_price: Optional benchmark price data
            _skip_init: Internal parameter to skip standard initialization
        """
        self._curve_name = curve_name
        if not _skip_init:
            self._load_config()
            self.create_benchmark_objects()
            self._initialize_dependencies()

            if benchmark_price is not None:
                self.benchmark_price = benchmark_price
                self._finalize_initialization()

    def _load_config(self):
        """Load curve configuration from CURVES config."""
        config = CURVES[self._curve_name]
        for key, value in config.items():
            setattr(self, f"_{key}", value)

    @abstractmethod
    def _initialize_dependencies(self):
        """Initialize any curve dependencies. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _finalize_initialization(self):
        """Finalize initialization after benchmark price is set. Must be implemented by subclasses."""
        pass

    # Properties
    @property
    def curve_factors(self):
        """Get curve factors dictionary."""
        if not hasattr(self, "_curve_factors") or self._curve_factors is None:
            raise ValueError(
                "Curve factors have not been calculated yet. Call get_curve_factors() first."
            )
        return self._curve_factors

    @property
    def benchmark_objects(self):
        """Get benchmark objects dictionary."""
        if not hasattr(self, "_benchmark_objects"):
            raise ValueError(
                "Benchmark objects have not been created yet. Call create_benchmark_objects() first."
            )
        return self._benchmark_objects

    @property
    def benchmark_price(self):
        """Get benchmark price DataFrame."""
        if not hasattr(self, "_benchmark_price") or self._benchmark_price is None:
            raise ValueError("Benchmark price has not been set yet.")
        return self._benchmark_price

    @benchmark_price.setter
    def benchmark_price(self, value: pd.DataFrame):
        """Set benchmark price DataFrame and input to benchmark objects."""
        self._benchmark_price = value
        self._set_benchmark_prices(value)

    @property
    def curve_name(self):
        """Get curve name."""
        return self._curve_name

    @property
    def day_count(self):
        """Get day count convention."""
        return self._day_count

    # Benchmark management methods
    @classmethod
    def clear_benchmark_cache(cls):
        """Clear the benchmark objects cache. Useful for testing or memory management."""
        cls._benchmark_objects_cache.clear()

    def create_benchmark_objects(self):
        """Create or retrieve cached benchmark objects."""
        if not hasattr(self, "_benchmarks"):
            raise ValueError("Missing benchmarks in curve configuration.")

        # Check cache first
        cache_key = (self._curve_name, tuple(tuple(b) for b in self._benchmarks))
        if cache_key in Curve._benchmark_objects_cache:
            self._benchmark_objects = Curve._benchmark_objects_cache[cache_key]
            return self._benchmark_objects

        # Create new benchmark objects if not cached
        benchmark_objects = {}
        for benchmark in self._benchmarks:
            benchmark_type = benchmark[0]
            benchmark_name = benchmark[1]
            benchmark_objects[benchmark_name] = create_instru(benchmark_type, benchmark_name)

        # Cache for future use
        Curve._benchmark_objects_cache[cache_key] = benchmark_objects
        self._benchmark_objects = benchmark_objects
        return benchmark_objects

    def _set_benchmark_prices(self, benchmark_price: pd.DataFrame):
        """Set price data to all benchmark objects."""
        for benchmark_name, benchmark_object in self._benchmark_objects.items():
            if benchmark_name not in benchmark_price.columns:
                raise ValueError(f"Missing price data for benchmark: {benchmark_name}")
            price_series = benchmark_price[benchmark_name]
            if hasattr(benchmark_object, "_currency_pair_obj"):
                price_spot_series = benchmark_price[benchmark_object._currency_pair_name]
                benchmark_object._currency_pair_obj.df(price=price_spot_series)
            benchmark_object.df(price=price_series)
