import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Union
from Quant_Lib.helpers.date_utils.holiday_checker import HolidayChecker as hc


class BaseInstrument(ABC):
    """Abstract base class for all financial instruments."""

    def __init__(self, instrument_name: str, config_dict: dict):
        """Initialize instrument with configuration.

        Args:
            instrument_name: Name of the instrument
            config_dict: Configuration dictionary for this instrument
        """
        self._instrument_name = instrument_name
        self._load_config(config_dict)
        self._initialize_dependencies()

    def _load_config(self, config: dict):
        """Load configuration from config dictionary."""
        for key, value in config.items():
            setattr(self, f"_{key}", value)

    def _initialize_dependencies(self):
        """Initialize any dependencies (like currency pairs). Override if needed."""
        pass

    @abstractmethod
    def df(self, price: pd.Series) -> pd.DataFrame:
        """Calculate instrument DataFrame from price series.

        Args:
            price: Time series of instrument prices

        Returns:
            DataFrame with calculated values including functions for curve building
        """
        pass

    def _validate_price_input(self, price: pd.Series):
        """Validate price input is correct type."""
        if not isinstance(price, pd.Series):
            raise TypeError("price must be a pandas Series")
        if not isinstance(price.index, (pd.DatetimeIndex, np.ndarray)):
            # If it's an ndarray, check if it's datetime64 type
            if isinstance(price.index, np.ndarray) and not np.issubdtype(
                price.index.dtype, np.datetime64
            ):
                raise TypeError(
                    "price index must be a pandas DatetimeIndex or numpy datetime64 array"
                )

    def spot(self, rpds: Union[pd.DatetimeIndex, np.ndarray]) -> pd.Series:
        """Calculate spot dates from report dates.

        Args:
            rpds: Report dates (DatetimeIndex or numpy datetime64 array)

        Returns:
            Series of spot dates (datetime64[ns])

        Note: Override in subclass if needed
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement spot()")

    def start(self, spots: pd.Series) -> pd.Series:
        """Calculate start dates from spot dates.

        Args:
            spots: Spot dates

        Returns:
            Series of start dates

        Note: Override in subclass if needed
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement start()")

    def end(self, starts: pd.Series) -> pd.Series:
        """Calculate end dates from start dates.

        Args:
            starts: Start dates

        Returns:
            Series of end dates

        Note: Override in subclass if needed
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement end()")

    @abstractmethod
    def func(self, **kwargs) -> Callable:
        """Create pricing function for curve building.

        Returns:
            Callable function for use in curve bootstrapping
        """
        pass

    @property
    def instrument_name(self):
        """Get instrument name."""
        return self._instrument_name

    @property
    def dataframe(self):
        """Get calculated DataFrame."""
        if not hasattr(self, "_df"):
            raise ValueError("DataFrame not calculated yet. Call df() first.")
        return self._df
