import pandas as pd
from Quant_Lib.configs import CURRENCY_PAIRS


class CurrencyPair:
    """Represents a currency pair with configuration-based initialization.
    
    This class loads configuration from CURRENCY_PAIRS config and provides
    methods to generate DataFrames from price series for currency pair analysis.
    
    The configuration is dynamically loaded and stored as private attributes
    with underscore prefixes.
    """
    
    def __init__(self, currency_pair_name: str):
        """Initialize a CurrencyPair instance with configuration.
        
        Args:
            currency_pair_name (str): Name of the currency pair to load from CURRENCY_PAIRS config
        
        Raises:
            KeyError: If currency_pair_name is not found in CURRENCY_PAIRS config
        """
        config = CURRENCY_PAIRS[currency_pair_name]
        for key, value in config.items():
            setattr(self, f"_{key}", value)

    def df(self, price: pd.Series):
        """Generate DataFrame for currency pair prices."""
        if not isinstance(price, pd.Series):
            raise TypeError("Price must be a pandas Series")
        if not isinstance(price.index, pd.DatetimeIndex):
            raise TypeError("Price index must be a pandas DatetimeIndex")

        df = pd.DataFrame(index=price.index, columns=["price"])
        df["price"] = price

        self._df = df
        return df
