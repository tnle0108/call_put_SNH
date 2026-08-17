class Currency:
    """Represents a currency with its ISO code, full name, and symbol.
    
    This class provides a simple data structure to store currency information
    and access currency properties through private attributes.
    
    Attributes:
        _code (str): ISO currency code (e.g., 'USD', 'EUR', 'VND')
        _name (str): Full currency name (e.g., 'US Dollar', 'Vietnamese Dong')
        _symbol (str): Currency symbol (e.g., '$', '€', '₫')
    """
    
    def __init__(self, code: str, name: str, symbol: str):
        """Initialize a Currency instance.
        
        Args:
            code (str): ISO currency code (e.g., 'USD', 'EUR', 'VND')
            name (str): Full currency name (e.g., 'US Dollar', 'Vietnamese Dong') 
            symbol (str): Currency symbol (e.g., '$', '€', '₫')
        """
        self._code = code
        self._name = name
        self._symbol = symbol
