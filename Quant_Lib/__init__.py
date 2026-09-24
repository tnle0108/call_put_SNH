"""
Quant_Lib - Market Risk Quantitative Library v1.0.0

Professional quantitative finance library for market risk management and analysis.
Developed by Hai Anh Nguyen.

This package provides comprehensive tools for:
- Financial instrument modeling and valuation
- Yield curve construction and operations  
- Volatility surface modeling
- Market data processing and analysis
- Risk factor calculations

Copyright (c) 2026 Hai Anh Nguyen. All rights reserved.
"""

__version__ = "1.0.0"
__author__ = "Hai Anh Nguyen"
__email__ = "nhaianh2208@gmail.com"

# Import main modules for easy access
from . import instrus
from . import helpers  
from . import curves
from . import configs

# Version info
VERSION = __version__
