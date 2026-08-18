# from .curr_swap import CurrSwap
from .rate_index import RateIndex
# from .fx_swap_instrument import FXSwapInstrument


def create_instru(instru_type: str, instru_name: str, **kwargs):
    """Factory function to create financial instrument instances.
    
    This function creates and returns instances of various financial instruments
    based on the specified instrument type. It supports rate indices, FX swaps,
    and currency swaps.
    
    Args:
        instru_type (str): Type of instrument to create. Supported types:
            - 'currency': Currency instrument (currently not implemented)
            - 'rate_index': Rate index instrument
            - 'fx_swap': Foreign exchange swap instrument  
            - 'curr_swap': Currency swap instrument
        instru_name (str): Name of the specific instrument to create
        **kwargs: Additional keyword arguments passed to instrument constructors
        
    Returns:
        BaseInstrument: An instance of the requested instrument type
        
    Raises:
        ValueError: If instru_type is not recognized
        
    Examples:
        >>> rate_idx = create_instru('rate_index', 'SOFR')
        >>> fx_swap = create_instru('fx_swap', 'USDVND_1M')
        >>> curr_swap = create_instru('curr_swap', 'VND_IRS_5Y')
    """
    match instru_type:
        case "currency":
            pass
        case "rate_index":
            return RateIndex(rate_index_name=instru_name)
        # case "fx_swap":
        #     return FXSwapInstrument(fx_swap_name=instru_name)
        # case "curr_swap":
        #     return CurrSwap(curr_swap_name=instru_name)
        case _:
            raise ValueError(f"Unknown instrument type: {instru_type}")
