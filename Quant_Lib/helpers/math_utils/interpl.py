import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from functools import lru_cache
import warnings


def warn_and_print_x_y(message, **kwargs):
    if "invalid value encountered in divide" in str(message):
        print("Warning: invalid value encountered in divide")
        print("x:", kwargs.get("x", []))
        print("y:", kwargs.get("y", []))
        if "z" in kwargs:
            print("z:", kwargs["z"])
    return warnings.defaultaction


def _interpolate_with_extrapolation_1d(x_sorted, y_sorted, x_star, short_end, long_end):
    """
    Helper function to perform interpolation with custom extrapolation for 1D data.

    Parameters:
    -----------
    x_sorted : np.ndarray
        Sorted x values
    y_sorted : np.ndarray
        Sorted y values corresponding to x_sorted
    x_star : float or np.ndarray
        Target x values
    short_end : str
        Extrapolation method for values below min(x)
    long_end : str
        Extrapolation method for values above max(x)

    Returns:
    --------
    float or np.ndarray
        Interpolated/extrapolated values
    """
    x_star_arr = np.atleast_1d(x_star)
    result = np.empty_like(x_star_arr, dtype=float)

    # Identify regions
    below_mask = x_star_arr < x_sorted[0]
    above_mask = x_star_arr > x_sorted[-1]
    within_mask = ~(below_mask | above_mask)

    # Interpolate within bounds
    if np.any(within_mask):
        f = interp1d(x_sorted, y_sorted, kind="linear", bounds_error=False)
        result[within_mask] = f(x_star_arr[within_mask])

    # Handle short end
    if np.any(below_mask):
        if short_end == "Flat":
            result[below_mask] = y_sorted[0]
        else:  # Default - linear extrapolation
            if len(x_sorted) >= 2:
                slope = (y_sorted[1] - y_sorted[0]) / (x_sorted[1] - x_sorted[0])
                result[below_mask] = y_sorted[0] + slope * (x_star_arr[below_mask] - x_sorted[0])
            else:
                result[below_mask] = y_sorted[0]

    # Handle long end
    if np.any(above_mask):
        if long_end == "Flat":
            result[above_mask] = y_sorted[-1]
        else:  # Default - linear extrapolation
            if len(x_sorted) >= 2:
                slope = (y_sorted[-1] - y_sorted[-2]) / (x_sorted[-1] - x_sorted[-2])
                result[above_mask] = y_sorted[-1] + slope * (x_star_arr[above_mask] - x_sorted[-1])
            else:
                result[above_mask] = y_sorted[-1]

    # Return scalar if input was scalar
    return result[0] if np.isscalar(x_star) else result


def _prepare_data_1d(x, y):
    """
    Prepare 1D data for interpolation: clean, sort, deduplicate.

    Returns tuple of (x_sorted, y_sorted) arrays ready for interpolation.
    Uses numpy operations for better performance than pandas when possible.
    """
    # Convert to numeric, handling NaN
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Remove NaN values using numpy (faster than pandas for simple operations)
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if len(x_clean) == 0:
        print("Original X:", x)
        print("Original Y:", y)
        raise ValueError("No valid data points after removing NaN values")

    # Sort by x
    sort_idx = np.argsort(x_clean)
    x_sorted = x_clean[sort_idx]
    y_sorted = y_clean[sort_idx]

    # Remove duplicates, keeping last - using numpy for efficiency
    if len(x_sorted) > 1:
        unique_mask = np.concatenate([[True], x_sorted[1:] != x_sorted[:-1]])
        # Keep last occurrence of duplicates
        if not np.all(unique_mask):
            # Find last occurrence of each unique value
            unique_vals, last_idx = np.unique(x_sorted[::-1], return_index=True)
            last_idx = len(x_sorted) - 1 - last_idx
            sort_last_idx = np.sort(last_idx)
            x_sorted = x_sorted[sort_last_idx]
            y_sorted = y_sorted[sort_last_idx]

    return x_sorted, y_sorted


def interpolate(
    x: float | np.ndarray[float],
    y: float | np.ndarray[float],
    x_star: float | np.ndarray[float],
    short_end: str = "Default",
    long_end: str = "Default",
):
    """
    Linear interpolation with configurable extrapolation.

    Optimized version with:
    - Efficient numpy operations for data preparation
    - Vectorized processing for 2D y arrays
    - Reduced pandas overhead for simple operations
    - Better memory efficiency
    - Configurable extrapolation methods

    Parameters:
    -----------
    x : float or array-like
        Known x values (1D)
    y : float or array-like
        Known y values (1D or 2D). If 2D, each column is interpolated separately.
    x_star : float or array-like
        Target x values for interpolation
    short_end : str, default="Default"
        Extrapolation method for values below min(x):
        - "Default": Linear extrapolation (extends the line)
        - "Flat": Flat extrapolation (uses y value at min(x))
    long_end : str, default="Default"
        Extrapolation method for values above max(x):
        - "Default": Linear extrapolation (extends the line)
        - "Flat": Flat extrapolation (uses y value at max(x))

    Returns:
    --------
    float or np.ndarray
        Interpolated values at x_star positions
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_star = np.asarray(x_star)

    if x.ndim != 1:
        raise ValueError("x must be a 1-dimensional array")

    y = np.asarray(y)

    if y.ndim == 1:
        # 1D interpolation
        if len(x) != len(y):
            raise ValueError(f"The length of x ({len(x)}) must match the length of y ({len(y)})")

        # Use optimized data preparation
        x_sorted, y_sorted = _prepare_data_1d(x, y)

        if x_sorted.shape[0] == 1:
            # Single point - return constant value
            return np.full_like(x_star, y_sorted[0], dtype=float)
        else:
            # Determine fill_value based on extrapolation settings
            if short_end == "Default" and long_end == "Default":
                # Both linear extrapolation
                fill_value = "extrapolate"
            elif short_end == "Flat" and long_end == "Flat":
                # Both flat extrapolation
                fill_value = (y_sorted[0], y_sorted[-1])
            elif short_end == "Flat":
                # Short end flat, long end linear
                fill_value = (y_sorted[0], "extrapolate")
            else:  # long_end == "Flat"
                # Short end linear, long end flat
                fill_value = ("extrapolate", y_sorted[-1])

            # Create interpolator
            f = interp1d(
                x_sorted,
                y_sorted,
                kind="linear",
                fill_value=fill_value,
                bounds_error=False,
            )

            # Handle mixed extrapolation (one side linear, one side flat)
            if (short_end == "Flat" and long_end == "Default") or (
                short_end == "Default" and long_end == "Flat"
            ):
                x_star_arr = np.asarray(x_star)
                result = np.empty_like(x_star_arr, dtype=float)

                # Identify regions
                below_mask = x_star_arr < x_sorted[0]
                above_mask = x_star_arr > x_sorted[-1]
                within_mask = ~(below_mask | above_mask)

                # Interpolate within bounds
                if np.any(within_mask):
                    result[within_mask] = f(x_star_arr[within_mask])

                # Handle short end
                if np.any(below_mask):
                    if short_end == "Flat":
                        result[below_mask] = y_sorted[0]
                    else:  # Default - linear extrapolation
                        # Compute slope from first two points
                        slope = (y_sorted[1] - y_sorted[0]) / (x_sorted[1] - x_sorted[0])
                        result[below_mask] = y_sorted[0] + slope * (
                            x_star_arr[below_mask] - x_sorted[0]
                        )

                # Handle long end
                if np.any(above_mask):
                    if long_end == "Flat":
                        result[above_mask] = y_sorted[-1]
                    else:  # Default - linear extrapolation
                        # Compute slope from last two points
                        slope = (y_sorted[-1] - y_sorted[-2]) / (x_sorted[-1] - x_sorted[-2])
                        result[above_mask] = y_sorted[-1] + slope * (
                            x_star_arr[above_mask] - x_sorted[-1]
                        )

                return result
            else:
                return f(x_star)
    elif y.ndim == 2:
        # 2D interpolation - each column is a separate series
        # Align dimensions
        if len(x) == y.shape[0]:
            pass  # Correct orientation
        elif len(x) == y.shape[1]:
            y = y.T  # Transpose to match
        else:
            raise ValueError(
                f"Incompatible shapes: the length of x ({len(x)}) does not match any dimension of y {y.shape}. "
                "At least one dimension of y must match the length of x."
            )

        # Find valid rows (non-NaN x values)
        x = np.asarray(x)
        valid_mask = ~np.isnan(x)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask, :]

        if len(x_valid) == 0:
            raise ValueError("No valid data points after removing NaN x values")

        # Sort by x
        sort_idx = np.argsort(x_valid)
        x_sorted = x_valid[sort_idx]
        y_sorted = y_valid[sort_idx, :]

        # Remove duplicate x, keep last occurrence
        if len(x_sorted) > 1:
            # Use return_index to find first occurrence, then reverse to get last
            unique_vals, first_idx = np.unique(x_sorted[::-1], return_index=True)
            last_idx = len(x_sorted) - 1 - first_idx
            sort_last_idx = np.sort(last_idx)
            x_sorted = x_sorted[sort_last_idx]
            y_sorted = y_sorted[sort_last_idx, :]

        if x_sorted.shape[0] == 1:
            # Single point - return constant values for all columns
            return np.tile(
                y_sorted[0, :],
                (np.asarray(x_star).shape[0] if np.asarray(x_star).ndim > 0 else 1, 1),
            )
        else:
            # Determine fill_value based on extrapolation settings
            if short_end == "Default" and long_end == "Default":
                fill_value = "extrapolate"
            elif short_end == "Flat" and long_end == "Flat":
                # Will handle per-column
                fill_value = None
            else:
                # Mixed extrapolation - will handle per-column
                fill_value = None

            # Vectorized interpolation for all columns
            # Pre-allocate result array for better performance
            x_star_arr = np.asarray(x_star)
            n_cols = y_sorted.shape[1]

            if x_star_arr.ndim == 0:
                # Single x_star value
                y_star = np.empty(n_cols)
                for col_idx in range(n_cols):
                    if fill_value == "extrapolate":
                        y_star[col_idx] = interp1d(
                            x_sorted,
                            y_sorted[:, col_idx],
                            kind="linear",
                            fill_value="extrapolate",
                            bounds_error=False,
                        )(x_star_arr)
                    else:
                        # Custom extrapolation
                        y_star[col_idx] = _interpolate_with_extrapolation_1d(
                            x_sorted, y_sorted[:, col_idx], x_star_arr, short_end, long_end
                        )
            else:
                # Multiple x_star values
                n_rows = len(x_star_arr)
                y_star = np.empty((n_rows, n_cols))
                for col_idx in range(n_cols):
                    if fill_value == "extrapolate":
                        y_star[:, col_idx] = interp1d(
                            x_sorted,
                            y_sorted[:, col_idx],
                            kind="linear",
                            fill_value="extrapolate",
                            bounds_error=False,
                        )(x_star_arr)
                    else:
                        # Custom extrapolation
                        y_star[:, col_idx] = _interpolate_with_extrapolation_1d(
                            x_sorted, y_sorted[:, col_idx], x_star_arr, short_end, long_end
                        )

            warnings.showwarning = lambda message, **kwargs: warn_and_print_x_y(
                message, x=x_sorted, y=y_sorted, **kwargs
            )
            return y_star
    else:
        raise ValueError("y must be 1D or 2D array")
