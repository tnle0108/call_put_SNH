from pathlib import Path
import json
from typing import Any
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import logging


logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DATASETS_DIR = _PACKAGE_ROOT / "datasets"
_SPECS_DIR = _PACKAGE_ROOT / "specs"

_DATA_CACHE: dict[str, dict] = {}
_SPEC_CACHE: dict[str, dict[str, Any]] = {}

for _dir, _name in [(_DATASETS_DIR, "datasets"), (_SPECS_DIR, "specs")]:
    if not _dir.exists():
        logger.warning(
            f"{_name} directory not found at {_dir}. "
            f"Loading may fail. Package root: {_PACKAGE_ROOT}"
        )


def _load_data_from_csv(file_path: Path) -> DataFrame:
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"{file_path} is empty.")
            return DataFrame()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype("float64")
        return df
    except FileNotFoundError:
        logger.warning(f"Data file not found: {file_path}")
        return DataFrame()
    except KeyError as e:
        logger.error(f"Required column missing in data file {file_path}: {e}")
        return DataFrame()
    except (ValueError, pd.errors.ParserError) as e:
        logger.error(f"Error parsing data from {file_path}: {e}")
        return DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading data from {file_path}: {e}")
        return DataFrame()


def _load_holiday_from_csv(file_path: Path) -> ndarray:
    _dtype = np.dtype("datetime64[D]")
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"])
        if df.empty:
            logger.warning(f"{file_path} is empty.")
            return np.array([], dtype=_dtype)
        dates = df["Date"].values.astype(_dtype)
        return np.sort(dates)
    except FileNotFoundError:
        logger.warning(f"Holiday file not found: {file_path}")
        return np.array([], dtype=_dtype)
    except Exception as e:
        logger.error(f"Error loading holiday data from {file_path}: {e}")
        return np.array([], dtype=_dtype)


def load_data(data: str, refresh: bool = True) -> dict[str, DataFrame] | dict[str, ndarray]:
    global _DATA_CACHE
    data = data.lower()

    if refresh or data not in _DATA_CACHE:
        logger.debug(f"Loading {data} data...")
        loaded_data = {}
        data_dir = _DATASETS_DIR / data
        if data_dir.exists():
            for data_file in data_dir.glob("*.csv"):
                key = data_file.stem.lower()
                if data == "holiday":
                    loaded_data[key] = _load_holiday_from_csv(data_file)
                else:
                    loaded_data[key] = _load_data_from_csv(data_file)
        _DATA_CACHE[data] = loaded_data
    return _DATA_CACHE[data]


def _load_spec_from_json(file_path: Path) -> dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Specification file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(
            f"Unexpected error loading specification from {file_path}: {e}"
        )
        return {}


def load_spec(spec: str) -> dict[str, Any]:
    global _SPEC_CACHE
    spec = spec.lower()
    if spec not in _SPEC_CACHE:
        logger.debug(f"Loading {spec} specification...")
        spec_path = _SPECS_DIR / f"{spec}.json"
        if spec_path.exists():
            _SPEC_CACHE[spec] = _load_spec_from_json(spec_path)
        else:
            logger.warning(f"Specification file not found: {spec_path}")
            _SPEC_CACHE[spec] = {}
    return _SPEC_CACHE[spec]
