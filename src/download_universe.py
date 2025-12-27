"""
Module 1: Download Universe
Downloads company tickers from SEC and creates a company universe table.
"""

import json
import logging
import os
import time
from typing import Optional

import pandas as pd
import requests

from .config import (
    CACHE_DIR,
    COMPANY_TICKERS_URL,
    REQUEST_DELAY,
    USER_AGENT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_headers() -> dict:
    """Get HTTP headers with required User-Agent."""
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }


def download_company_tickers(
    cache: bool = True,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Download company tickers from SEC.

    Parameters
    ----------
    cache : bool
        Whether to use cached data if available.
    max_retries : int
        Maximum number of retry attempts.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: cik, ticker, name, exchange
    """
    cache_path = os.path.join(CACHE_DIR, "company_tickers_exchange.json")

    # Check cache first
    if cache and os.path.exists(cache_path):
        logger.info(f"Loading company tickers from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _parse_company_tickers(data)

    # Download from SEC
    logger.info(f"Downloading company tickers from: {COMPANY_TICKERS_URL}")

    for attempt in range(max_retries):
        try:
            response = requests.get(
                COMPANY_TICKERS_URL,
                headers=get_headers(),
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # Cache the response
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.info(f"Cached company tickers to: {cache_path}")

            time.sleep(REQUEST_DELAY)
            return _parse_company_tickers(data)

        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))
            else:
                raise

    raise RuntimeError("Failed to download company tickers")


def _parse_company_tickers(data: dict) -> pd.DataFrame:
    """
    Parse SEC company tickers JSON into a DataFrame.

    Parameters
    ----------
    data : dict
        Raw JSON data from SEC.

    Returns
    -------
    pd.DataFrame
        Parsed company tickers with columns: cik, ticker, name, exchange
    """
    # The SEC file has 'data' as a list of lists and 'fields' as column names
    if "data" in data and "fields" in data:
        df = pd.DataFrame(data["data"], columns=data["fields"])
        # Normalize column names
        df.columns = df.columns.str.lower()
        if "cik" not in df.columns and "cik_str" in df.columns:
            df["cik"] = df["cik_str"]
    else:
        # Alternative format: dict with numeric keys
        df = pd.DataFrame.from_dict(data, orient="index")
        df.columns = df.columns.str.lower()

    # Ensure required columns exist
    required_cols = ["cik", "ticker", "name"]
    for col in required_cols:
        if col not in df.columns:
            # Try to find alternative names
            alt_names = {
                "cik": ["cik_str", "CIK"],
                "ticker": ["symbol", "TICKER"],
                "name": ["company", "title", "NAME"],
            }
            for alt in alt_names.get(col, []):
                if alt in df.columns:
                    df[col] = df[alt]
                    break
            else:
                df[col] = None

    # Format CIK as integer
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")

    # Add exchange column if not present
    if "exchange" not in df.columns:
        df["exchange"] = None

    # Select and order columns
    result = df[["cik", "ticker", "name", "exchange"]].copy()

    # Remove rows with missing CIK
    result = result.dropna(subset=["cik"])

    logger.info(f"Loaded {len(result)} companies")
    return result


def get_company_universe(
    cache: bool = True,
    filter_exchange: Optional[list] = None,
) -> pd.DataFrame:
    """
    Get the universe of companies for analysis.

    Parameters
    ----------
    cache : bool
        Whether to use cached data.
    filter_exchange : list, optional
        List of exchanges to filter by (e.g., ['NYSE', 'NASDAQ']).

    Returns
    -------
    pd.DataFrame
        Company universe with columns: cik, ticker, name, exchange
    """
    df = download_company_tickers(cache=cache)

    if filter_exchange:
        df = df[df["exchange"].isin(filter_exchange)]
        logger.info(f"Filtered to {len(df)} companies on exchanges: {filter_exchange}")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Test the module
    universe = get_company_universe()
    print(universe.head())
    print(f"\nTotal companies: {len(universe)}")
