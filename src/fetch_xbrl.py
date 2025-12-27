"""
Module 4: Fetch XBRL
Retrieves financial data from SEC XBRL API for calculating ratios and Z'' score.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from .config import (
    CACHE_DIR,
    END_YEAR,
    REQUEST_DELAY,
    SEC_DATA_URL,
    START_YEAR,
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


def format_cik(cik: int) -> str:
    """Format CIK as 10-digit string with leading zeros."""
    return str(int(cik)).zfill(10)


# XBRL tags to fetch - primary and fallback options
XBRL_TAGS = {
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "AssetsCurrent": ["AssetsCurrent"],
    "LiabilitiesCurrent": ["LiabilitiesCurrent"],
    "StockholdersEquity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "Equity",
    ],
    "RetainedEarnings": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarnings",
    ],
    "EBIT": [
        "EarningsBeforeInterestAndTaxes",
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
    ],
    "NetIncome": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "CashAndCashEquivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
    ],
    "InterestExpense": [
        "InterestExpense",
        "InterestExpenseDebt",
    ],
}


def get_company_facts(
    cik: int,
    cache: bool = True,
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Get XBRL company facts from SEC API.

    Parameters
    ----------
    cik : int
        Company CIK number.
    cache : bool
        Whether to use cached data.
    max_retries : int
        Maximum retry attempts.

    Returns
    -------
    dict or None
        Company facts JSON or None if failed.
    """
    cik_str = format_cik(cik)
    cache_dir = os.path.join(CACHE_DIR, "xbrl")
    cache_path = os.path.join(cache_dir, f"CIK{cik_str}.json")

    # Check cache first
    if cache and os.path.exists(cache_path):
        logger.debug(f"Loading XBRL from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Download from SEC
    url = f"{SEC_DATA_URL}/api/xbrl/companyfacts/CIK{cik_str}.json"
    logger.debug(f"Downloading XBRL: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=get_headers(), timeout=30)
            response.raise_for_status()
            data = response.json()

            # Cache the response
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            time.sleep(REQUEST_DELAY)
            return data

        except requests.RequestException as e:
            logger.warning(f"CIK {cik}: Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))
            else:
                logger.error(f"CIK {cik}: Failed to download XBRL")
                return None

    return None


def extract_annual_value(
    facts: dict,
    tag_alternatives: List[str],
    fiscal_year: int,
) -> tuple:
    """
    Extract annual value for a given XBRL tag.

    Parameters
    ----------
    facts : dict
        Company facts JSON.
    tag_alternatives : list
        List of possible tag names to try.
    fiscal_year : int
        Fiscal year to extract.

    Returns
    -------
    tuple
        (value, source_tag) or (None, None) if not found.
    """
    if "facts" not in facts or "us-gaap" not in facts.get("facts", {}):
        return None, None

    us_gaap = facts["facts"]["us-gaap"]

    for tag in tag_alternatives:
        if tag not in us_gaap:
            continue

        units = us_gaap[tag].get("units", {})

        # Try USD first, then other units
        for unit_key in ["USD", "USD/shares", "pure"]:
            if unit_key not in units:
                continue

            for entry in units[unit_key]:
                # Check if this is an annual value
                if entry.get("form") != "10-K":
                    continue

                # Check fiscal year
                fy = entry.get("fy")
                if fy != fiscal_year:
                    continue

                # Prefer end date values (not duration)
                if "val" in entry:
                    return entry["val"], tag

    return None, None


def get_financial_data_for_year(
    facts: dict,
    fiscal_year: int,
) -> Dict[str, any]:
    """
    Extract all required financial data for a fiscal year.

    Parameters
    ----------
    facts : dict
        Company facts JSON.
    fiscal_year : int
        Fiscal year to extract.

    Returns
    -------
    dict
        Financial data with values and source tags.
    """
    result = {"fiscal_year": fiscal_year}
    missing_flags = []

    for field_name, tag_alternatives in XBRL_TAGS.items():
        value, source_tag = extract_annual_value(facts, tag_alternatives, fiscal_year)
        result[field_name] = value
        result[f"{field_name}_source"] = source_tag

        if value is None:
            missing_flags.append(field_name)

    result["missing_flags"] = ",".join(missing_flags) if missing_flags else None

    return result


def get_financial_data_for_company(
    cik: int,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    cache: bool = True,
) -> List[dict]:
    """
    Get all financial data for a company across years.

    Parameters
    ----------
    cik : int
        Company CIK.
    start_year : int
        Start fiscal year.
    end_year : int
        End fiscal year.
    cache : bool
        Whether to use cached data.

    Returns
    -------
    list of dict
        Financial data for each year.
    """
    facts = get_company_facts(cik, cache=cache)
    if facts is None:
        return []

    results = []
    for year in range(start_year, end_year + 1):
        data = get_financial_data_for_year(facts, year)
        data["cik"] = cik
        results.append(data)

    return results


def get_financial_data_for_companies(
    ciks: List[int],
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    cache: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Get financial data for multiple companies.

    Parameters
    ----------
    ciks : list of int
        List of company CIKs.
    start_year : int
        Start fiscal year.
    end_year : int
        End fiscal year.
    cache : bool
        Whether to use cached data.
    progress_callback : callable, optional
        Callback for progress updates.

    Returns
    -------
    pd.DataFrame
        Financial data for all companies and years.
    """
    all_data = []
    total = len(ciks)

    for idx, cik in enumerate(ciks):
        if progress_callback:
            progress_callback(idx + 1, total, cik)
        else:
            if (idx + 1) % 100 == 0 or idx == 0:
                logger.info(f"Processing XBRL {idx + 1}/{total}")

        company_data = get_financial_data_for_company(
            cik, start_year=start_year, end_year=end_year, cache=cache
        )
        all_data.extend(company_data)

    df = pd.DataFrame(all_data)
    logger.info(f"Retrieved financial data for {len(ciks)} companies")

    return df


if __name__ == "__main__":
    # Test the module
    test_cik = 320193  # Apple
    data = get_financial_data_for_company(test_cik, start_year=2020, end_year=2023)
    print(pd.DataFrame(data))
