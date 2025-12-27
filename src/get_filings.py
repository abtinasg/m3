"""
Module 2: Get Filings
Retrieves 10-K filings for each company from SEC EDGAR.
"""

import json
import logging
import os
import time
from typing import List, Optional

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


def get_company_submissions(
    cik: int,
    cache: bool = True,
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Get submissions data for a company from SEC.

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
        Submissions JSON data or None if failed.
    """
    cik_str = format_cik(cik)
    cache_dir = os.path.join(CACHE_DIR, "submissions")
    cache_path = os.path.join(cache_dir, f"CIK{cik_str}.json")

    # Check cache first
    if cache and os.path.exists(cache_path):
        logger.debug(f"Loading submissions from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Download from SEC
    url = f"{SEC_DATA_URL}/submissions/CIK{cik_str}.json"
    logger.debug(f"Downloading submissions: {url}")

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
                logger.error(f"CIK {cik}: Failed to download submissions")
                return None

    return None


def extract_10k_filings(
    submissions: dict,
    cik: int,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> List[dict]:
    """
    Extract 10-K filings from submissions data.

    Parameters
    ----------
    submissions : dict
        Raw submissions JSON from SEC.
    cik : int
        Company CIK.
    start_year : int
        Start year for filtering.
    end_year : int
        End year for filtering.

    Returns
    -------
    list of dict
        List of 10-K filing records.
    """
    filings = []

    if "filings" not in submissions or "recent" not in submissions["filings"]:
        return filings

    recent = submissions["filings"]["recent"]

    # Extract lists once for efficiency
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    report_dates = recent.get("reportDate", [])
    primary_documents = recent.get("primaryDocument", [])
    n = len(forms)

    for i in range(n):
        form = forms[i] if i < len(forms) else None
        filing_date = filing_dates[i] if i < len(filing_dates) else None

        # Filter for 10-K forms only
        if form != "10-K":
            continue

        # Parse filing date and filter by year
        if filing_date:
            try:
                year = int(filing_date[:4])
                if year < start_year or year > end_year:
                    continue
            except (ValueError, IndexError):
                continue
        else:
            continue

        # Extract filing details
        filing = {
            "cik": cik,
            "accessionNumber": accession_numbers[i] if i < len(accession_numbers) else None,
            "filingDate": filing_date,
            "reportDate": report_dates[i] if i < len(report_dates) else None,
            "primaryDocument": primary_documents[i] if i < len(primary_documents) else None,
            "form": form,
        }

        filings.append(filing)

    return filings


def get_10k_filings_for_companies(
    ciks: List[int],
    cache: bool = True,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Get all 10-K filings for a list of companies.

    Parameters
    ----------
    ciks : list of int
        List of company CIKs.
    cache : bool
        Whether to use cached data.
    start_year : int
        Start year for filtering.
    end_year : int
        End year for filtering.
    progress_callback : callable, optional
        Callback function for progress updates.

    Returns
    -------
    pd.DataFrame
        DataFrame with 10-K filing records.
    """
    all_filings = []
    total = len(ciks)

    for idx, cik in enumerate(ciks):
        if progress_callback:
            progress_callback(idx + 1, total, cik)
        else:
            if (idx + 1) % 100 == 0 or idx == 0:
                logger.info(f"Processing CIK {idx + 1}/{total}")

        submissions = get_company_submissions(cik, cache=cache)
        if submissions:
            filings = extract_10k_filings(
                submissions, cik, start_year=start_year, end_year=end_year
            )
            all_filings.extend(filings)

    df = pd.DataFrame(all_filings)
    logger.info(f"Found {len(df)} 10-K filings from {len(ciks)} companies")

    return df


def build_filing_url(cik: int, accession_number: str, primary_document: str) -> str:
    """
    Build the URL for a filing document.

    Parameters
    ----------
    cik : int
        Company CIK.
    accession_number : str
        Accession number (e.g., "0000320193-24-000123").
    primary_document : str
        Primary document filename.

    Returns
    -------
    str
        Full URL to the filing document.
    """
    # Remove dashes from accession number
    accession_no_dashes = accession_number.replace("-", "")

    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_no_dashes}/{primary_document}"
    )


if __name__ == "__main__":
    # Test the module
    from .download_universe import get_company_universe

    universe = get_company_universe()
    # Test with first 5 companies
    test_ciks = universe["cik"].head(5).tolist()
    filings = get_10k_filings_for_companies(test_ciks)
    print(filings.head())
    print(f"\nTotal 10-K filings: {len(filings)}")
