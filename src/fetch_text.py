"""
Module 3: Fetch Text
Downloads 10-K HTML files and extracts Item 1A (Risk Factors) and Item 7 (MD&A).
"""

import logging
import os
import re
import time
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup

from .config import CACHE_DIR, REQUEST_DELAY, USER_AGENT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_headers() -> dict:
    """Get HTTP headers with required User-Agent."""
    return {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    }


def download_filing_html(
    url: str,
    cik: int,
    accession_number: str,
    cache: bool = True,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Download 10-K filing HTML.

    Parameters
    ----------
    url : str
        URL to the filing HTML.
    cik : int
        Company CIK.
    accession_number : str
        Filing accession number.
    cache : bool
        Whether to use cached data.
    max_retries : int
        Maximum retry attempts.

    Returns
    -------
    str or None
        HTML content or None if failed.
    """
    # Create cache path
    accession_no_dashes = accession_number.replace("-", "")
    cache_dir = os.path.join(CACHE_DIR, "filings", str(cik))
    cache_path = os.path.join(cache_dir, f"{accession_no_dashes}.html")

    # Check cache first
    if cache and os.path.exists(cache_path):
        logger.debug(f"Loading filing from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # Download from SEC
    logger.debug(f"Downloading filing: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=get_headers(), timeout=60)
            response.raise_for_status()
            html_content = response.text

            # Cache the response
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            time.sleep(REQUEST_DELAY)
            return html_content

        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))
            else:
                logger.error(f"Failed to download filing: {url}")
                return None

    return None


def clean_html_text(html_content: str) -> str:
    """
    Extract clean text from HTML.

    Parameters
    ----------
    html_content : str
        Raw HTML content.

    Returns
    -------
    str
        Cleaned text.
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text(separator=" ")

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def extract_item_section(
    text: str,
    item_start_patterns: list,
    item_end_patterns: list,
) -> Optional[str]:
    """
    Extract a section from the document text.

    Parameters
    ----------
    text : str
        Full document text.
    item_start_patterns : list
        Regex patterns to find section start.
    item_end_patterns : list
        Regex patterns to find section end.

    Returns
    -------
    str or None
        Extracted section text or None if not found.
    """
    text_upper = text.upper()

    # Find start of section
    start_pos = None
    for pattern in item_start_patterns:
        match = re.search(pattern, text_upper)
        if match:
            start_pos = match.start()
            break

    if start_pos is None:
        return None

    # Find end of section
    end_pos = len(text)
    for pattern in item_end_patterns:
        match = re.search(pattern, text_upper[start_pos + 50:])
        if match:
            end_pos = start_pos + 50 + match.start()
            break

    # Extract section
    section_text = text[start_pos:end_pos]

    # Basic cleanup
    section_text = re.sub(r"\s+", " ", section_text).strip()

    return section_text


def extract_item_1a(text: str) -> Optional[str]:
    """
    Extract Item 1A - Risk Factors from 10-K text.

    Parameters
    ----------
    text : str
        Full document text.

    Returns
    -------
    str or None
        Item 1A text or None if not found.
    """
    start_patterns = [
        r"ITEM\s*1A[\.\:\s\-]+\s*RISK\s*FACTORS",
        r"ITEM\s*1A\s*[\.\:]\s*RISK",
        r"ITEM\s+1A\s+RISK\s+FACTORS",
        r"ITEM\s*1A\b",
    ]

    end_patterns = [
        r"ITEM\s*1B[\.\:\s\-]+\s*UNRESOLVED",
        r"ITEM\s*2[\.\:\s\-]+\s*PROPERTIES",
        r"ITEM\s+1B\b",
        r"ITEM\s+2\b",
    ]

    return extract_item_section(text, start_patterns, end_patterns)


def extract_item_7(text: str) -> Optional[str]:
    """
    Extract Item 7 - MD&A from 10-K text.

    Parameters
    ----------
    text : str
        Full document text.

    Returns
    -------
    str or None
        Item 7 text or None if not found.
    """
    start_patterns = [
        r"ITEM\s*7[\.\:\s\-]+\s*MANAGEMENT['S]*\s*DISCUSSION",
        r"ITEM\s*7\s*[\.\:]\s*MD&A",
        r"ITEM\s+7\s+MANAGEMENT",
        r"ITEM\s*7\b(?!\s*A)",
    ]

    end_patterns = [
        r"ITEM\s*7A[\.\:\s\-]+\s*QUANT",
        r"ITEM\s*8[\.\:\s\-]+\s*FINANCIAL\s*STATEMENTS",
        r"ITEM\s+7A\b",
        r"ITEM\s+8\b",
    ]

    return extract_item_section(text, start_patterns, end_patterns)


def extract_text_from_filing(
    url: str,
    cik: int,
    accession_number: str,
    cache: bool = True,
) -> Tuple[Optional[str], Optional[str], bool, str]:
    """
    Download and extract Item 1A and Item 7 from a 10-K filing.

    Parameters
    ----------
    url : str
        URL to the filing HTML.
    cik : int
        Company CIK.
    accession_number : str
        Filing accession number.
    cache : bool
        Whether to use cached data.

    Returns
    -------
    tuple
        (item1a_text, item7_text, text_extract_ok, filing_url)
    """
    html_content = download_filing_html(url, cik, accession_number, cache=cache)

    if html_content is None:
        return None, None, False, url

    # Clean HTML to text
    text = clean_html_text(html_content)

    # Extract sections
    item1a_text = extract_item_1a(text)
    item7_text = extract_item_7(text)

    # Determine if extraction was successful
    text_extract_ok = item1a_text is not None or item7_text is not None

    return item1a_text, item7_text, text_extract_ok, url


def process_filings_text(
    filings_df,
    cache: bool = True,
    progress_callback=None,
):
    """
    Process all filings to extract text sections.

    Parameters
    ----------
    filings_df : pd.DataFrame
        DataFrame with filing records.
    cache : bool
        Whether to use cached data.
    progress_callback : callable, optional
        Callback for progress updates.

    Returns
    -------
    pd.DataFrame
        DataFrame with added text columns.
    """
    from .get_filings import build_filing_url

    results = []
    total = len(filings_df)

    for idx, row in filings_df.iterrows():
        if progress_callback:
            progress_callback(idx + 1, total, row["cik"])
        else:
            if (idx + 1) % 100 == 0 or idx == 0:
                logger.info(f"Processing filing {idx + 1}/{total}")

        # Build URL
        url = build_filing_url(
            row["cik"],
            row["accessionNumber"],
            row["primaryDocument"],
        )

        # Extract text
        item1a, item7, extract_ok, filing_url = extract_text_from_filing(
            url, row["cik"], row["accessionNumber"], cache=cache
        )

        result = row.to_dict()
        result["item1a_text"] = item1a
        result["item7_text"] = item7
        result["text_extract_ok"] = extract_ok
        result["filing_url"] = filing_url

        results.append(result)

    import pandas as pd
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the module
    test_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    item1a, item7, ok, url = extract_text_from_filing(
        test_url, 320193, "0000320193-24-000123"
    )
    print(f"Item 1A found: {item1a is not None}")
    print(f"Item 7 found: {item7 is not None}")
    print(f"Extract OK: {ok}")
