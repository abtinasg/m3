#!/usr/bin/env python3
"""
SEC EDGAR Dataset Pipeline
Main runner script that orchestrates all modules to build the dataset.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

import pandas as pd

from src.build_features import build_all_features
from src.config import CACHE_DIR, END_YEAR, OUTPUT_DIR, START_YEAR
from src.download_universe import get_company_universe
from src.export_dataset import export_dataset
from src.fetch_text import process_filings_text
from src.fetch_xbrl import get_financial_data_for_companies
from src.get_filings import get_10k_filings_for_companies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    max_companies: Optional[int] = None,
    cache: bool = True,
    output_dir: str = OUTPUT_DIR,
    output_formats: List[str] = None,
    skip_text: bool = False,
) -> dict:
    """
    Run the complete SEC EDGAR dataset pipeline.

    Parameters
    ----------
    start_year : int
        Start year for filings.
    end_year : int
        End year for filings.
    max_companies : int, optional
        Maximum number of companies to process (for testing).
    cache : bool
        Whether to use cached data.
    output_dir : str
        Output directory for dataset files.
    output_formats : list
        Output formats ('parquet', 'csv').
    skip_text : bool
        Skip text extraction (faster for testing).

    Returns
    -------
    dict
        Dictionary with output file paths.
    """
    if output_formats is None:
        output_formats = ["parquet"]

    logger.info("=" * 60)
    logger.info("SEC EDGAR Dataset Pipeline")
    logger.info("=" * 60)

    # Step 1: Download company universe
    logger.info("\n[Step 1/6] Downloading company universe...")
    universe = get_company_universe(cache=cache)
    logger.info(f"Found {len(universe)} companies in universe")

    if max_companies:
        universe = universe.head(max_companies)
        logger.info(f"Limited to {max_companies} companies for processing")

    ciks = universe["cik"].tolist()

    # Step 2: Get 10-K filings
    logger.info("\n[Step 2/6] Getting 10-K filings...")
    filings_df = get_10k_filings_for_companies(
        ciks,
        cache=cache,
        start_year=start_year,
        end_year=end_year,
    )
    logger.info(f"Found {len(filings_df)} 10-K filings")

    if filings_df.empty:
        logger.warning("No filings found. Exiting.")
        return {}

    # Add company info to filings
    filings_df = filings_df.merge(
        universe[["cik", "ticker", "name", "exchange"]].rename(
            columns={"name": "company_name"}
        ),
        on="cik",
        how="left",
    )

    # Step 3: Extract text from filings (optional)
    if not skip_text:
        logger.info("\n[Step 3/6] Extracting text from filings...")
        filings_df = process_filings_text(filings_df, cache=cache)
        logger.info(f"Text extraction complete. Success rate: {filings_df['text_extract_ok'].mean()*100:.1f}%")
    else:
        logger.info("\n[Step 3/6] Skipping text extraction...")
        filings_df["item1a_text"] = None
        filings_df["item7_text"] = None
        filings_df["text_extract_ok"] = False
        filings_df["filing_url"] = None

    # Step 4: Get XBRL financial data
    logger.info("\n[Step 4/6] Fetching XBRL financial data...")
    xbrl_df = get_financial_data_for_companies(
        ciks,
        start_year=start_year,
        end_year=end_year,
        cache=cache,
    )
    logger.info(f"Retrieved financial data for {xbrl_df['cik'].nunique()} companies")

    # Merge filings with financial data
    # Extract fiscal year from filing date
    filings_df["fiscal_year"] = pd.to_datetime(
        filings_df["filingDate"], errors="coerce"
    ).dt.year

    # For a more accurate match, we'd use report date, but filing date year works as fallback
    if "reportDate" in filings_df.columns:
        report_year = pd.to_datetime(
            filings_df["reportDate"], errors="coerce"
        ).dt.year
        filings_df["fiscal_year"] = report_year.fillna(filings_df["fiscal_year"])

    # Merge on CIK and fiscal year
    merged_df = filings_df.merge(
        xbrl_df,
        on=["cik", "fiscal_year"],
        how="left",
    )
    logger.info(f"Merged dataset has {len(merged_df)} records")

    # Step 5: Build features
    logger.info("\n[Step 5/6] Building features (ratios, Z'' score, labels)...")
    dataset = build_all_features(merged_df)

    # Add SIC code placeholder (would need additional API call to get)
    dataset["sic"] = None

    # Rename columns for output
    if "accessionNumber" in dataset.columns:
        dataset["accession"] = dataset["accessionNumber"]
    if "filingDate" in dataset.columns:
        dataset["filing_date"] = dataset["filingDate"]
    if "reportDate" in dataset.columns:
        dataset["report_date"] = dataset["reportDate"]

    # Step 6: Export dataset
    logger.info("\n[Step 6/6] Exporting dataset...")
    outputs = export_dataset(
        dataset,
        output_dir=output_dir,
        formats=output_formats,
    )

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(dataset)}")
    logger.info(f"Total companies: {dataset['cik'].nunique()}")

    if "label_3class" in dataset.columns:
        class_counts = dataset["label_3class"].value_counts()
        logger.info(f"Class distribution:")
        for label, count in class_counts.items():
            logger.info(f"  {label}: {count}")

    logger.info(f"\nOutput files:")
    for fmt, path in outputs.items():
        logger.info(f"  {fmt}: {path}")

    return outputs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SEC EDGAR Dataset Pipeline - Build a dataset from 10-K filings"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=f"Start year for filings (default: {START_YEAR})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help=f"End year for filings (default: {END_YEAR})",
    )
    parser.add_argument(
        "--max-companies",
        type=int,
        default=None,
        help="Maximum number of companies to process (for testing)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (re-download all data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-formats",
        type=str,
        nargs="+",
        default=["parquet"],
        choices=["parquet", "csv"],
        help="Output formats (default: parquet)",
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip text extraction (faster for testing)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (overrides SEC_CACHE_DIR env variable)",
    )

    args = parser.parse_args()

    # Override cache directory if specified
    if args.cache_dir:
        os.environ["SEC_CACHE_DIR"] = args.cache_dir
        from src import config
        config.CACHE_DIR = args.cache_dir

    try:
        outputs = run_pipeline(
            start_year=args.start_year,
            end_year=args.end_year,
            max_companies=args.max_companies,
            cache=not args.no_cache,
            output_dir=args.output_dir,
            output_formats=args.output_formats,
            skip_text=args.skip_text,
        )

        if outputs:
            logger.info("\nPipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("\nPipeline completed with no output.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
