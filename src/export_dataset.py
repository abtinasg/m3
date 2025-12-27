"""
Module 6: Export Dataset
Exports the final dataset to parquet and generates quality summary.
"""

import json
import logging
import os
from typing import Dict, List

import pandas as pd

from .config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum character length for text fields in CSV export
CSV_TEXT_TRUNCATE_LENGTH = 1000

# Define columns for final dataset
IDENTIFIER_COLS = [
    "cik",
    "ticker",
    "company_name",
    "accessionNumber",
    "filingDate",
    "reportDate",
    "fiscal_year",
    "sic",
]

DOCUMENT_COLS = [
    "filing_url",
]

TEXT_COLS = [
    "item1a_text",
    "item7_text",
]

FINANCIAL_RAW_COLS = [
    "Assets",
    "Liabilities",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "RetainedEarnings",
    "EBIT",
    "NetIncome",
    "Revenue",
    "CashAndCashEquivalents",
]

RATIO_COLS = [
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "debt_to_assets",
    "debt_to_equity",
    "roa",
    "roe",
    "operating_margin",
    "asset_turnover",
]

ZSCORE_COLS = [
    "x1_wc_ta",
    "x2_re_ta",
    "x3_ebit_ta",
    "x4_eq_tl",
    "zscore_zpp",
    "label_3class",
]

QUALITY_COLS = [
    "text_extract_ok",
    "ebit_source_used",
    "missing_flags",
]


def get_available_columns(df: pd.DataFrame, column_list: List[str]) -> List[str]:
    """Get columns that exist in the DataFrame."""
    return [col for col in column_list if col in df.columns]


def export_to_parquet(
    df: pd.DataFrame,
    output_path: str = None,
) -> str:
    """
    Export dataset to parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        Final dataset.
    output_path : str, optional
        Output file path. Defaults to OUTPUT_DIR/firmyear_10k_2010_2024.parquet

    Returns
    -------
    str
        Path to saved file.
    """
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "firmyear_10k_2010_2024.parquet")

    # Select columns in order
    all_cols = (
        IDENTIFIER_COLS
        + DOCUMENT_COLS
        + TEXT_COLS
        + FINANCIAL_RAW_COLS
        + RATIO_COLS
        + ZSCORE_COLS
        + QUALITY_COLS
    )

    available_cols = get_available_columns(df, all_cols)
    export_df = df[available_cols].copy()

    # Save to parquet
    export_df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info(f"Exported dataset to: {output_path}")

    return output_path


def export_to_csv(
    df: pd.DataFrame,
    output_path: str = None,
) -> str:
    """
    Export dataset to CSV format.

    Parameters
    ----------
    df : pd.DataFrame
        Final dataset.
    output_path : str, optional
        Output file path.

    Returns
    -------
    str
        Path to saved file.
    """
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "firmyear_10k_2010_2024.csv")

    # Select columns in order
    all_cols = (
        IDENTIFIER_COLS
        + DOCUMENT_COLS
        + TEXT_COLS
        + FINANCIAL_RAW_COLS
        + RATIO_COLS
        + ZSCORE_COLS
        + QUALITY_COLS
    )

    available_cols = get_available_columns(df, all_cols)
    export_df = df[available_cols].copy()

    # For CSV, truncate long text fields
    for col in TEXT_COLS:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: (x[:CSV_TEXT_TRUNCATE_LENGTH] + "...") if isinstance(x, str) and len(x) > CSV_TEXT_TRUNCATE_LENGTH else x
            )

    export_df.to_csv(output_path, index=False)
    logger.info(f"Exported dataset to: {output_path}")

    return output_path


def generate_quality_summary(df: pd.DataFrame) -> Dict:
    """
    Generate quality summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Final dataset.

    Returns
    -------
    dict
        Quality summary statistics.
    """
    summary = {
        "total_rows": len(df),
        "total_companies": df["cik"].nunique() if "cik" in df.columns else 0,
        "year_range": {
            "min": int(df["fiscal_year"].min()) if "fiscal_year" in df.columns and not df["fiscal_year"].isna().all() else None,
            "max": int(df["fiscal_year"].max()) if "fiscal_year" in df.columns and not df["fiscal_year"].isna().all() else None,
        },
        "class_distribution": {},
        "text_extraction": {},
        "missing_data": {},
    }

    # Class distribution
    if "label_3class" in df.columns:
        class_counts = df["label_3class"].value_counts().to_dict()
        summary["class_distribution"] = {
            "distress": int(class_counts.get("distress", 0)),
            "gray": int(class_counts.get("gray", 0)),
            "safe": int(class_counts.get("safe", 0)),
            "missing": int(df["label_3class"].isna().sum()),
        }

    # Text extraction success rate
    if "text_extract_ok" in df.columns:
        total_with_status = df["text_extract_ok"].notna().sum()
        success_count = df["text_extract_ok"].sum()
        summary["text_extraction"] = {
            "total": int(total_with_status),
            "success": int(success_count),
            "failure": int(total_with_status - success_count),
            "success_rate": round(success_count / max(total_with_status, 1) * 100, 2),
        }

    # Missing data counts
    key_fields = ["Assets", "Liabilities", "EBIT", "StockholdersEquity", "RetainedEarnings"]
    for field in key_fields:
        if field in df.columns:
            summary["missing_data"][field] = int(df[field].isna().sum())

    # EBIT source breakdown
    if "ebit_source_used" in df.columns:
        ebit_sources = df["ebit_source_used"].value_counts().to_dict()
        summary["ebit_sources"] = {str(k): int(v) for k, v in ebit_sources.items()}

    return summary


def export_summary_json(
    summary: Dict,
    output_path: str = None,
) -> str:
    """
    Export quality summary to JSON.

    Parameters
    ----------
    summary : dict
        Quality summary dictionary.
    output_path : str, optional
        Output file path.

    Returns
    -------
    str
        Path to saved file.
    """
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "summary.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported summary to: {output_path}")
    return output_path


def export_dataset(
    df: pd.DataFrame,
    output_dir: str = None,
    formats: List[str] = None,
) -> Dict[str, str]:
    """
    Export complete dataset and summary.

    Parameters
    ----------
    df : pd.DataFrame
        Final dataset.
    output_dir : str, optional
        Output directory.
    formats : list, optional
        Output formats ('parquet', 'csv'). Default is ['parquet'].

    Returns
    -------
    dict
        Dictionary of output file paths.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    if formats is None:
        formats = ["parquet"]

    os.makedirs(output_dir, exist_ok=True)

    outputs = {}

    # Export dataset
    if "parquet" in formats:
        parquet_path = os.path.join(output_dir, "firmyear_10k_2010_2024.parquet")
        outputs["parquet"] = export_to_parquet(df, parquet_path)

    if "csv" in formats:
        csv_path = os.path.join(output_dir, "firmyear_10k_2010_2024.csv")
        outputs["csv"] = export_to_csv(df, csv_path)

    # Generate and export summary
    summary = generate_quality_summary(df)
    summary_path = os.path.join(output_dir, "summary.json")
    outputs["summary"] = export_summary_json(summary, summary_path)

    logger.info(f"Export complete. Files saved to: {output_dir}")

    return outputs


if __name__ == "__main__":
    # Test the module
    test_data = pd.DataFrame({
        "cik": [320193, 320193, 789019],
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "company_name": ["Apple Inc", "Apple Inc", "Microsoft Corp"],
        "fiscal_year": [2022, 2023, 2023],
        "zscore_zpp": [0.5, 1.5, 3.5],
        "label_3class": ["distress", "gray", "safe"],
        "text_extract_ok": [True, True, False],
        "Assets": [352583000000, 352583000000, None],
        "EBIT": [119437000000, 119437000000, 88523000000],
        "ebit_source_used": ["OperatingIncomeLoss", "OperatingIncomeLoss", "OperatingIncomeLoss"],
    })

    outputs = export_dataset(test_data, output_dir="/tmp/test_export", formats=["parquet", "csv"])
    print("Outputs:", outputs)

    # Print summary
    with open(outputs["summary"], "r") as f:
        print("\nSummary:")
        print(json.dumps(json.load(f), indent=2))
