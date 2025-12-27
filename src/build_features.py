"""
Module 5: Build Features
Calculates financial ratios, Z'' score, and 3-class labels.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .config import Z_DISTRESS_THRESHOLD, Z_SAFE_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Altman Z'' Score coefficients (four-factor model for non-manufacturing)
# Reference: Altman, E.I. (1968) and subsequent modifications
ZSCORE_COEF_X1 = 6.56  # Working Capital / Total Assets
ZSCORE_COEF_X2 = 3.26  # Retained Earnings / Total Assets
ZSCORE_COEF_X3 = 6.72  # EBIT / Total Assets
ZSCORE_COEF_X4 = 1.05  # Equity / Total Liabilities


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    default: float = np.nan,
) -> pd.Series:
    """
    Safely divide two series, handling division by zero.

    Parameters
    ----------
    numerator : pd.Series
        Numerator values.
    denominator : pd.Series
        Denominator values.
    default : float
        Default value when division is not possible.

    Returns
    -------
    pd.Series
        Result of division.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], default)
    return result


def calculate_liquidity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate liquidity ratios.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial data.

    Returns
    -------
    pd.DataFrame
        DataFrame with added ratio columns.
    """
    df = df.copy()

    # Current Ratio = Current Assets / Current Liabilities
    df["current_ratio"] = safe_divide(
        df["AssetsCurrent"], df["LiabilitiesCurrent"]
    )

    # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
    # Since we may not have inventory, use current assets as approximation
    df["quick_ratio"] = safe_divide(
        df["AssetsCurrent"], df["LiabilitiesCurrent"]
    )

    # Cash Ratio = Cash / Current Liabilities
    if "CashAndCashEquivalents" in df.columns:
        df["cash_ratio"] = safe_divide(
            df["CashAndCashEquivalents"], df["LiabilitiesCurrent"]
        )
    else:
        df["cash_ratio"] = np.nan

    return df


def calculate_leverage_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate leverage ratios.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial data.

    Returns
    -------
    pd.DataFrame
        DataFrame with added ratio columns.
    """
    df = df.copy()

    # Debt to Assets = Total Liabilities / Total Assets
    df["debt_to_assets"] = safe_divide(df["Liabilities"], df["Assets"])

    # Debt to Equity = Total Liabilities / Stockholders' Equity
    df["debt_to_equity"] = safe_divide(
        df["Liabilities"], df["StockholdersEquity"]
    )

    return df


def calculate_profitability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate profitability ratios.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial data.

    Returns
    -------
    pd.DataFrame
        DataFrame with added ratio columns.
    """
    df = df.copy()

    # ROA = Net Income / Total Assets
    if "NetIncome" in df.columns:
        df["roa"] = safe_divide(df["NetIncome"], df["Assets"])
    else:
        df["roa"] = np.nan

    # ROE = Net Income / Stockholders' Equity
    if "NetIncome" in df.columns:
        df["roe"] = safe_divide(df["NetIncome"], df["StockholdersEquity"])
    else:
        df["roe"] = np.nan

    # Operating Margin = EBIT / Revenue
    if "Revenue" in df.columns:
        df["operating_margin"] = safe_divide(df["EBIT"], df["Revenue"])
    else:
        df["operating_margin"] = np.nan

    return df


def calculate_efficiency_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate efficiency ratios.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial data.

    Returns
    -------
    pd.DataFrame
        DataFrame with added ratio columns.
    """
    df = df.copy()

    # Asset Turnover = Revenue / Total Assets
    if "Revenue" in df.columns:
        df["asset_turnover"] = safe_divide(df["Revenue"], df["Assets"])
    else:
        df["asset_turnover"] = np.nan

    return df


def calculate_zscore_zpp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Altman Z'' Score (four-factor model for non-manufacturing).

    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    Where:
    - X1 = (Current Assets - Current Liabilities) / Total Assets
    - X2 = Retained Earnings / Total Assets
    - X3 = EBIT / Total Assets
    - X4 = Equity / Total Liabilities

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial data.

    Returns
    -------
    pd.DataFrame
        DataFrame with Z'' score column.
    """
    df = df.copy()

    # Calculate components
    # X1: Working Capital / Total Assets
    df["x1_wc_ta"] = safe_divide(
        df["AssetsCurrent"] - df["LiabilitiesCurrent"],
        df["Assets"],
    )

    # X2: Retained Earnings / Total Assets
    df["x2_re_ta"] = safe_divide(df["RetainedEarnings"], df["Assets"])

    # X3: EBIT / Total Assets
    df["x3_ebit_ta"] = safe_divide(df["EBIT"], df["Assets"])

    # X4: Equity / Total Liabilities
    df["x4_eq_tl"] = safe_divide(df["StockholdersEquity"], df["Liabilities"])

    # Calculate Z'' Score
    df["zscore_zpp"] = (
        ZSCORE_COEF_X1 * df["x1_wc_ta"].fillna(0)
        + ZSCORE_COEF_X2 * df["x2_re_ta"].fillna(0)
        + ZSCORE_COEF_X3 * df["x3_ebit_ta"].fillna(0)
        + ZSCORE_COEF_X4 * df["x4_eq_tl"].fillna(0)
    )

    # Set to NaN if any component is missing
    missing_mask = (
        df["x1_wc_ta"].isna()
        | df["x2_re_ta"].isna()
        | df["x3_ebit_ta"].isna()
        | df["x4_eq_tl"].isna()
    )
    df.loc[missing_mask, "zscore_zpp"] = np.nan

    return df


def assign_3class_label(zscore: Optional[float]) -> Optional[str]:
    """
    Assign 3-class label based on Z'' score.

    Parameters
    ----------
    zscore : float or None
        Z'' score value.

    Returns
    -------
    str or None
        'distress', 'gray', or 'safe'; None if zscore is None.
    """
    if pd.isna(zscore):
        return None
    if zscore < Z_DISTRESS_THRESHOLD:
        return "distress"
    elif zscore <= Z_SAFE_THRESHOLD:
        return "gray"
    else:
        return "safe"


def calculate_3class_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 3-class labels based on Z'' score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Z'' score.

    Returns
    -------
    pd.DataFrame
        DataFrame with label column.
    """
    df = df.copy()
    df["label_3class"] = df["zscore_zpp"].apply(assign_3class_label)
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features: ratios, Z'' score, and labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw financial data.

    Returns
    -------
    pd.DataFrame
        DataFrame with all calculated features.
    """
    logger.info("Calculating liquidity ratios...")
    df = calculate_liquidity_ratios(df)

    logger.info("Calculating leverage ratios...")
    df = calculate_leverage_ratios(df)

    logger.info("Calculating profitability ratios...")
    df = calculate_profitability_ratios(df)

    logger.info("Calculating efficiency ratios...")
    df = calculate_efficiency_ratios(df)

    logger.info("Calculating Z'' score...")
    df = calculate_zscore_zpp(df)

    logger.info("Assigning 3-class labels...")
    df = calculate_3class_labels(df)

    # Track EBIT source
    if "EBIT_source" in df.columns:
        df["ebit_source_used"] = df["EBIT_source"]
    else:
        df["ebit_source_used"] = None

    logger.info(f"Built features for {len(df)} records")

    return df


if __name__ == "__main__":
    # Test the module
    test_data = pd.DataFrame({
        "cik": [320193],
        "fiscal_year": [2023],
        "Assets": [352583000000],
        "Liabilities": [290437000000],
        "AssetsCurrent": [143566000000],
        "LiabilitiesCurrent": [145308000000],
        "StockholdersEquity": [62146000000],
        "RetainedEarnings": [-214000000],
        "EBIT": [119437000000],
        "EBIT_source": ["OperatingIncomeLoss"],
        "NetIncome": [97000000000],
        "Revenue": [383285000000],
    })

    result = build_all_features(test_data)
    print(result.T)
