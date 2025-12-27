# SEC EDGAR Dataset Pipeline

A Python pipeline for building a financial dataset from SEC EDGAR 10-K filings for training a 3-class XGBoost model for financial distress prediction.

## Overview

This pipeline:
1. Downloads company tickers from SEC
2. Retrieves 10-K filings (2010-2024)
3. Extracts text sections (Item 1A - Risk Factors, Item 7 - MD&A)
4. Fetches XBRL financial data
5. Calculates financial ratios and Altman Z'' score
6. Exports a labeled dataset for machine learning

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd m3

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set environment variables (optional):

```bash
# Your contact email for SEC User-Agent (required by SEC)
export SEC_USER_AGENT="Mozilla/5.0 (compatible; YourApp/1.0; your@email.com)"

# Cache directory for downloaded data
export SEC_CACHE_DIR="./cache"

# Output directory for results
export SEC_OUTPUT_DIR="./output"
```

## Usage

### Run the Full Pipeline

```bash
python run_pipeline.py
```

### Command Line Options

```bash
# Limit to first 100 companies (for testing)
python run_pipeline.py --max-companies 100

# Custom year range
python run_pipeline.py --start-year 2015 --end-year 2023

# Output both parquet and CSV
python run_pipeline.py --output-formats parquet csv

# Skip text extraction (faster testing)
python run_pipeline.py --skip-text

# Custom output directory
python run_pipeline.py --output-dir ./my_output

# Disable caching (re-download everything)
python run_pipeline.py --no-cache
```

### Using Individual Modules

```python
from src.download_universe import get_company_universe
from src.get_filings import get_10k_filings_for_companies
from src.fetch_text import process_filings_text
from src.fetch_xbrl import get_financial_data_for_companies
from src.build_features import build_all_features
from src.export_dataset import export_dataset

# Step 1: Get company list
universe = get_company_universe()

# Step 2: Get 10-K filings
ciks = universe["cik"].head(100).tolist()
filings = get_10k_filings_for_companies(ciks)

# Step 3: Extract text
filings_with_text = process_filings_text(filings)

# Step 4: Get XBRL data
financial_data = get_financial_data_for_companies(ciks)

# Step 5: Build features
dataset = build_all_features(merged_data)

# Step 6: Export
export_dataset(dataset)
```

## Output Files

### Dataset (`firmyear_10k_2010_2024.parquet`)

| Column | Description |
|--------|-------------|
| `cik` | SEC Central Index Key |
| `ticker` | Stock ticker symbol |
| `company_name` | Company name |
| `accession` | Filing accession number |
| `filing_date` | Date filed with SEC |
| `report_date` | Period end date |
| `fiscal_year` | Fiscal year |
| `sic` | Standard Industrial Classification |
| `filing_url` | URL to original filing |
| `item1a_text` | Item 1A - Risk Factors text |
| `item7_text` | Item 7 - MD&A text |
| `Assets` | Total Assets |
| `Liabilities` | Total Liabilities |
| `AssetsCurrent` | Current Assets |
| `LiabilitiesCurrent` | Current Liabilities |
| `StockholdersEquity` | Stockholders' Equity |
| `RetainedEarnings` | Retained Earnings |
| `EBIT` | Earnings Before Interest and Taxes |
| `current_ratio` | Current Assets / Current Liabilities |
| `quick_ratio` | Quick Ratio |
| `cash_ratio` | Cash Ratio |
| `debt_to_assets` | Liabilities / Assets |
| `debt_to_equity` | Liabilities / Equity |
| `roa` | Return on Assets |
| `roe` | Return on Equity |
| `operating_margin` | EBIT / Revenue |
| `asset_turnover` | Revenue / Assets |
| `zscore_zpp` | Altman Z'' Score |
| `label_3class` | Classification label (distress/gray/safe) |
| `text_extract_ok` | Whether text extraction succeeded |
| `ebit_source_used` | XBRL tag used for EBIT |
| `missing_flags` | List of missing financial fields |

### Quality Summary (`summary.json`)

```json
{
  "total_rows": 15000,
  "total_companies": 1000,
  "year_range": {"min": 2010, "max": 2024},
  "class_distribution": {
    "distress": 1500,
    "gray": 5000,
    "safe": 8500
  },
  "text_extraction": {
    "success_rate": 85.5
  },
  "missing_data": {
    "Assets": 50,
    "EBIT": 200
  }
}
```

## Z'' Score Calculation

The Altman Z'' Score (for non-manufacturing/service firms) is calculated as:

```
Z'' = 6.56×X₁ + 3.26×X₂ + 6.72×X₃ + 1.05×X₄
```

Where:
- X₁ = (Current Assets - Current Liabilities) / Total Assets
- X₂ = Retained Earnings / Total Assets
- X₃ = EBIT / Total Assets
- X₄ = Equity / Total Liabilities

### Classification Thresholds

| Z'' Score | Classification |
|-----------|----------------|
| < 1.1 | Distress |
| 1.1 - 2.6 | Gray Zone |
| > 2.6 | Safe |

## Project Structure

```
m3/
├── README.md
├── requirements.txt
├── run_pipeline.py          # Main pipeline runner
└── src/
    ├── __init__.py
    ├── config.py            # Configuration settings
    ├── download_universe.py # Step 1: Download company list
    ├── get_filings.py       # Step 2: Get 10-K filings
    ├── fetch_text.py        # Step 3: Extract text sections
    ├── fetch_xbrl.py        # Step 4: Get XBRL financial data
    ├── build_features.py    # Step 5: Calculate ratios & Z''
    └── export_dataset.py    # Step 6: Export dataset
```

## Important Notes

1. **User-Agent Required**: SEC requires a User-Agent header with contact information
2. **Rate Limiting**: The pipeline includes delays between requests (0.2s default)
3. **Caching**: Data is cached to avoid re-downloading; use `--no-cache` to refresh
4. **Missing Data**: Records with missing XBRL data are kept with `missing_flags` set
5. **Text Extraction**: Some 10-K formats may not parse correctly; check `text_extract_ok`

## License

MIT License