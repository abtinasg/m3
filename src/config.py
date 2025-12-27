"""Configuration settings for SEC EDGAR pipeline."""

import os

# SEC API Configuration
SEC_BASE_URL = "https://www.sec.gov"
SEC_DATA_URL = "https://data.sec.gov"
COMPANY_TICKERS_URL = f"{SEC_BASE_URL}/files/company_tickers_exchange.json"

# User-Agent is required by SEC (change this to your own email)
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Mozilla/5.0 (compatible; SEC-DataPipeline/1.0; Contact: user@example.com)"
)

# Rate limiting
REQUEST_DELAY = 0.2  # seconds between requests

# Data settings
START_YEAR = 2010
END_YEAR = 2024

# Cache directory
CACHE_DIR = os.environ.get("SEC_CACHE_DIR", "./cache")
OUTPUT_DIR = os.environ.get("SEC_OUTPUT_DIR", "./output")

# Z'' score thresholds for 3-class classification
Z_DISTRESS_THRESHOLD = 1.1
Z_SAFE_THRESHOLD = 2.6
