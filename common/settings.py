# QuantETFUS_small/common/settings.py

# Use the shared Excel and force the small-universe tab
ETF_LIST_PATH = "/Users/Finance/QuantShared/ETF_list.xlsx"
DEFAULT_ETF_SHEET = "US_small"

def default_etf_sheet() -> str:
    return DEFAULT_ETF_SHEET

# Keep US bucket
TARGET_BUCKET = "targeted_ETFs_US"