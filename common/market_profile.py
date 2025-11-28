from dataclasses import dataclass
from pathlib import Path

@dataclass
class MarketProfile:
    name: str
    tz: str
    calendar: str        # pandas_market_calendars code
    currency: str
    root: Path

US = MarketProfile(
    name="US",
    tz="America/New_York",
    calendar="XNYS",
    currency="USD",
    root=Path(__file__).resolve().parents[1]
)