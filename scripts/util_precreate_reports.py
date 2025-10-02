#!/usr/bin/env python3
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    paths = [
        root / "reports",
        root / "reports" / "debug_enriched",
    ]
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
        print(f"[OK] ensured → {p}")

if __name__ == "__main__":
    main()