"""Bulk-download the last 3 years of daily OHLCV data for all BIST-30 stocks via
yfinance, saving each as data/raw/<symbol>.csv (without the ".IS" suffix).

Continues past per-symbol failures (empty data / request errors) and prints a
success/failure summary at the end.

Usage:
    python scripts/fetch_bist30.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REQUEST_DELAY_SECONDS = 0.5

BIST30_SYMBOLS = [
    "AEFES.IS",
    "AKBNK.IS",
    "ASELS.IS",
    "ASTOR.IS",
    "BIMAS.IS",
    "DSTKF.IS",
    "EKGYO.IS",
    "ENKAI.IS",
    "EREGL.IS",
    "FROTO.IS",
    "GARAN.IS",
    "GUBRF.IS",
    "ISCTR.IS",
    "KCHOL.IS",
    "KRDMD.IS",
    "MGROS.IS",
    "PETKM.IS",
    "PGSUS.IS",
    "SAHOL.IS",
    "SASA.IS",
    "SISE.IS",
    "TAVHL.IS",
    "TCELL.IS",
    "THYAO.IS",
    "TOASO.IS",
    "TRALT.IS",
    "TTKOM.IS",
    "TUPRS.IS",
    "VAKBN.IS",
    "YKBNK.IS",
]


def fetch_symbol(symbol: str) -> pd.DataFrame:
    """Downloads 3 years of daily OHLCV data for `symbol` via yfinance.

    Raises:
        ValueError: If yfinance returns no data for `symbol`.
    """
    data = yf.Ticker(symbol).history(period="3y", interval="1d")
    if data.empty:
        raise ValueError(f"'{symbol}' için veri döndürülmedi.")
    return data


def save_symbol_data(symbol: str, data: pd.DataFrame) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / f"{symbol.removesuffix('.IS')}.csv"
    data.to_csv(output_path)
    return output_path


def fetch_all(symbols: list[str]) -> pd.DataFrame:
    """Fetches and saves `symbols` one by one, sleeping between requests and
    skipping past failures.

    Returns:
        A summary DataFrame with one row per symbol: symbol, status, rows.
    """
    rows = []

    for i, symbol in enumerate(symbols):
        try:
            data = fetch_symbol(symbol)
            output_path = save_symbol_data(symbol, data)
            print(f"[OK]   '{symbol}': {len(data)} satır -> '{output_path}'")
            rows.append({"symbol": symbol, "status": "başarılı", "rows": len(data)})
        except Exception as exc:
            print(f"[HATA] '{symbol}': {exc}")
            rows.append({"symbol": symbol, "status": "başarısız", "rows": 0})

        if i < len(symbols) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    succeeded = summary.loc[summary["status"] == "başarılı", "symbol"].tolist()
    failed = summary.loc[summary["status"] == "başarısız", "symbol"].tolist()

    print("\n--- Özet ---")
    print(f"Başarılı ({len(succeeded)}): {', '.join(succeeded) if succeeded else '-'}")
    print(f"Başarısız ({len(failed)}): {', '.join(failed) if failed else '-'}")
    print()
    print(summary.to_string(index=False))


def main() -> int:
    summary = fetch_all(BIST30_SYMBOLS)
    print_summary(summary)
    return 1 if (summary["status"] == "başarısız").any() else 0


if __name__ == "__main__":
    sys.exit(main())
