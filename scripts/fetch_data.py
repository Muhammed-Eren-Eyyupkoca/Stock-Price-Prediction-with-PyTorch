"""Download the last 5 years of daily OHLCV data for one or more stock symbols via
yfinance and save each as data/raw/<symbol>.csv.

Usage:
    python scripts/fetch_data.py AAPL
    python scripts/fetch_data.py AAPL --years 5
    python scripts/fetch_data.py THYAO.IS GARAN.IS AKBNK.IS
    python scripts/fetch_data.py --list-file scripts/symbols/bist30.txt
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily OHLCV data for one or more stock symbols and save each as CSV."
    )
    parser.add_argument(
        "symbols", nargs="*", help="One or more stock ticker symbols, e.g. AAPL THYAO.IS"
    )
    parser.add_argument(
        "--list-file",
        type=Path,
        help="Path to a text file with one ticker symbol per line (# comments allowed)",
    )
    parser.add_argument(
        "--years", type=int, default=5, help="Number of years of history to fetch (default: 5)"
    )
    return parser.parse_args()


def load_symbols(args: argparse.Namespace) -> list[str]:
    symbols = list(args.symbols)

    if args.list_file:
        if not args.list_file.exists():
            raise FileNotFoundError(f"Sembol listesi dosyası bulunamadı: '{args.list_file}'")
        for line in args.list_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line)

    seen: set[str] = set()
    unique_symbols = []
    for symbol in symbols:
        symbol = symbol.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            unique_symbols.append(symbol)

    return unique_symbols


def fetch_ohlcv(symbol: str, years: int) -> pd.DataFrame:
    end = date.today()
    start = end.replace(year=end.year - years)

    try:
        data = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(f"'{symbol}' sembolü için veri indirilirken hata oluştu: {exc}") from exc

    if data.empty:
        raise ValueError(
            f"'{symbol}' sembolü bulunamadı veya bu sembol için veri döndürülmedi. "
            "Sembolü kontrol edip tekrar deneyin."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def get_opening_price(symbol: str) -> float:
    """Returns `symbol`'s opening price for today (the exchange's opening
    auction print, ~09:55-10:00 Turkey time), read from today's 5-minute
    intraday bars.

    Raises:
        RuntimeError: If today has no intraday data yet (market not open /
            not a trading day) or the price can't be retrieved.
    """
    try:
        history = yf.Ticker(symbol).history(period="1d", interval="5m")
    except Exception as exc:
        raise RuntimeError(f"'{symbol}' için güne başlangıç fiyatı alınamadı: {exc}") from exc

    if history.empty:
        raise RuntimeError(
            f"'{symbol}' için bugüne ait gün içi veri yok (piyasa açık olmayabilir)."
        )

    history = history.tz_convert("Europe/Istanbul")

    today = pd.Timestamp.now(tz="Europe/Istanbul").date()
    todays_bars = history[history.index.date == today]
    if todays_bars.empty:
        raise RuntimeError(
            f"'{symbol}' için bugüne ait gün içi veri yok (piyasa açık olmayabilir)."
        )

    return float(todays_bars.iloc[0]["Open"])


def fetch_and_save(symbol: str, years: int) -> int:
    try:
        data = fetch_ohlcv(symbol, years)
    except (ValueError, RuntimeError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 0

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / f"{symbol}.csv"
    data.to_csv(output_path)

    print(f"'{symbol}' için {len(data)} satır OHLCV verisi '{output_path}' konumuna kaydedildi.")
    return 1


def main() -> int:
    args = parse_args()

    try:
        symbols = load_symbols(args)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1

    if not symbols:
        print(
            "Hata: En az bir sembol belirtmelisiniz (pozisyonel argüman olarak ya da --list-file ile).",
            file=sys.stderr,
        )
        return 1

    success_count = sum(fetch_and_save(symbol, args.years) for symbol in symbols)
    failure_count = len(symbols) - success_count

    if len(symbols) > 1:
        print(f"Toplam {len(symbols)} sembolden {success_count} başarılı, {failure_count} başarısız.")

    return 1 if failure_count else 0


if __name__ == "__main__":
    sys.exit(main())
