"""Build the processed feature set for one stock symbol from data/raw/<symbol>.csv,
split it chronologically into train/val/test (no shuffling), fit a StandardScaler on
the train split only, and save the three scaled splits under data/processed/. The
fitted scaler is also saved (data/processed/scaler.pkl) so predictions can later
be inverse-transformed back to real units (see scripts/evaluate.py).

The prediction target is "Return", the day-over-day pct_change() of Close, rather
than the raw Close price. Raw price levels drift outside the range a scaler was
fit on (a rallying stock's test-period prices can end up far above anything seen
in train), which made a price-target model's predictions collapse toward a flat
line on the test set. Returns stay in a narrow, roughly stationary range across
train/val/test, so the model doesn't have to extrapolate.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --symbol THYAO.IS
    python scripts/preprocess.py --symbol GARAN.IS --train-frac 0.7 --val-frac 0.15
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kronolojik train/val/test bölme ve StandardScaler ile ölçekleme uygular."
    )
    parser.add_argument(
        "--symbol", default="THYAO.IS", help="data/raw/ altındaki hisse sembolü (varsayılan: THYAO.IS)"
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.70, help="Train oranı (varsayılan: 0.70)"
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15, help="Validation oranı (varsayılan: 0.15)"
    )
    return parser.parse_args()


def load_and_clean(symbol: str) -> pd.DataFrame:
    raw_path = RAW_DATA_DIR / f"{symbol}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Ham veri dosyası bulunamadı: '{raw_path}'")

    df = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
    df = df.ffill().dropna()
    return df


def add_return_target(df: pd.DataFrame) -> pd.DataFrame:
    """Adds "Return", the day-over-day pct_change() of Close, as the prediction
    target column. Drops the first row (pct_change() has no prior day there)."""
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    return df.dropna(subset=["Return"])


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if HAS_PANDAS_TA:
        df["SMA_20"] = ta.sma(df["Close"], length=20)
        df["RSI_14"] = ta.rsi(df["Close"], length=14)
        macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        df["MACD"] = macd_df["MACD_12_26_9"]
        df["MACD_Signal"] = macd_df["MACDs_12_26_9"]
        df["MACD_Hist"] = macd_df["MACDh_12_26_9"]
    else:
        df["SMA_20"] = df["Close"].rolling(window=20).mean()

        def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        df["RSI_14"] = compute_rsi(df["Close"], period=14)

        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # SMA/RSI/MACD ısınma pencereleri baştaki satırlarda NaN üretir
    return df.dropna()


def chronological_split(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac ve val_frac (0, 1) aralığında olmalı ve toplamları 1'den küçük olmalı")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def scale_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fits a StandardScaler on `train_df` only, then transforms all three splits
    with it (val/test never influence the fit — no lookahead into the scaler)."""
    scaler = StandardScaler()
    scaler.fit(train_df)

    train_scaled = pd.DataFrame(
        scaler.transform(train_df), index=train_df.index, columns=train_df.columns
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df), index=val_df.index, columns=val_df.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df), index=test_df.index, columns=test_df.columns
    )
    return train_scaled, val_scaled, test_scaled, scaler


def preprocess_symbol(symbol: str, train_frac: float = 0.70, val_frac: float = 0.15) -> dict:
    """Runs the full preprocessing pipeline for one symbol and writes
    train.csv/val.csv/test.csv/scaler.pkl under data/processed/.

    Args:
        symbol: data/raw/<symbol>.csv to read.
        train_frac: Fraction of rows (chronologically) used for training.
        val_frac: Fraction of rows used for validation (test gets the remainder).

    Returns:
        Row counts: {"train": int, "val": int, "test": int}.

    Raises:
        FileNotFoundError: If data/raw/<symbol>.csv does not exist.
    """
    df = load_and_clean(symbol)
    df = add_return_target(df)
    df = add_technical_indicators(df)
    train_df, val_df, test_df = chronological_split(df, train_frac, val_frac)
    train_scaled, val_scaled, test_scaled, scaler = scale_splits(train_df, val_df, test_df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_scaled.to_csv(PROCESSED_DATA_DIR / "train.csv")
    val_scaled.to_csv(PROCESSED_DATA_DIR / "val.csv")
    test_scaled.to_csv(PROCESSED_DATA_DIR / "test.csv")
    joblib.dump(scaler, PROCESSED_DATA_DIR / "scaler.pkl")

    return {"train": len(train_scaled), "val": len(val_scaled), "test": len(test_scaled)}


def main() -> int:
    args = parse_args()

    try:
        counts = preprocess_symbol(args.symbol, args.train_frac, args.val_frac)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1

    print(
        f"'{args.symbol}' için işlenmiş veri bölündü ve ölçeklendi: "
        f"train={counts['train']}, val={counts['val']}, test={counts['test']} satır."
    )
    print(f"Kaydedildi: {PROCESSED_DATA_DIR} (train.csv, val.csv, test.csv, scaler.pkl)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
