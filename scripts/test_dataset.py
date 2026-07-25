"""Sanity-check src/dataset.py's StockDataset against the processed train/val/test
splits: builds a DataLoader (batch_size=32, shuffle=False) for each split and
prints the shape of the first (X, y) batch.

Usage:
    python scripts/test_dataset.py
    python scripts/test_dataset.py --lookback 30 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import StockDataset  # noqa: E402

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StockDataset + DataLoader'ı train/val/test bölümleri üzerinde test eder."
    )
    parser.add_argument("--lookback", type=int, default=30, help="Pencere uzunluğu (varsayılan: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch boyutu (varsayılan: 32)")
    return parser.parse_args()


def check_split(name: str, csv_path: Path, lookback: int, batch_size: int) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"'{csv_path}' bulunamadı. Önce 'python scripts/preprocess.py' çalıştırılmalı."
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    n_features = df.shape[1]

    dataset = StockDataset(df, lookback=lookback)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x_batch, y_batch = next(iter(loader))

    expected_batch = min(batch_size, len(dataset))
    expected_x_shape = (expected_batch, lookback, n_features)
    expected_y_shape = (expected_batch, 1)

    print(f"[{name}] dataset uzunluğu: {len(dataset)}")
    print(f"[{name}] X shape: {tuple(x_batch.shape)} (beklenen: {expected_x_shape})")
    print(f"[{name}] y shape: {tuple(y_batch.shape)} (beklenen: {expected_y_shape})")

    if tuple(x_batch.shape) != expected_x_shape:
        raise AssertionError(f"[{name}] X shape uyuşmuyor: {tuple(x_batch.shape)} != {expected_x_shape}")
    if tuple(y_batch.shape) != expected_y_shape:
        raise AssertionError(f"[{name}] y shape uyuşmuyor: {tuple(y_batch.shape)} != {expected_y_shape}")

    print(f"[{name}] OK\n")


def main() -> int:
    args = parse_args()

    splits = {
        "train": PROCESSED_DATA_DIR / "train.csv",
        "val": PROCESSED_DATA_DIR / "val.csv",
        "test": PROCESSED_DATA_DIR / "test.csv",
    }

    try:
        for name, csv_path in splits.items():
            check_split(name, csv_path, args.lookback, args.batch_size)
    except (FileNotFoundError, AssertionError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1

    print("Tüm bölümler için StockDataset + DataLoader shape kontrolleri başarılı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
