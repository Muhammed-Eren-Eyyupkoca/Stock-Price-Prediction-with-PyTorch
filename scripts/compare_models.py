"""Trains LSTMModel and GRUModel separately with identical hyperparameters and
data, then compares their final validation loss and training time side by side.

Usage:
    python scripts/compare_models.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import StockDataset  # noqa: E402
from models import GRUModel, LSTMModel  # noqa: E402
from train import train_one_epoch, validate  # noqa: E402

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

LOOKBACK = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15


def train_and_evaluate(
    model_cls: type,
    model_name: str,
    input_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict:
    model = model_cls(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    start = time.time()
    final_val_loss = float("inf")
    for _ in range(NUM_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, criterion)
        final_val_loss = validate(model, val_loader, criterion)
    training_time_sec = time.time() - start

    print(f"{model_name}: final val_loss={final_val_loss:.6f}, eğitim süresi={training_time_sec:.2f}s")
    return {"model": model_name, "final_val_loss": final_val_loss, "training_time_sec": training_time_sec}


def main() -> int:
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv", index_col="Date", parse_dates=True)
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv", index_col="Date", parse_dates=True)

    train_dataset = StockDataset(train_df, lookback=LOOKBACK)
    val_dataset = StockDataset(val_df, lookback=LOOKBACK)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_size = train_df.shape[1]

    results = [
        train_and_evaluate(LSTMModel, "LSTM", input_size, train_loader, val_loader),
        train_and_evaluate(GRUModel, "GRU", input_size, train_loader, val_loader),
    ]

    results_df = pd.DataFrame(results)
    print("\nLSTM vs GRU karşılaştırması:")
    print(results_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
