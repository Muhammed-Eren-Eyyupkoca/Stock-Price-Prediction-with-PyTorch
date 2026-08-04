"""Manual grid search over learning_rate, hidden_size and lookback for LSTMModel.

Optuna is not installed in this project's environment, so this uses the manual
grid-search alternative mentioned in the task: each combination is trained for a
few epochs and scored by its best validation loss, then the top 3 combinations
are printed as a pandas DataFrame.

Usage:
    python scripts/hyperparameter_search.py
"""

import sys
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import StockDataset  # noqa: E402
from models import LSTMModel  # noqa: E402
from train import train_one_epoch, validate  # noqa: E402

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

LEARNING_RATES = [1e-2, 1e-3, 1e-4]
HIDDEN_SIZES = [32, 64]
LOOKBACKS = [15, 30]

NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_SIZE = 32
EPOCHS_PER_TRIAL = 5
TOP_N = 3


def run_trial(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    learning_rate: float,
    hidden_size: int,
    lookback: int,
) -> float:
    train_dataset = StockDataset(train_df, lookback=lookback)
    val_dataset = StockDataset(val_df, lookback=lookback)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(train_df.shape[1], hidden_size, NUM_LAYERS, OUTPUT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for _ in range(EPOCHS_PER_TRIAL):
        train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        best_val_loss = min(best_val_loss, val_loss)

    return best_val_loss


def main() -> int:
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv", index_col="Date", parse_dates=True)
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv", index_col="Date", parse_dates=True)

    combinations = list(product(LEARNING_RATES, HIDDEN_SIZES, LOOKBACKS))
    results = []

    for trial_num, (lr, hidden_size, lookback) in enumerate(combinations, start=1):
        val_loss = run_trial(train_df, val_df, lr, hidden_size, lookback)
        print(
            f"[{trial_num}/{len(combinations)}] lr={lr}, hidden_size={hidden_size}, "
            f"lookback={lookback} -> val_loss={val_loss:.6f}"
        )
        results.append(
            {"learning_rate": lr, "hidden_size": hidden_size, "lookback": lookback, "val_loss": val_loss}
        )

    results_df = pd.DataFrame(results).sort_values("val_loss").reset_index(drop=True)

    print(f"\nEn iyi {TOP_N} kombinasyon:")
    print(results_df.head(TOP_N).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
