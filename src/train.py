"""Training loop for the LSTM/GRU stock price models."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import StockDataset
from models import LSTMModel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Runs one training epoch: forward, backward and an optimizer step per batch.

    Args:
        model: Model to train; switched to train() mode.
        dataloader: Yields (X, y) batches.
        optimizer: Optimizer updating model.parameters().
        criterion: Loss function applied to (predictions, y).

    Returns:
        Mean loss over the epoch, weighted by batch size.
    """
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(dataloader.dataset)


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> float:
    """Evaluates `model` on `dataloader` without updating its weights.

    Args:
        model: Model to evaluate; switched to eval() mode.
        dataloader: Yields (X, y) batches.
        criterion: Loss function applied to (predictions, y).

    Returns:
        Mean loss over the dataloader, weighted by batch size.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(dataloader.dataset)


if __name__ == "__main__":
    lookback, hidden_size, num_layers, output_size = 30, 64, 2, 1
    batch_size = 32
    num_epochs = 25

    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    train_df = pd.read_csv(processed_dir / "train.csv", index_col="Date", parse_dates=True)
    val_df = pd.read_csv(processed_dir / "val.csv", index_col="Date", parse_dates=True)

    train_dataset = StockDataset(train_df, lookback=lookback)
    val_dataset = StockDataset(val_df, lookback=lookback)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = train_df.shape[1]
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_model.pt"

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch} - train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> yeni en iyi val loss ({best_val_loss:.6f}), model '{best_model_path}' olarak kaydedildi.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    loss_curve_path = RESULTS_DIR / "loss_curve.png"

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTMModel - Train vs. Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    print(f"Loss eğrisi '{loss_curve_path}' olarak kaydedildi.")
