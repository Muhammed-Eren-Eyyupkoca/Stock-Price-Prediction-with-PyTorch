"""Training loop for the LSTM/GRU stock price models."""

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import StockDataset
from models import LSTMModel


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
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


if __name__ == "__main__":
    lookback, hidden_size, num_layers, output_size = 30, 64, 2, 1

    train_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "train.csv"
    train_df = pd.read_csv(train_path, index_col="Date", parse_dates=True)

    train_dataset = StockDataset(train_df, lookback=lookback)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    input_size = train_df.shape[1]
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    print(f"Epoch 1 - ortalama loss: {epoch_loss:.6f}")
