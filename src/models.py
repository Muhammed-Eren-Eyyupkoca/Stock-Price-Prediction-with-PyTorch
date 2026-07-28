"""LSTM sequence model for next-day Close price prediction."""

import torch
from torch import nn


class LSTMModel(nn.Module):
    """Many-to-one LSTM: consumes a (batch, lookback, input_size) window and
    predicts a single value from the last time step's hidden state."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)


if __name__ == "__main__":
    batch_size, lookback, input_size = 32, 30, 11
    hidden_size, num_layers, output_size = 64, 2, 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    sample_batch = torch.randn(batch_size, lookback, input_size)

    prediction = model(sample_batch)

    print(f"Girdi shape: {tuple(sample_batch.shape)}")
    print(f"Çıktı shape: {tuple(prediction.shape)} (beklenen: ({batch_size}, {output_size}))")
