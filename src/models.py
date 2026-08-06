"""LSTM & GRU sequence models for next-day Close price prediction."""

import torch
from torch import nn


class LSTMModel(nn.Module):
    """Many-to-one LSTM: consumes a (batch, lookback, input_size) window and
    predicts a single value from the last time step's hidden state."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        """
        Args:
            input_size: Number of features per time step.
            hidden_size: Number of units in each LSTM layer.
            num_layers: Number of stacked LSTM layers.
            output_size: Size of the final prediction (1 for next-day Close).
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input batch of shape (batch, lookback, input_size).

        Returns:
            Predictions of shape (batch, output_size).
        """
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)


class GRUModel(nn.Module):
    """Same interface as LSTMModel (same input/output shape), backed by nn.GRU."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        """
        Args:
            input_size: Number of features per time step.
            hidden_size: Number of units in each GRU layer.
            num_layers: Number of stacked GRU layers.
            output_size: Size of the final prediction (1 for next-day Close).
        """
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input batch of shape (batch, lookback, input_size).

        Returns:
            Predictions of shape (batch, output_size).
        """
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        return self.fc(last_step)


if __name__ == "__main__":
    batch_size, lookback, input_size = 32, 30, 11
    hidden_size, num_layers, output_size = 64, 2, 1
    expected_shape = (batch_size, output_size)

    sample_batch = torch.randn(batch_size, lookback, input_size)

    for name, model_cls in [("LSTMModel", LSTMModel), ("GRUModel", GRUModel)]:
        model = model_cls(input_size, hidden_size, num_layers, output_size)
        prediction = model(sample_batch)

        print(f"{name} çıktı shape: {tuple(prediction.shape)} (beklenen: {expected_shape})")
        assert tuple(prediction.shape) == expected_shape, (
            f"{name} beklenmeyen shape üretti: {tuple(prediction.shape)} != {expected_shape}"
        )

    print("LSTMModel ve GRUModel aynı girdiyle aynı çıktı boyutunu üretiyor.")
