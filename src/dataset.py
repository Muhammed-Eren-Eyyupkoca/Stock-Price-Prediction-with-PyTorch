"""PyTorch Dataset for sliding-window stock price sequences."""

from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Windowed dataset over a scaled feature table (e.g. data/processed/train.csv).

    Sample i is (X, y) where X is the `lookback` days ending at day i (all feature
    columns) and y is the target column's value on the following day (by default
    "Return", the day-over-day pct_change() of Close — see scripts/preprocess.py).
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        lookback: int,
        target_col: Union[str, int] = "Return",
    ) -> None:
        """Builds the windowed feature/target tensors from a scaled feature table.

        Args:
            data: Scaled feature table, either a DataFrame (columns are feature
                names) or a numpy array (columns are feature positions).
            lookback: Number of preceding days used as the input window for each sample.
            target_col: Prediction-target column name (DataFrame) or column index
                (numpy array).

        Raises:
            ValueError: If lookback < 1, or data has too few rows for that lookback.
            TypeError: If a numpy array is passed with a string target_col.
        """
        if lookback < 1:
            raise ValueError("lookback en az 1 olmalı")

        if isinstance(data, pd.DataFrame):
            target_idx = data.columns.get_loc(target_col) if isinstance(target_col, str) else target_col
            values = data.to_numpy(dtype=np.float32)
        else:
            if isinstance(target_col, str):
                raise TypeError(
                    "numpy array girişinde target_col bir sütun adı değil, tam sayı indeks olmalı"
                )
            values = np.asarray(data, dtype=np.float32)
            target_idx = target_col

        if len(values) <= lookback:
            raise ValueError(
                f"Veri seti çok kısa: {len(values)} satır, lookback={lookback} için en az "
                f"{lookback + 1} satır gerekli"
            )

        self.lookback = lookback
        self.target_idx = target_idx
        self.features = torch.from_numpy(values)

    def __len__(self) -> int:
        """Returns the number of (X, y) windows the dataset can produce."""
        return len(self.features) - self.lookback

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns sample `idx`.

        Args:
            idx: Window start position, in [0, len(self)).

        Returns:
            A tuple (X, y): X has shape (lookback, n_features) and holds the
            `lookback` days ending at `idx`; y has shape (1,) and holds the
            target column's value on the following day.
        """
        x = self.features[idx : idx + self.lookback]
        y = self.features[idx + self.lookback, self.target_idx].unsqueeze(0)
        return x, y
