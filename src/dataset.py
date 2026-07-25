"""PyTorch Dataset for sliding-window stock price sequences."""

from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Windowed dataset over a scaled feature table (e.g. data/processed/train.csv).

    Sample i is (X, y) where X is the `lookback` days ending at day i (all feature
    columns) and y is the Close price on the following day.
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        lookback: int,
        close_col: Union[str, int] = "Close",
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback en az 1 olmalı")

        if isinstance(data, pd.DataFrame):
            close_idx = data.columns.get_loc(close_col) if isinstance(close_col, str) else close_col
            values = data.to_numpy(dtype=np.float32)
        else:
            if isinstance(close_col, str):
                raise TypeError(
                    "numpy array girişinde close_col bir sütun adı değil, tam sayı indeks olmalı"
                )
            values = np.asarray(data, dtype=np.float32)
            close_idx = close_col

        if len(values) <= lookback:
            raise ValueError(
                f"Veri seti çok kısa: {len(values)} satır, lookback={lookback} için en az "
                f"{lookback + 1} satır gerekli"
            )

        self.lookback = lookback
        self.close_idx = close_idx
        self.features = torch.from_numpy(values)

    def __len__(self) -> int:
        return len(self.features) - self.lookback

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.lookback]
        y = self.features[idx + self.lookback, self.close_idx].unsqueeze(0)
        return x, y
