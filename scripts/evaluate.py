"""Loads the best checkpoint (models/best_model.pt), runs it on the test set,
reconstructs real Close prices from the model's predicted Return values, prints
RMSE/MAE/MAPE, and plots predicted vs. actual price to results/predictions_vs_actual.png.

The model predicts "Return" (day-over-day pct_change of Close), not price
directly (see scripts/preprocess.py). To compare against real prices, each
prediction is turned back into a price via:

    predicted_price[t] = actual_price[t-1] * (1 + predicted_return[t])

i.e. the predicted return is applied to the previous day's REAL (not predicted)
closing price. actual_price[t-1] comes from inverse-transforming the "Close"
feature column, which is still carried alongside "Return" in the processed data.

Usage:
    python scripts/evaluate.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import StockDataset  # noqa: E402
from models import LSTMModel  # noqa: E402

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Must match the architecture best_model.pt was trained with (see src/train.py).
LOOKBACK = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_SIZE = 32


def inverse_transform_column(scaler, values: np.ndarray, column_idx: int, n_features: int) -> np.ndarray:
    """Inverse-transforms a single scaled column via a zero-filled dummy matrix,
    since StandardScaler.inverse_transform expects all fitted feature columns."""
    dummy = np.zeros((len(values), n_features))
    dummy[:, column_idx] = values
    return scaler.inverse_transform(dummy)[:, column_idx]


def returns_to_prices(prev_prices: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Reconstructs price[t] = prev_prices[t] * (1 + returns[t]), applying each
    return to the REAL previous-day price rather than a predicted one, so errors
    don't compound across the test set."""
    return prev_prices * (1 + returns)


def verify_inverse_transform(scaler, test_df: pd.DataFrame, close_idx: int, n_features: int) -> None:
    """Sanity-checks the two inverse-transform steps this module relies on:

    1. scaler.inverse_transform(scaler.transform(x)) == x (round-trip on the
       already-scaled test_df, so this also implicitly checks
       inverse_transform_column against the real scaler).
    2. returns_to_prices reconstructs the REAL actual Close prices from the
       REAL actual Returns — since Return was computed as an exact pct_change()
       of Close in preprocess.py, this should round-trip almost exactly.

    Raises:
        AssertionError: If either round-trip check is off by more than a small
            floating-point tolerance.
    """
    scaled_sample = test_df.iloc[:5]
    unscaled_sample = pd.DataFrame(
        scaler.inverse_transform(scaled_sample), index=scaled_sample.index, columns=scaled_sample.columns
    )
    round_tripped = scaler.transform(unscaled_sample)
    assert np.allclose(scaled_sample.to_numpy(), round_tripped, atol=1e-4), (
        "Scaler inverse_transform round-trip başarısız: transform(inverse_transform(x)) != x"
    )
    print("[Doğrulama] scaler.inverse_transform round-trip OK (transform edilmiş değer geri çevrilince orijinaliyle eşleşiyor).")

    return_idx = test_df.columns.get_loc("Return")
    real_close = inverse_transform_column(
        scaler, test_df.iloc[:, close_idx].to_numpy(), close_idx, n_features
    )
    real_return = inverse_transform_column(
        scaler, test_df.iloc[:, return_idx].to_numpy(), return_idx, n_features
    )
    reconstructed = returns_to_prices(real_close[:-1], real_return[1:])
    assert np.allclose(real_close[1:], reconstructed, rtol=1e-3), (
        "Getiri->fiyat dönüşümü başarısız: prev_close * (1 + actual_return) != actual_close"
    )
    print("[Doğrulama] getiri->fiyat dönüşümü OK (gerçek getiriden yeniden kurulan fiyat, gerçek kapanış fiyatıyla eşleşiyor).")


def run_evaluation() -> dict:
    """Runs best_model.pt on the test set and returns dates, real-price actuals/
    predictions (reconstructed from predicted Returns) and the RMSE/MAE/MAPE
    metrics. Raises FileNotFoundError if the preprocessing/training artifacts it
    depends on are missing."""
    test_path = PROCESSED_DATA_DIR / "test.csv"
    scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
    model_path = MODELS_DIR / "best_model.pt"

    for path in (test_path, scaler_path, model_path):
        if not path.exists():
            raise FileNotFoundError(
                f"'{path}' bulunamadı. Önce preprocess.py ve train.py çalıştırılmalı."
            )

    test_df = pd.read_csv(test_path, index_col="Date", parse_dates=True)
    scaler = joblib.load(scaler_path)
    close_idx = test_df.columns.get_loc("Close")
    n_features = test_df.shape[1]

    verify_inverse_transform(scaler, test_df, close_idx, n_features)

    test_dataset = StockDataset(test_df, lookback=LOOKBACK)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(n_features, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaled_preds = []
    with torch.no_grad():
        for x_batch, _y_batch in test_loader:
            predictions = model(x_batch)
            scaled_preds.append(predictions.squeeze(1).numpy())

    scaled_preds = np.concatenate(scaled_preds)

    return_idx = test_df.columns.get_loc("Return")
    predicted_returns = inverse_transform_column(scaler, scaled_preds, return_idx, n_features)

    # Sample i's target is the value 'LOOKBACK' days after test_df.index[i]
    dates = test_df.index[LOOKBACK:]

    real_close = inverse_transform_column(
        scaler, test_df["Close"].to_numpy(), close_idx, n_features
    )
    prev_close = real_close[LOOKBACK - 1 : -1]  # real Close the day before each target day
    actuals = real_close[LOOKBACK:]  # real Close on each target day (ground truth, no reconstruction)
    preds = returns_to_prices(prev_close, predicted_returns)

    return {
        "dates": dates,
        "actuals": actuals,
        "preds": preds,
        "rmse": mean_squared_error(actuals, preds) ** 0.5,
        "mae": mean_absolute_error(actuals, preds),
        "mape": mean_absolute_percentage_error(actuals, preds) * 100,
    }


def plot_predictions_vs_actual(dates, actuals: np.ndarray, preds: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(dates, actuals, label="Gerçek Fiyat", color="#2563EB")
    plt.plot(dates, preds, label="Tahmin Edilen Fiyat", color="#F59E0B")
    plt.xlabel("Tarih")
    plt.ylabel("Kapanış Fiyatı")
    plt.title("Test Seti: Gerçek vs. Tahmin Edilen Kapanış Fiyatı")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> int:
    try:
        results = run_evaluation()
    except FileNotFoundError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1

    print(f"Test seti örnek sayısı: {len(results['actuals'])}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE:  {results['mae']:.4f}")
    print(f"MAPE: {results['mape']:.2f}%")

    plot_path = RESULTS_DIR / "predictions_vs_actual.png"
    plot_predictions_vs_actual(results["dates"], results["actuals"], results["preds"], plot_path)
    print(f"Grafik kaydedildi: {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
