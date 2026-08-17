"""Compares the old price-target pipeline (MinMaxScaler, raw Close as the
prediction target) against the new return-target pipeline (StandardScaler,
day-over-day Return as the target — see scripts/preprocess.py) on ASELS.IS.

The old-pipeline numbers below are a fixed baseline measured before the fix,
on data/raw/ASELS.IS.csv, with the same 70/15/15 chronological split and the
same LSTM architecture/hyperparameters as scripts/evaluate.py: MinMaxScaler
fit on train only, Close used directly as the StockDataset target. Test-period
prices for ASELS.IS reach ~4.8x the train-period max, which is outside the
range MinMaxScaler was fit on — its predictions collapsed toward a near-flat
line at the edge of the fitted range.

The new-pipeline numbers are computed live by calling scripts/evaluate.py's
run_evaluation() on whatever is currently in data/processed/ (run
scripts/preprocess.py --symbol ASELS.IS and src/train.py first).

Usage:
    python scripts/compare_price_vs_return.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import run_evaluation  # noqa: E402

# Fixed baseline, measured on ASELS.IS before the price->return fix (git history
# has the pre-fix scripts/preprocess.py, src/dataset.py, scripts/evaluate.py).
OLD_PIPELINE_RESULTS = {"rmse": 218.9908, "mae": 214.4078, "mape": 60.20}


def main() -> int:
    try:
        new_results = run_evaluation()
    except FileNotFoundError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1

    comparison = pd.DataFrame(
        [
            {
                "pipeline": "Eski (fiyat hedefi, MinMaxScaler)",
                "RMSE": OLD_PIPELINE_RESULTS["rmse"],
                "MAE": OLD_PIPELINE_RESULTS["mae"],
                "MAPE (%)": OLD_PIPELINE_RESULTS["mape"],
            },
            {
                "pipeline": "Yeni (getiri hedefi, StandardScaler)",
                "RMSE": new_results["rmse"],
                "MAE": new_results["mae"],
                "MAPE (%)": new_results["mape"],
            },
        ]
    )

    print("\nASELS.IS - Eski (fiyat bazlı) vs Yeni (getiri bazlı) pipeline karşılaştırması:")
    print(comparison.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
