"""Build the training dataset parquet from the ingested tables.

Run from backend/:  python -m ml.dataset
"""
import pandas as pd

import config
from ml.features import build_training_dataset

DATASET_PARQUET = config.PROCESSED_DIR / "dataset.parquet"


def build() -> pd.DataFrame:
    results = pd.read_parquet(config.RESULTS_PARQUET)
    races = pd.read_parquet(config.RACES_PARQUET)
    df = build_training_dataset(results, races)
    df.to_parquet(DATASET_PARQUET, index=False)
    return df


if __name__ == "__main__":
    config.ensure_dirs()
    df = build()
    n_races = df.groupby(["season", "round"]).ngroups
    print(f"Dataset: {len(df)} rows across {n_races} races -> {DATASET_PARQUET}")
