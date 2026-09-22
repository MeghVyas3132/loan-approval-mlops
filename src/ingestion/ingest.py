"""DVC stage 1 - data ingestion.

Reads the DVC-tracked raw CSV, applies the only transformation that belongs
before validation (encoding the target to 0/1), and produces a stratified
train/test split. Splitting here - rather than inside the training script -
guarantees the test set is untouched by every later stage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config, load_config
from src.utils.io import write_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_MAP = {"Y": 1, "N": 0, "y": 1, "n": 0, 1: 1, 0: 0, "1": 1, "0": 0}


def encode_target(series: pd.Series) -> pd.Series:
    """Map the raw ``Y``/``N`` label onto 1/0, failing loudly on unknown values."""
    encoded = series.map(TARGET_MAP)
    unknown = series[encoded.isna()].unique().tolist()
    if unknown:
        raise ValueError(f"Unrecognised target values: {unknown}")
    return encoded.astype(int)


def split_dataset(
    frame: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_on = frame[target_column] if stratify else None
    train_df, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_on,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def ingest(config: Config | None = None) -> tuple[Path, Path]:
    config = config or load_config()
    data_cfg = config.section("data")
    target = config.get_nested("base.target_column")

    raw_path = config.get_path("data.raw_path")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run `dvc pull` or "
            "`python -m src.ingestion.generate_data` first."
        )

    frame = pd.read_csv(raw_path)
    logger.info("Loaded raw dataset %s with shape %s", raw_path.name, frame.shape)

    before = len(frame)
    frame = frame.drop_duplicates(subset=["Loan_ID"]) if "Loan_ID" in frame else frame.drop_duplicates()
    if before != len(frame):
        logger.warning("Dropped %d duplicate applications", before - len(frame))

    frame[target] = encode_target(frame[target])

    train_df, test_df = split_dataset(
        frame,
        target_column=target,
        test_size=float(data_cfg["test_size"]),
        random_state=int(config.get_nested("base.random_state", 42)),
        stratify=bool(data_cfg.get("stratify", True)),
    )

    train_path = write_csv(config.get_path("data.train_path"), train_df)
    test_path = write_csv(config.get_path("data.test_path"), test_df)
    logger.info("Train %s -> %s | Test %s -> %s", train_df.shape, train_path, test_df.shape, test_path)
    return train_path, test_path


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    ingest()
