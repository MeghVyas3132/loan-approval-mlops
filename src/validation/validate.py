"""DVC stage 2 - data validation.

Checks the ingested split against :mod:`src.validation.schema` and writes a
machine-readable report. Any hard failure aborts the pipeline, so a bad data
drop can never reach training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Config, load_config
from src.utils.io import write_json
from src.utils.logger import get_logger
from src.validation.schema import LOAN_SCHEMA, TARGET_COLUMN

logger = get_logger(__name__)


class ValidationError(RuntimeError):
    """Raised when the incoming data breaks the contract."""


def _column_report(
    frame: pd.DataFrame, max_missing_fraction: float
) -> tuple[list[dict[str, Any]], list[str]]:
    reports: list[dict[str, Any]] = []
    errors: list[str] = []

    for spec in LOAN_SCHEMA.columns:
        if spec.name not in frame.columns:
            if spec.required:
                errors.append(f"missing required column '{spec.name}'")
            continue

        series = frame[spec.name]
        missing_fraction = float(series.isna().mean())
        entry: dict[str, Any] = {
            "column": spec.name,
            "dtype": spec.dtype,
            "missing_fraction": round(missing_fraction, 4),
        }

        if missing_fraction > max_missing_fraction:
            errors.append(
                f"'{spec.name}' is {missing_fraction:.1%} null (limit {max_missing_fraction:.0%})"
            )
        if not spec.nullable and series.isna().any():
            errors.append(f"'{spec.name}' contains nulls but is declared non-nullable")

        present = series.dropna()
        if spec.is_numeric():
            numeric = pd.to_numeric(present, errors="coerce")
            if numeric.isna().any():
                errors.append(f"'{spec.name}' contains non-numeric values")
            numeric = numeric.dropna()
            if not numeric.empty:
                entry.update(
                    min=float(numeric.min()),
                    max=float(numeric.max()),
                    mean=round(float(numeric.mean()), 4),
                )
                if spec.minimum is not None and numeric.min() < spec.minimum:
                    errors.append(f"'{spec.name}' below minimum {spec.minimum}")
                if spec.maximum is not None and numeric.max() > spec.maximum:
                    errors.append(f"'{spec.name}' above maximum {spec.maximum}")
        else:
            observed = sorted({str(v) for v in present.unique()})
            entry["observed_values"] = observed
            if spec.allowed_values:
                unexpected = sorted(set(observed) - set(spec.allowed_values))
                if unexpected:
                    errors.append(f"'{spec.name}' has unexpected categories {unexpected}")

        reports.append(entry)

    return reports, errors


def validate_frame(
    frame: pd.DataFrame,
    *,
    max_missing_fraction: float = 0.15,
    min_rows: int = 1,
    target_balance: tuple[float, float] | list[float] = (0.0, 1.0),
    require_target: bool = True,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Validate ``frame`` and return a report. Raises :class:`ValidationError` on failure."""
    errors: list[str] = []

    if len(frame) < min_rows:
        errors.append(f"only {len(frame)} rows, expected at least {min_rows}")

    column_reports, column_errors = _column_report(frame, max_missing_fraction)
    errors.extend(column_errors)

    target_summary: dict[str, Any] = {}
    if require_target:
        if TARGET_COLUMN not in frame.columns:
            errors.append(f"missing target column '{TARGET_COLUMN}'")
        else:
            target = frame[TARGET_COLUMN]
            if target.isna().any():
                errors.append("target column contains nulls")
            invalid = sorted(set(target.dropna().unique()) - {0, 1})
            if invalid:
                errors.append(f"target column must be 0/1, found {invalid}")
            else:
                positive_rate = float(target.mean())
                low, high = float(target_balance[0]), float(target_balance[1])
                target_summary = {
                    "positive_rate": round(positive_rate, 4),
                    "allowed_range": [low, high],
                }
                if not low <= positive_rate <= high:
                    errors.append(
                        f"target positive rate {positive_rate:.2%} outside [{low:.0%}, {high:.0%}]"
                    )

    report = {
        "dataset": dataset_name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "column_checks": column_reports,
        "target": target_summary,
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }

    if errors:
        logger.error("Validation failed for %s: %s", dataset_name, errors)
        raise ValidationError(f"{dataset_name}: " + "; ".join(errors))

    logger.info("Validation passed for %s (%d rows)", dataset_name, len(frame))
    return report


def validate_stage(config: Config | None = None) -> Path:
    """Validate both splits and persist a combined report."""
    config = config or load_config()
    validation_cfg = config.section("validation")

    reports = []
    for label, key in (("train", "data.train_path"), ("test", "data.test_path")):
        path = config.get_path(key)
        frame = pd.read_csv(path)
        reports.append(
            validate_frame(
                frame,
                max_missing_fraction=float(validation_cfg["max_missing_fraction"]),
                min_rows=int(validation_cfg["min_rows"]) if label == "train" else 1,
                target_balance=validation_cfg["expected_target_balance"],
                dataset_name=label,
            )
        )

    output = write_json(
        config.get_path("validation.report_path"),
        {"status": "passed", "datasets": reports},
    )
    logger.info("Validation report written to %s", output)
    return output


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    validate_stage()
