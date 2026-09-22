"""Data-drift detection for the serving feed.

Compares a batch of production applications against the training distribution
using the Population Stability Index (PSI) - the standard measure in credit
risk. PSI < 0.1 means no meaningful shift, 0.1-0.25 warrants a look, and
> 0.25 means the model is scoring a population it was not trained on.

Run it on a schedule (CronJob in ``deployment/kubernetes/`` or the CI
``monitoring`` job) and ship the report to Prometheus via the pushgateway or
the JSON file this module writes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.utils.io import write_json
from src.utils.logger import get_logger
from src.validation.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = get_logger(__name__)

EPSILON = 1e-6


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """PSI between a reference and a current distribution."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return 0.0

    # Quantile edges from the reference keep the buckets balanced at training time.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)

    reference_share = np.clip(reference_share, EPSILON, None)
    current_share = np.clip(current_share, EPSILON, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def categorical_drift(reference: pd.Series, current: pd.Series) -> float:
    """PSI over category shares - same formula, buckets are the categories."""
    reference_share = reference.astype("string").fillna("__missing__").value_counts(normalize=True)
    current_share = current.astype("string").fillna("__missing__").value_counts(normalize=True)
    categories = reference_share.index.union(current_share.index)

    ref = np.clip(reference_share.reindex(categories, fill_value=0.0).to_numpy(), EPSILON, None)
    cur = np.clip(current_share.reindex(categories, fill_value=0.0).to_numpy(), EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def classify(psi: float, warn: float, alert: float) -> str:
    if psi >= alert:
        return "alert"
    if psi >= warn:
        return "warning"
    return "stable"


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    warn: float = 0.1,
    alert: float = 0.25,
) -> dict[str, Any]:
    """Per-feature PSI plus an overall verdict."""
    features: list[dict[str, Any]] = []

    for column in NUMERIC_FEATURES:
        if column in reference.columns and column in current.columns:
            psi = population_stability_index(reference[column], current[column])
            features.append({"feature": column, "type": "numeric", "psi": round(psi, 5),
                             "status": classify(psi, warn, alert)})

    for column in CATEGORICAL_FEATURES:
        if column in reference.columns and column in current.columns:
            psi = categorical_drift(reference[column], current[column])
            features.append({"feature": column, "type": "categorical", "psi": round(psi, 5),
                             "status": classify(psi, warn, alert)})

    drifted = [f for f in features if f["status"] != "stable"]
    worst = max(features, key=lambda f: f["psi"], default=None)

    report = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features": sorted(features, key=lambda f: f["psi"], reverse=True),
        "drifted_features": [f["feature"] for f in drifted],
        "max_psi": round(worst["psi"], 5) if worst else 0.0,
        "status": "alert" if any(f["status"] == "alert" for f in features)
                  else ("warning" if drifted else "stable"),
        "thresholds": {"warn": warn, "alert": alert},
    }
    logger.info("Drift status=%s max_psi=%.4f drifted=%s",
                report["status"], report["max_psi"], report["drifted_features"])
    return report


def run(current_path: str | Path, config: Config | None = None) -> dict[str, Any]:
    config = config or load_config()
    monitoring_cfg = config.section("monitoring")

    reference = pd.read_csv(config.get_path("monitoring.reference_path"))
    current = pd.read_csv(current_path)

    report = detect_drift(
        reference,
        current,
        warn=float(monitoring_cfg["psi_warn"]),
        alert=float(monitoring_cfg["psi_alert"]),
    )
    write_json(config.get_path("monitoring.drift_report_path"), report)
    return report


def main() -> int:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="Check production data for drift")
    parser.add_argument("--current", required=True, help="CSV of recently scored applications")
    parser.add_argument("--fail-on-alert", action="store_true",
                        help="Exit non-zero when any feature is in alert (for CI/CronJob)")
    args = parser.parse_args()

    report = run(args.current)
    print(f"status={report['status']} max_psi={report['max_psi']}")
    return 1 if args.fail_on_alert and report["status"] == "alert" else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
