"""Rewrite absolute MLflow artifact paths in the tracking database to relative ones.

MLflow expands an experiment's artifact location to an absolute path when the
experiment is created, so a SQLite tracking database records the machine it was
trained on. Copy the project elsewhere - or hand it to someone else - and the
runs still list their parameters and metrics, but every artifact link points at
a directory that does not exist.

This script strips the leading path up to and including the project directory,
leaving paths like ``mlartifacts/<run-id>/artifacts`` that MLflow resolves
against the working directory. Run it from the project root before sharing:

    python scripts/make_mlflow_portable.py
    python scripts/make_mlflow_portable.py --database mlflow.db --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

# (table, column) pairs that store an artifact location.
PATH_COLUMNS = (("experiments", "artifact_location"), ("runs", "artifact_uri"))


def to_relative(value: str, project_name: str) -> str:
    """Strip everything up to and including ``/<project_name>/`` from ``value``."""
    marker = f"/{project_name}/"
    index = value.find(marker)
    if index == -1:
        return value
    return value[index + len(marker):]


def rewrite(database: Path, project_name: str, dry_run: bool = False) -> dict[str, int]:
    """Make every stored artifact path relative. Returns rows changed per column."""
    if not database.exists():
        raise FileNotFoundError(f"No MLflow database at {database}")

    changed: dict[str, int] = {}
    connection = sqlite3.connect(database)
    try:
        cursor = connection.cursor()
        for table, column in PATH_COLUMNS:
            rows = cursor.execute(
                f"select rowid, {column} from {table} where {column} is not null"  # noqa: S608
            ).fetchall()
            updates = [
                (to_relative(value, project_name), rowid)
                for rowid, value in rows
                if to_relative(value, project_name) != value
            ]
            changed[f"{table}.{column}"] = len(updates)
            if updates and not dry_run:
                cursor.executemany(
                    f"update {table} set {column} = ? where rowid = ?", updates  # noqa: S608
                )
        if not dry_run:
            connection.commit()
    finally:
        connection.close()
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="mlflow.db", help="Path to the tracking database")
    parser.add_argument("--project-name", default=Path.cwd().name,
                        help="Directory name to cut the paths at (default: this directory)")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    changed = rewrite(Path(args.database), args.project_name, dry_run=args.dry_run)
    prefix = "would update" if args.dry_run else "updated"
    for location, count in changed.items():
        print(f"{prefix} {count} rows in {location}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
