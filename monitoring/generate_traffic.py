"""Send sample traffic to the API so Prometheus and Grafana have data to show.

    python monitoring/generate_traffic.py --requests 300 --url http://localhost:8000

Applications are drawn from the same generator the training data came from, so
the live score distribution should resemble the training one - which is exactly
what the drift panels in the Grafana dashboard compare against.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.ingestion.generate_data import generate_loan_dataset  # noqa: E402

FIELDS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area",
]


def _payloads(n: int, seed: int) -> list[dict]:
    frame = generate_loan_dataset(n_samples=n, missing_rate=0.02, random_state=seed)[FIELDS]
    records = json.loads(frame.to_json(orient="records"))
    return [{k: v for k, v in record.items() if v is not None} for record in records]


def _post(url: str, payload: dict, timeout: float = 5.0) -> tuple[int, dict | None]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except urllib.error.URLError:
        return 0, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate demo traffic for the dashboards")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds between requests")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    payloads = _payloads(args.requests, args.seed)
    approved = rejected = failed = 0

    for i, payload in enumerate(payloads, start=1):
        status, body = _post(f"{args.url}/predict", payload)
        if status == 200 and body:
            approved += body["prediction"] == 1
            rejected += body["prediction"] == 0
        else:
            failed += 1
        if i % 50 == 0:
            print(f"{i}/{len(payloads)} sent | approved={approved} rejected={rejected} failed={failed}")
        time.sleep(max(0.0, args.delay + random.uniform(-0.01, 0.01)))

    print(f"\nDone: {len(payloads)} requests | approved={approved} rejected={rejected} failed={failed}")
    if failed:
        print("Some requests failed - is the API running? (make compose-up)")
    return 1 if failed == len(payloads) else 0


if __name__ == "__main__":
    raise SystemExit(main())
