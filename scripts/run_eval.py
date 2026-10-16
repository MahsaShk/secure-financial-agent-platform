from __future__ import annotations

import json
from pathlib import Path

from secure_financial_agent.evaluate import (
    evaluate_baselines,
    evaluate_security,
    metrics_to_dict,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    data_dir = repository_root / "data"
    baseline_metrics = evaluate_baselines(data_dir)
    security_metrics = evaluate_security(data_dir)
    payload = {
        "dataset": "synthetic-smoke-test-v1",
        "baselines": [metrics_to_dict(item) for item in baseline_metrics],
        "security": security_metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

