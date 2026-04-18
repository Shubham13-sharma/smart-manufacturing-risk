from pathlib import Path
import argparse
import sys

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.downtime_risk.model import load_or_create_dataset, train_and_select_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the downtime risk classifier.")
    parser.add_argument(
        "--dataset",
        default="data/manufacturing_downtime_sample.csv",
        help="Path to the source CSV dataset.",
    )
    args = parser.parse_args()

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    df = load_or_create_dataset(dataset_path)
    results = train_and_select_model(df)

    joblib.dump(results["model"], artifacts_dir / "best_model.joblib")
    joblib.dump(results["feature_columns"], artifacts_dir / "feature_columns.joblib")
    (artifacts_dir / "metrics.json").write_text(results["metrics_json"], encoding="utf-8")

    print(f"Best model: {results['model_name']}")
    print("Metrics:")
    for metric_name, metric_value in results["metrics"].items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
