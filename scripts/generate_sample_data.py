from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.downtime_risk.data import generate_sample_dataset


def main() -> None:
    output_path = Path("data") / "manufacturing_downtime_sample.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_sample_dataset(num_rows=1200, random_state=42)
    df.to_csv(output_path, index=False)

    print(f"Sample dataset saved to: {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
