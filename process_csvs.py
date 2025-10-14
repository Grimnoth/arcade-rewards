import pathlib
from typing import Optional

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
WEEKLY_INPUT_DIR = PROJECT_ROOT / "weekly-input"


def load_csv(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main(input_file: Optional[str] = None) -> None:
    if input_file is None:
        # Example: pick first CSV in weekly-input if provided
        candidates = sorted(WEEKLY_INPUT_DIR.glob("*.csv"))
        if not candidates:
            print("No input CSV found in weekly-input/. Specify a file path.")
            return
        csv_path = candidates[0]
    else:
        csv_path = pathlib.Path(input_file)

    df = load_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    # TODO: implement processing logic


if __name__ == "__main__":
    main()
