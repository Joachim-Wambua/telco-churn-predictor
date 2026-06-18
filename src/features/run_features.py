import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import DATA_DIR
from src.features.engineering import create_features, select_features

CLEANED_PATH = DATA_DIR / "cleaned.csv"
OUTPUT_PATH  = DATA_DIR / "features.csv"


def main() -> None:
    t0 = time.time()

    print(f"Loading  {CLEANED_PATH}")
    df = pd.read_csv(CLEANED_PATH)
    print(f"  Input shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")

    print("Running  create_features ...")
    featured_df = create_features(df)
    n_engineered = featured_df.shape[1] - df.shape[1]
    print(f"  After engineering : {featured_df.shape[1]} cols (+{n_engineered} new)")

    print("Running  select_features ...")
    selected_cols, reduced_df = select_features(featured_df)
    n_numeric_before = featured_df.select_dtypes(include="number").shape[1]
    dropped = [c for c in featured_df.select_dtypes(include="number").columns
               if c not in selected_cols]
    print(f"  Numeric cols before : {n_numeric_before}")
    print(f"  Numeric cols after  : {len(selected_cols)}")
    if dropped:
        print(f"  Dropped ({len(dropped)})        : {', '.join(dropped)}")
    print(f"  Final shape         : {reduced_df.shape[0]:,} rows × {reduced_df.shape[1]} cols")

    print(f"\nKept features ({len(selected_cols)}):")
    for col in selected_cols:
        print(f"  {col}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reduced_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")

    print(f"Done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
