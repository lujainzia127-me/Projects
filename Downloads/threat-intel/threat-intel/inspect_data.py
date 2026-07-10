"""
Step 1: Load and inspect the dataset.
Run this FIRST, before writing any model code.

    python inspect_data.py

Copy two things from the output:
  1. The column list  -> goes into src/preprocess.py (FEATURE_COLUMNS)
  2. The Attack Type counts -> tells you the class imbalance to fix later
"""

import pandas as pd

DATA_PATH = "data/cicids2017_cleaned.csv"
TARGET = "Attack Type"   # change if your file names the label column differently

def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE (rows, columns):", df.shape)

    print("=" * 60)
    print("COLUMNS:")
    print(df.columns.tolist())

    print("=" * 60)
    print(f"{TARGET} distribution:")
    print(df[TARGET].value_counts())

    print("=" * 60)
    print("Missing values (only columns with any):")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.any() else "None - dataset is clean.")

    print("=" * 60)
    print("Data types:")
    print(df.dtypes.value_counts())

    print("=" * 60)
    print("First 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()
