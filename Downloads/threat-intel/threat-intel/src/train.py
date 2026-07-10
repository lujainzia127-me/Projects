"""
Train the Random Forest model on CICIDS2017/CICFlowMeter 52-feature data.

Run:
    python src/train.py
"""

import os
import sys

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import FEATURE_COLUMNS, TARGET, preprocess_for_training

DATA_PATH = "data/cicids2017_cleaned.csv"
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Change this lower if your laptop still runs out of RAM.
NORMAL_TRAFFIC_LIMIT = 300_000

# Smaller chunks use less memory but load a bit slower.
CHUNK_SIZE = 100_000


def downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage by converting 64-bit numeric columns to 32-bit."""
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")

    return df


def load_training_data_in_chunks() -> pd.DataFrame:
    """
    Load the large CICIDS dataset safely.

    Instead of loading the whole CSV into RAM at once, this reads the file in
    chunks. It keeps all attack rows, but limits the huge Normal Traffic class.
    """
    print("Loading training data in chunks...")

    needed_columns = set(FEATURE_COLUMNS + [TARGET])
    chunks = []
    normal_seen = 0

    for i, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=lambda c: str(c).strip() in needed_columns,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()
        chunk = downcast_numeric_columns(chunk)

        if TARGET not in chunk.columns:
            continue

        normal_rows = chunk[chunk[TARGET] == "Normal Traffic"]
        attack_rows = chunk[chunk[TARGET] != "Normal Traffic"]

        remaining_normal = NORMAL_TRAFFIC_LIMIT - normal_seen

        if remaining_normal > 0:
            normal_keep = normal_rows.head(remaining_normal)
            normal_seen += len(normal_keep)
            chunk_keep = pd.concat([normal_keep, attack_rows], ignore_index=True)
        else:
            chunk_keep = attack_rows

        if not chunk_keep.empty:
            chunks.append(chunk_keep)

        print(
            f"Chunk {i}: kept={len(chunk_keep):,}, "
            f"normal_kept={normal_seen:,}"
        )

    if not chunks:
        raise RuntimeError(
            "No training data was loaded. Check DATA_PATH, FEATURE_COLUMNS, and TARGET."
        )

    df = pd.concat(chunks, ignore_index=True)

    print("\nFinal loaded dataset:")
    print("Shape:", df.shape)
    print(df[TARGET].value_counts())

    return df


def build_smote_strategy(y_train: pd.Series) -> dict:
    """
    Oversample rare classes only when they exist and have enough samples.

    SMOTE with k_neighbors=5 needs at least 6 samples in the class.
    """
    counts = pd.Series(y_train).value_counts()
    strategy = {}

    for cls in ["Bots", "Web Attacks"]:
        if cls in counts and 6 <= counts[cls] < 20_000:
            strategy[cls] = 20_000

    return strategy


def main():
    df = load_training_data_in_chunks()

    X, y = preprocess_for_training(df)

    print("\nAfter preprocessing:")
    print("X shape:", X.shape)
    print(y.value_counts())

    # Stratified split keeps class distribution similar in train and test.
    stratify_target = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    # Fit scaler only on training data, then apply to both train and test.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nBefore SMOTE:")
    print(pd.Series(y_train).value_counts())

    smote_strategy = build_smote_strategy(y_train)

    if smote_strategy:
        print("\nApplying SMOTE:")
        print(smote_strategy)

        X_train, y_train = SMOTE(
            sampling_strategy=smote_strategy,
            k_neighbors=5,
            random_state=42,
        ).fit_resample(X_train, y_train)
    else:
        print("\nSMOTE skipped: no eligible rare classes found.")

    print("\nAfter SMOTE:")
    print(pd.Series(y_train).value_counts())

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, preds))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nSaved -> {MODEL_PATH} and {SCALER_PATH}")


if __name__ == "__main__":
    main()