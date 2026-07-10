"""
Shared preprocessing. BOTH src/train.py and app.py import from here.
This is what keeps training and inference consistent - the single most
important file for making the upload feature actually work.

After running inspect_data.py, paste the real feature columns into
FEATURE_COLUMNS below (everything EXCEPT the Attack Type label column).
"""

import numpy as np
import pandas as pd

TARGET = "Attack Type"

# TODO: fill this from inspect_data.py output (df.columns minus the label).
# Leave as None to auto-use every column except TARGET while prototyping.
FEATURE_COLUMNS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size',
    'Subflow Fwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min',
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Housekeeping that must happen for BOTH the dataset and any upload."""
    df = df.copy()
    df.columns = df.columns.str.strip()                 # strip whitespace in names
    df = df.replace([np.inf, -np.inf], np.nan)          # infinities -> NaN
    df = df.dropna()                                    # drop rows with NaN
    df = df.drop_duplicates()
    return df


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the model's feature columns, in a fixed order."""
    cols = FEATURE_COLUMNS
    if cols is None:
        cols = [c for c in df.columns if c != TARGET]
    return df[cols]


def preprocess_for_training(df: pd.DataFrame):
    """Dataset path: returns (X features, y labels). No scaling here -
    scaling happens in train.py AFTER the split."""
    df = clean(df)
    y = df[TARGET]
    X = get_features(df)
    return X, y


def preprocess_upload(df: pd.DataFrame) -> tuple:
    """App path: an uploaded log -> (model-ready feature matrix, format_name).
    Auto-detects the log format and maps it to CICIDS features if needed.
    The app applies the SAVED scaler after this.

    Returns:
        X   : pd.DataFrame with exactly FEATURE_COLUMNS columns, ready to scale
        fmt : str  ('cicids' | 'zeek_conn' | 'honeypot' | 'generic')
    """
    from src.adapter import adapt
    df = df.replace([np.inf, -np.inf], np.nan)
    X, fmt = adapt(df)
    return X, fmt