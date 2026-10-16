"""
src/adapter.py
Detects the uploaded log format and maps it to the CICIDS2017 feature columns
the model was trained on. Missing features are filled with 0 (safe default).

Supported formats:
  - CICIDS2017  : already in the right format, pass-through
  - Zeek conn   : Zeek/Bro connection log (ts, uid, id.orig_h ...)
  - Zeek dns    : Zeek DNS log
  - Nlc/Honeypot: JSON-derived CSV (DateTime, RemoteAddr, Protocol ...)
  - Generic     : any CSV — maps whatever columns overlap, fills rest with 0
"""

import pandas as pd
import numpy as np

# ── The 52 features the model needs, in order ────────────────────────────────
CICIDS_FEATURES = [
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


# ── Format detection ──────────────────────────────────────────────────────────

def detect_format(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.strip().str.lower())

    if 'destination port' in cols and 'flow duration' in cols:
        return 'cicids'

    if 'uid' in cols and 'id.orig_h' in cols:
        return 'zeek_conn'

    if 'uid' in cols and 'query' in cols and 'qtype_name' in cols:
        return 'zeek_dns'

    if 'remoteaddr' in cols or 'datetime' in cols and 'protocol' in cols:
        return 'honeypot'

    return 'generic'


# ── Per-format mappers ────────────────────────────────────────────────────────

def _map_zeek_conn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zeek conn.log columns:
      ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p,
      proto, service, duration, orig_bytes, resp_bytes,
      conn_state, local_orig, missed_bytes, history,
      orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes, tunnel_parents
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    out = pd.DataFrame(0.0, index=df.index, columns=CICIDS_FEATURES)

    def safe(col):
        return pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    # Direct mappings
    out['Destination Port']            = safe('id.resp_p')
    out['Flow Duration']               = safe('duration') * 1_000_000   # s → µs
    out['Total Fwd Packets']           = safe('orig_pkts')
    out['Total Length of Fwd Packets'] = safe('orig_bytes')
    out['Bwd Packet Length Max']       = safe('resp_bytes')
    out['Subflow Fwd Bytes']           = safe('orig_bytes')

    # Derived
    duration_s = safe('duration').replace(0, np.nan)
    orig_bytes = safe('orig_bytes')
    resp_bytes = safe('resp_bytes')
    orig_pkts  = safe('orig_pkts').replace(0, np.nan)
    resp_pkts  = safe('resp_pkts').replace(0, np.nan)

    total_bytes = orig_bytes + resp_bytes
    total_pkts  = safe('orig_pkts') + safe('resp_pkts')

    out['Flow Bytes/s']        = (total_bytes / duration_s).fillna(0)
    out['Flow Packets/s']      = (total_pkts  / duration_s).fillna(0)
    out['Fwd Packets/s']       = (safe('orig_pkts') / duration_s).fillna(0)
    out['Bwd Packets/s']       = (safe('resp_pkts') / duration_s).fillna(0)
    out['Average Packet Size'] = (total_bytes / total_pkts.replace(0, np.nan)).fillna(0)

    out['Fwd Packet Length Mean'] = (orig_bytes / orig_pkts).fillna(0)
    out['Bwd Packet Length Mean'] = (resp_bytes / resp_pkts).fillna(0)
    out['Packet Length Mean']     = (total_bytes / total_pkts.replace(0, np.nan)).fillna(0)

    out['Min Packet Length'] = out[['Fwd Packet Length Mean',
                                    'Bwd Packet Length Mean']].min(axis=1)
    out['Max Packet Length'] = out[['Fwd Packet Length Mean',
                                    'Bwd Packet Length Mean']].max(axis=1)

    # Conn-state flag heuristics
    state = df.get('conn_state', pd.Series([''] * len(df))).fillna('')
    out['FIN Flag Count'] = state.str.contains('F', regex=False).astype(int)
    out['ACK Flag Count'] = state.str.contains('A', regex=False).astype(int)

    return out


def _map_honeypot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Honeypot / NLC JSON-derived CSV columns:
      DateTime, RemoteAddr, Protocol, Command, Status, Msg,
      User, Password, Client, ...
    These logs have no packet-level stats at all — we encode
    categorical signals that correlate with attack patterns.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    out = pd.DataFrame(0.0, index=df.index, columns=CICIDS_FEATURES)

    # Protocol → destination port heuristic
    proto_port = {'SSH': 22, 'HTTP': 80, 'HTTPS': 443, 'FTP': 21,
                  'TELNET': 23, 'SMTP': 25, 'DNS': 53}
    proto = df.get('Protocol', pd.Series([''] * len(df))).str.upper().fillna('')
    out['Destination Port'] = proto.map(proto_port).fillna(0)

    # Parse port from RemoteAddr  (e.g. "1.2.3.4:55123")
    if 'RemoteAddr' in df.columns:
        src_port = df['RemoteAddr'].str.extract(r':(\d+)$')[0]
        out['Fwd Header Length'] = pd.to_numeric(src_port, errors='coerce').fillna(0)

    # Each log row = 1 connection attempt → minimal packet counts
    out['Total Fwd Packets'] = 1
    out['Fwd Packets/s']     = 1
    out['ACK Flag Count']    = 1   # every completed attempt has an ACK

    # High-volume SSH attempts → Brute Force signal
    is_ssh = (proto == 'SSH').astype(float)
    out['FIN Flag Count']    = is_ssh   # SSH login attempt ends connection

    return out


def _map_generic(df: pd.DataFrame) -> pd.DataFrame:
    """
    For any unknown CSV: map whatever column names overlap (case-insensitive),
    fill everything else with 0.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Build a lowercase lookup of the upload's columns
    col_lower = {c.lower(): c for c in df.columns}
    feat_lower = {f.lower(): f for f in CICIDS_FEATURES}

    out = pd.DataFrame(0.0, index=df.index, columns=CICIDS_FEATURES)

    for fl, fn in feat_lower.items():
        if fl in col_lower:
            out[fn] = pd.to_numeric(df[col_lower[fl]], errors='coerce').fillna(0)

    return out


# ── Public entry point ────────────────────────────────────────────────────────

def adapt(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Detect format, map to CICIDS features, return (mapped_df, format_name).
    Raises ValueError if the file is completely unrecognisable (0 rows after clean).
    """
    df.columns = df.columns.str.strip()
    fmt = detect_format(df)

    if fmt == 'cicids':
        mapped = df                  # pass-through; preprocess.py handles it
    elif fmt == 'zeek_conn':
        mapped = _map_zeek_conn(df)
    elif fmt == 'honeypot':
        mapped = _map_honeypot(df)
    else:
        mapped = _map_generic(df)    # zeek_dns falls here too

    # Final safety net: ensure all 52 columns exist, fill missing with 0
    for col in CICIDS_FEATURES:
        if col not in mapped.columns:
            mapped[col] = 0.0

    mapped = mapped[CICIDS_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)

    if len(mapped) == 0:
        raise ValueError("No usable rows found after adapting the uploaded file.")

    return mapped, fmt
# ── Ground-truth label detection ─────────────────────────────────────────────
# Some uploads are already labelled (a Kaggle-style dataset with an
# Intrusion / Scan_Type / Label column). Those files have NO flow features for
# the model to read — the verdict is already in the file — so we read the
# labels directly instead of predicting on an all-zero feature matrix.

LABEL_TYPE_COLS = {          # categorical: WHAT kind of attack
    'scan_type', 'attack_type', 'attack', 'attack_cat', 'category',
    'label', 'class', 'traffic_type', 'threat', 'activity',
}
LABEL_FLAG_COLS = {          # binary: attack yes/no
    'intrusion', 'is_attack', 'is_malicious', 'malicious', 'anomaly', 'alert',
}
BENIGN_VALUES = {            # values meaning "not an attack", any column
    'normal', 'benign', 'none', 'clean', 'background', 'legitimate',
    '0', 'false', 'no', 'ok',
}
# Map dataset labels onto the model's class names so risk.py SEVERITY and the
# "Normal Traffic" bookkeeping in app.py keep working unchanged.
LABEL_NORMALISE = {
    'normal': 'Normal Traffic', 'benign': 'Normal Traffic',
    'portscan': 'Port Scanning', 'port scan': 'Port Scanning',
    'port scanning': 'Port Scanning', 'scan': 'Port Scanning',
    'botattack': 'Bots', 'bot': 'Bots', 'bots': 'Bots', 'botnet': 'Bots',
    'bruteforce': 'Brute Force', 'brute force': 'Brute Force',
    'web attack': 'Web Attacks', 'dos': 'DoS', 'ddos': 'DDoS',
}


def detect_labels(df: pd.DataFrame):
    """
    If the upload already has a ground-truth verdict column, return a per-row
    Series of model-class labels ('Port Scanning', 'Bots', 'Normal Traffic'...).
    Returns None when there's no such column, so the caller falls back to the model.
    """
    lower = {c.strip().lower(): c for c in df.columns}

    # Prefer a categorical type column — it tells us what the attack is.
    for key in LABEL_TYPE_COLS:
        if key in lower:
            vals = df[lower[key]].astype(str).str.strip()
            return vals.map(lambda v: 'Normal Traffic' if v.lower() in BENIGN_VALUES
                            else LABEL_NORMALISE.get(v.lower(), v))

    # Otherwise a binary flag — attack yes/no, type unknown.
    for key in LABEL_FLAG_COLS:
        if key in lower:
            vals = df[lower[key]].astype(str).str.strip().str.lower()
            return vals.map(lambda v: 'Normal Traffic' if v in BENIGN_VALUES else 'Attack')

    return None
# ── Behavioral scan detection for connection logs ────────────────────────────
# Zeek conn.log has none of the flow statistics the CICIDS model needs, so the
# model silently reports everything as Normal. Instead we read the scan signal
# directly: a source hitting many distinct ports/hosts with mostly unanswered
# or rejected connections is scanning.

# conn_states meaning "no successful data exchange" — the probe signature.
FAILED_STATES = {'S0', 'REJ', 'RSTO', 'RSTR', 'RSTOS0', 'RSTRH', 'SH', 'SHR'}


def detect_scans(df: pd.DataFrame,
                 min_ports: int = 50,
                 min_hosts: int = 25,
                 min_failed_ratio: float = 0.5) -> pd.Series:
    """
    Per-connection scan labels for a Zeek conn.log DataFrame.
    A source IP is flagged as a scanner when it touches >= min_ports distinct
    destination ports OR >= min_hosts distinct hosts, AND at least
    min_failed_ratio of its connections are unanswered/rejected (the guard that
    keeps busy legitimate servers from being flagged). Returns a Series of
    'Port Scanning' / 'Normal Traffic' aligned to df's rows.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    src, dst, dport, state = 'id.orig_h', 'id.resp_h', 'id.resp_p', 'conn_state'

    failed = df[state].astype(str).str.strip().isin(FAILED_STATES)
    agg = df.assign(_failed=failed).groupby(src).agg(
        dports=(dport, 'nunique'),
        dhosts=(dst,   'nunique'),
        fr=('_failed', 'mean'),
    )
    scanners = set(agg.index[
        ((agg.dports >= min_ports) | (agg.dhosts >= min_hosts)) &
        (agg.fr >= min_failed_ratio)
    ])

    preds = pd.Series('Normal Traffic', index=df.index)
    preds[df[src].isin(scanners) & failed] = 'Port Scanning'
    return preds.reset_index(drop=True)