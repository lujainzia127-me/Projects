"""
Generic network/security log analyser.

Used when the uploaded file does NOT contain the 52 CICIDS/CICFlowMeter
features required by the trained Random Forest model.

Supports common raw log formats such as:
- Zeek conn.log style CSVs
- SSH/honeypot logs  (Cowrie, Dionaea, T-Pot, Kippo, etc.)
- HTTP/security logs
- Generic CSV/log files

FIX 4 (new): also supports two additional shapes seen in the wild:
- Pre-labeled / feature-engineered web-attack datasets (e.g. CSIC-2010 style
  files that already ship a `class`/`prediction` column from another model).
- Aggregated time-series attack-count exports (e.g. T-Pot/honeypot dashboards
  that export 30-min bucketed counts per honeypot type, per port, or per
  source country instead of one row per event).
"""

import ipaddress
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

WEB_ATTACK_RE = re.compile(
    r"(?:union\s+select|select\s+.+\s+from|or\s+1\s*=\s*1|drop\s+table|"
    r"<script|javascript:|onerror=|onload=|\.\./|\.\.\\|/etc/passwd|"
    r"wp-login|xmlrpc\.php|phpmyadmin|\.env|cmd=|exec=|powershell|/bin/sh|"
    r"wget\s|curl\s)",
    re.IGNORECASE,
)

BOT_RE = re.compile(
    r"(?:bot|crawler|spider|python-requests|go-http-client|masscan|zgrab|"
    r"nmap|nikto|sqlmap|curl|wget)",
    re.IGNORECASE,
)

LOGIN_RE = re.compile(
    r"(?:login|auth|authentication|password|invalid user|failed|failure|ssh)",
    re.IGNORECASE,
)

SCAN_RE = re.compile(
    r"(?:scan|nmap|masscan|zmap|portscan|port scan|recon)",
    re.IGNORECASE,
)

# FIX 3: shell / post-exploitation commands commonly logged by honeypots
COMMAND_RE = re.compile(
    r"(?:/bin/sh\b|/bin/bash\b|cmd\.exe|"
    r"wget\s+https?://|curl\s+https?://|"
    r"chmod\s+[+\-0-7]|\bnc\s+\-[elu]|\bnetcat\b|mkfifo|"
    r"cat\s+/etc/passwd|uname\s+-a|\bwhoami\b|\bid\s*;|"
    r"base64\s+-d|python\d*\s+-c|perl\s+-e|"
    r"echo\s+.+>>?\s*/|rm\s+-rf\s+/)",
    re.IGNORECASE,
)

# FIX 3: event patterns emitted by honeypot daemons
HONEYPOT_EVENT_RE = re.compile(
    r"(?:cowrie\.|dionaea\.|glastopf\.|kippo\.|"
    r"login\.failed|login\.success|command\.input|"
    r"session\.connect|session\.closed|"
    r"capture\.connect|download\.complete|"
    r"login\s+attempt|credential\s+attempt|"
    r"payload\s+capture|exploit\s+attempt|shellcode)",
    re.IGNORECASE,
)

FAILED_CONN_STATES = {"REJ", "S0", "SH", "SHR", "RSTOS0"}

# FIX 3: column names that are strong honeypot format indicators
_HONEYPOT_COL_KEYS: frozenset[str] = frozenset({
    "eventid", "sensor", "payload", "honeypot", "cowrieversion",
    "dionaea", "kippo", "attacktype", "honeytype", "capture",
    "peeraddress", "sessionid",
})

# FIX 4 (new): column names that mark a pre-labeled / feature-engineered
# web-attack dataset (e.g. CSIC-2010 derived exports) rather than a raw log.
_PRELABELED_MARKER_COLS: frozenset[str] = frozenset({
    "class", "prediction", "badwords_count", "path_length",
    "body_length", "features",
})

# FIX 4 (new): column names that mark a per-event log, used to make sure the
# time-series detector doesn't accidentally swallow real event-level logs.
_PER_EVENT_COL_KEYS: frozenset[str] = frozenset({
    "sourceip", "srcip", "id.orig_h", "method", "path", "eventid",
    "username", "password", "protocol", "url", "uri",
})


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def looks_like_cicids(df: pd.DataFrame, feature_columns: list[str]) -> bool:
    uploaded_cols = {str(c).strip() for c in df.columns}
    return all(col in uploaded_cols for col in feature_columns)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def find_col(df: pd.DataFrame, candidates: list[str]):
    lookup = {
        re.sub(r"[\s_\-\.]+", "", str(col).lower()): col
        for col in df.columns
    }
    for candidate in candidates:
        key = re.sub(r"[\s_\-\.]+", "", candidate.lower())
        if key in lookup:
            return lookup[key]
    return None


def safe_series(df: pd.DataFrame, col, default="") -> pd.Series:
    if col and col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


# FIX 3: detect whether the uploaded file is a honeypot sensor log
def is_honeypot_format(df: pd.DataFrame) -> bool:
    """
    Return True when the DataFrame looks like a honeypot sensor log.

    Checks two things:
    1. Column names that are unique to honeypot daemons (Cowrie, Dionaea…).
    2. A sample of the 'eventid' column for known daemon prefixes.
    """
    col_keys = {re.sub(r"[\s_\-\.]+", "", c.lower()) for c in df.columns}
    if col_keys & _HONEYPOT_COL_KEYS:
        return True

    eventid_col = find_col(df, ["eventid", "EventID", "event_id"])
    if eventid_col:
        sample = df[eventid_col].dropna().astype(str).head(30)
        if sample.str.contains(
            r"(?:cowrie|dionaea|kippo)\.", regex=True, case=False
        ).any():
            return True

    return False


# FIX 4 (new): detect pre-labeled / feature-engineered web-attack datasets
def is_prelabeled_feature_format(df: pd.DataFrame) -> bool:
    """
    Return True when the file already ships model-ready features plus a
    `class`/`prediction` column (e.g. CSIC-2010-style exports), rather than
    being a raw log we need to derive attack signals from ourselves.
    """
    cols = {re.sub(r"[\s_\-\.]+", "", c.lower()) for c in df.columns}
    if not ({"class", "prediction"} <= cols):
        return False
    hits = cols & _PRELABELED_MARKER_COLS
    return len(hits) >= 3


# FIX 4 (new): detect aggregated time-series attack-count exports
def is_timeseries_attack_format(df: pd.DataFrame) -> bool:
    """
    Return True when each row is a *bucket* (a timestamp, optionally split
    by a category such as honeypot type / port / country) holding an
    already-aggregated attack count, rather than one row per event.

    Covers exports such as:
    - Timestamp, Attack_counts_<HoneypotName>, Attack_counts_<HoneypotName>...
    - Timestamp, Attack_counts_<Port>, Attack_counts_<Port>...
    - Timestamp, Attack_counts, Unique_ips
    - Country, Timestamp, Attacks
    """
    ts_col = find_col(df, ["Timestamp", "ts", "Time", "DateTime", "@timestamp"])
    if not ts_col:
        return False

    cols_norm = {re.sub(r"[\s_\-\.]+", "", c.lower()) for c in df.columns}
    if cols_norm & _PER_EVENT_COL_KEYS:
        # Looks like a per-event log (has source IP / method / path /
        # eventid columns) — let the honeypot/generic analysers handle it.
        return False

    count_like = [
        c for c in df.columns
        if re.match(r"(?i)^attack_?counts?", c)
        or c.lower() in ("attacks", "unique_ips", "count", "counts")
    ]
    return len(count_like) >= 1


def find_category_col(df: pd.DataFrame, exclude: set):
    col = find_col(df, ["Country", "Honeypot", "Port", "Sensor", "Type", "Category"])
    if col and col not in exclude:
        return col
    return None


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def valid_ip_or_blank(value: str) -> str:
    try:
        ipaddress.ip_address(value)
        return value
    except Exception:
        return ""


def extract_ip_from_series(series: pd.Series) -> pd.Series:
    extracted = (
        series.fillna("")
        .astype(str)
        .str.extract(r"((?:\d{1,3}\.){3}\d{1,3})", expand=False)
        .fillna("")
    )
    return extracted.map(valid_ip_or_blank)


def extract_source_ip(df: pd.DataFrame) -> pd.Series:
    col = find_col(
        df,
        [
            "id.orig_h", "orig_h", "SourceIp", "SourceIP",
            "src_ip", "srcip", "source_ip", "RemoteAddr",
            "remote_addr", "client_ip", "ip",
            # honeypot aliases
            "peerAddress", "attacker_ip", "attacker",
        ],
    )
    return extract_ip_from_series(safe_series(df, col))


def extract_destination_ip(df: pd.DataFrame) -> pd.Series:
    col = find_col(
        df,
        [
            "id.resp_h", "resp_h", "DestinationIp", "DestinationIP",
            "dst_ip", "dstip", "destination_ip", "server_ip",
        ],
    )
    return extract_ip_from_series(safe_series(df, col))


def extract_source_port(df: pd.DataFrame) -> pd.Series:
    col = find_col(
        df,
        ["id.orig_p", "orig_p", "SourcePort", "src_port", "sport", "source_port"],
    )
    if col:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index)


def extract_destination_port(df: pd.DataFrame) -> pd.Series:
    col = find_col(
        df,
        [
            "id.resp_p", "resp_p", "Destination Port", "DestinationPort",
            "dst_port", "dport", "destination_port", "port",
            # honeypot aliases
            "DestPort", "dest_port", "listenPort",
        ],
    )
    if col:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index)


def extract_datetime(df: pd.DataFrame) -> pd.Series:
    col = find_col(
        df,
        [
            "ts", "DateTime", "Timestamp", "Time", "Date",
            "StartTime", "EventTime", "@timestamp",
        ],
    )
    if not col:
        return pd.Series([pd.NaT] * len(df), index=df.index)

    raw = df[col]
    numeric = pd.to_numeric(raw, errors="coerce")

    if numeric.notna().sum() > len(df) * 0.7:
        return pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)

    return pd.to_datetime(raw, errors="coerce", utc=True)


def combined_text(df: pd.DataFrame) -> pd.Series:
    useful_cols = [
        "Protocol", "proto", "service", "conn_state", "history",
        "Command", "CommandOutput", "Status", "Msg", "Message",
        "Event", "User", "Username", "Password", "Client",
        "Headers", "Cookies", "UserAgent", "HostHTTPRequest",
        "Body", "HTTPMethod", "RequestURI", "URI", "URL",
        "Description", "Handler", "Action",
        # FIX 3: honeypot-specific field names
        "eventid", "EventID", "event_id",
        "Payload", "payload",
        "Sensor", "sensor",
        "AttackType", "RecordType",
        "Input", "input",
        "Country", "country",
        "peerAddress",
    ]

    cols = [col for col in useful_cols if col in df.columns]
    if not cols:
        cols = list(df.columns)

    text = pd.Series([""] * len(df), index=df.index, dtype="object")
    for col in cols:
        text = text + " " + df[col].fillna("").astype(str)

    return text.str.strip()


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def zeek_port_scan_mask(
    df: pd.DataFrame,
    source_ip: pd.Series,
    destination_ip: pd.Series,
    destination_port: pd.Series,
) -> pd.Series:
    """Detect scanning in Zeek-style connection logs."""

    if source_ip.eq("").all():
        return pd.Series([False] * len(df), index=df.index)

    conn_state_col = find_col(df, ["conn_state", "connection_state"])
    conn_state = safe_series(df, conn_state_col).str.upper()

    temp = pd.DataFrame(
        {
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "destination_port": destination_port,
            "failed": conn_state.isin(FAILED_CONN_STATES),
        },
        index=df.index,
    )
    temp = temp[temp["source_ip"] != ""]

    if temp.empty:
        return pd.Series([False] * len(df), index=df.index)

    source_stats = temp.groupby("source_ip").agg(
        events=("source_ip", "size"),
        unique_dest_ips=("destination_ip", "nunique"),
        unique_dest_ports=("destination_port", "nunique"),
        failed_connections=("failed", "sum"),
    )
    source_stats["failed_ratio"] = (
        source_stats["failed_connections"] / source_stats["events"]
    )

    scanner_sources = source_stats[
        (source_stats["events"] >= 100)
        & (source_stats["failed_ratio"] >= 0.50)
        & (
            (source_stats["unique_dest_ips"] >= 50)
            | (source_stats["unique_dest_ports"] >= 50)
        )
    ].index

    return source_ip.isin(scanner_sources)


def rate_attack_masks(
    df: pd.DataFrame,
    source_ip: pd.Series,
    event_time: pd.Series,
):
    """Basic DoS/DDoS-style detection from event rates."""

    false_mask = pd.Series([False] * len(df), index=df.index)

    if event_time.notna().sum() == 0 or source_ip.eq("").all():
        return false_mask, false_mask

    minute = event_time.dt.floor("min")

    temp = pd.DataFrame(
        {"minute": minute, "source_ip": source_ip},
        index=df.index,
    ).dropna()

    if temp.empty:
        return false_mask, false_mask

    per_ip_minute   = temp.groupby(["source_ip", "minute"])["source_ip"].transform("size")
    per_minute      = temp.groupby("minute")["source_ip"].transform("size")
    sources_per_min = temp.groupby("minute")["source_ip"].transform("nunique")

    dos_index  = temp.index[per_ip_minute >= 60]
    ddos_index = temp.index[(per_minute >= 200) & (sources_per_min >= 10)]

    dos_mask  = false_mask.copy()
    ddos_mask = false_mask.copy()
    dos_mask.loc[dos_index]   = True
    ddos_mask.loc[ddos_index] = True

    return dos_mask, ddos_mask


# ---------------------------------------------------------------------------
# FIX 4 (new): pre-labeled / feature-engineered web-attack dataset analyser
# ---------------------------------------------------------------------------

def analyze_prelabeled_features(df: pd.DataFrame):
    """
    Analyse files that already contain model-ready features plus a
    `class`/`prediction` column (e.g. CSIC-2010-derived exports), instead of
    re-deriving attack signals with regexes that were built for raw text.

    Also guards against a common data-quality issue seen in these exports:
    rows whose `method`/`path` fields are placeholder "0" values (i.e. no
    real request was captured). Those rows are reported separately as
    "Invalid/Empty Entry" and excluded from the risk/attack-rate metrics
    instead of silently being counted as "Normal Traffic".

    Returns:
        result_df – original DataFrame with a Predicted Attack Type /
                    Row Status column appended (all rows, for the preview
                    and CSV download)
        preds     – Series of labels for VALID rows only (used for the
                    headline metrics, so empty placeholder rows don't
                    dilute the risk score)
        meta      – dict with mode, skipped-row count, etc.
    """

    df = normalise_columns(df)
    result = df.copy()

    method_col = find_col(df, ["method", "HTTPMethod"])
    path_col   = find_col(df, ["path", "URI", "URL", "RequestURI"])
    pred_col   = find_col(df, ["prediction"])
    class_col  = find_col(df, ["class"])
    score_col  = pred_col or class_col

    method = safe_series(df, method_col)
    path   = safe_series(df, path_col)

    placeholder_values = {"0", "", "nan", "none"}
    valid_mask = ~(
        method.str.lower().isin(placeholder_values)
        & path.str.lower().isin(placeholder_values)
    )

    pred_numeric = pd.to_numeric(df[score_col], errors="coerce") if score_col else pd.Series(pd.NA, index=df.index)

    labels = pd.Series(["Invalid/Empty Entry"] * len(df), index=df.index, dtype="object")
    reasons = pd.Series(
        ["Row has no request data (method/path placeholder) — excluded from risk metrics"] * len(df),
        index=df.index,
        dtype="object",
    )

    attack_mask = valid_mask & (pred_numeric == 1)
    normal_mask = valid_mask & (pred_numeric == 0)

    labels.loc[attack_mask] = "Web Attacks"
    reasons.loc[attack_mask] = f"Existing '{score_col}' label from prior model marks this request as anomalous"

    labels.loc[normal_mask] = "Normal Traffic"
    reasons.loc[normal_mask] = f"Existing '{score_col}' label from prior model marks this request as normal"

    result["Predicted Attack Type"] = labels
    result["Detection Reason"] = reasons

    skipped = int((~valid_mask).sum())
    preds = labels[valid_mask].reset_index(drop=True)

    # Top attacked paths, for context in the dashboard (reuses the
    # "top_ips" meta slot the app already renders as a table).
    if path_col:
        top_paths = (
            result.loc[attack_mask, path_col]
            .value_counts()
            .head(10)
            .to_dict()
        )
    else:
        top_paths = {}

    meta = {
        "mode": "Pre-labeled web-attack feature dataset (existing model predictions used)",
        "top_ips": top_paths,
        "top_ips_label": "Top attacked paths",
        "top_ips_axis_label": "Path",
        "top_ips_value_label": "Attack count",
        "trend": pd.Series(dtype=int),
        "is_honeypot": False,
        "skipped_invalid_rows": skipped,
        "used_column": score_col,
    }

    return result, preds, meta


# ---------------------------------------------------------------------------
# FIX 4 (new): aggregated time-series attack-count analyser
# ---------------------------------------------------------------------------

def analyze_timeseries_logs(df: pd.DataFrame):
    """
    Analyse pre-aggregated attack-count exports where each row is a time
    bucket (optionally split by category, e.g. honeypot type / port /
    source country) rather than one row per individual event.

    Detects volume spikes (bucket total > mean + 2*std of the whole series)
    and labels those buckets "Attack Surge"; everything else stays
    "Normal Traffic". Category totals (busiest honeypot / port / country)
    are surfaced the same way the app already surfaces "top suspicious
    source IPs" for event-level logs.

    Returns:
        result_df – original DataFrame with extracted time + prediction
                    columns appended
        preds     – Series of labels, one per row
        meta      – dict with mode, top categories, trend series, etc.
    """

    df = normalise_columns(df)
    result = df.copy()

    ts_col = find_col(df, ["Timestamp", "ts", "Time", "DateTime", "@timestamp"])
    event_time = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    cat_col = find_category_col(df, exclude={ts_col})
    exclude = {ts_col, cat_col} if cat_col else {ts_col}
    numeric_cols = [c for c in df.columns if c not in exclude]
    wide_cols = [c for c in numeric_cols if re.match(r"(?i)^attack_?counts?_", c)]

    if cat_col:
        # Long format: one row per (category, timestamp), e.g. Country file.
        count_col = find_col(df, ["Attacks", "Attack_counts", "Count", "Counts"])
        long_df = pd.DataFrame({
            "time": event_time,
            "category": df[cat_col].astype(str),
            "count": pd.to_numeric(df[count_col], errors="coerce").fillna(0) if count_col else 0,
        })
    elif wide_cols:
        # Wide multi-category format, e.g. per-honeypot or per-port files.
        melt = df[[ts_col] + wide_cols].melt(
            id_vars=[ts_col], var_name="category", value_name="count"
        )
        melt["category"] = melt["category"].str.replace(
            r"(?i)^attack_?counts?_", "", regex=True
        )
        melt["count"] = pd.to_numeric(melt["count"], errors="coerce").fillna(0)
        melt["time"] = pd.to_datetime(melt[ts_col], errors="coerce", utc=True)
        long_df = melt[["time", "category", "count"]]
    else:
        # Simple summary format, e.g. Timestamp, Attack_counts, Unique_ips.
        count_col = find_col(df, ["Attack_counts", "Attacks", "Count", "Counts"])
        long_df = pd.DataFrame({
            "time": event_time,
            "category": "Total",
            "count": pd.to_numeric(df[count_col], errors="coerce").fillna(0) if count_col else 0,
        })

    long_df = long_df.dropna(subset=["time"])

    total_by_time = long_df.groupby("time")["count"].sum().sort_index()
    category_totals = (
        long_df.groupby("category")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    mean = total_by_time.mean() if len(total_by_time) else 0
    std = total_by_time.std(ddof=0) if len(total_by_time) else 0
    threshold = mean + 2 * std if std > 0 else mean * 1.5
    spike_times = set(total_by_time[total_by_time > threshold].index)

    labels = event_time.map(
        lambda t: "Attack Surge" if t in spike_times else "Normal Traffic"
    )
    labels = pd.Series(labels, index=df.index, dtype="object")
    reasons = labels.map({
        "Attack Surge": "Attack volume in this time bucket is a statistical outlier "
                         "(more than 2 standard deviations above the average bucket)",
        "Normal Traffic": "Attack volume in this time bucket is within the normal range",
    })

    result["Extracted Event Time"] = event_time
    result["Bucket Total Attacks"] = event_time.map(total_by_time.to_dict())
    result["Predicted Attack Type"] = labels
    result["Detection Reason"] = reasons

    # Hourly trend for the line chart (works even if buckets are 30 min).
    trend = total_by_time.resample("h").sum() if len(total_by_time) else pd.Series(dtype=int)

    meta = {
        "mode": "Aggregated time-series attack analysis",
        "top_ips": category_totals,
        "top_ips_label": f"Top {cat_col or 'attack'} categories by volume" if cat_col else "Top attack categories by volume",
        "top_ips_axis_label": cat_col or "Category",
        "top_ips_value_label": "Total attacks",
        "trend": trend,
        "is_honeypot": False,
        "spike_buckets": len(spike_times),
        "total_buckets": int(total_by_time.shape[0]),
    }

    return result, labels, meta


# ---------------------------------------------------------------------------
# Main analyser (raw / per-event logs — Zeek, honeypot events, HTTP, generic)
# ---------------------------------------------------------------------------

def analyze_generic_logs(df: pd.DataFrame):
    """
    Analyse arbitrary network/security logs.

    Returns:
        result_df  – original DataFrame with extracted + prediction columns appended
        preds      – Series of predicted attack-type labels (same index as result_df)
        meta       – dict with mode, top_ips, trend, is_honeypot
    """

    df = normalise_columns(df)
    result = df.copy()

    source_ip        = extract_source_ip(df)
    destination_ip   = extract_destination_ip(df)
    source_port      = extract_source_port(df)
    destination_port = extract_destination_port(df)
    event_time       = extract_datetime(df)
    text             = combined_text(df)

    protocol_col = find_col(df, ["Protocol", "proto", "service"])
    protocol     = safe_series(df, protocol_col).str.upper()

    # FIX 3: detect honeypot format once; used for catch-all at the end
    honeypot_format = is_honeypot_format(df)

    labels  = pd.Series(["Normal Traffic"] * len(df), index=df.index, dtype="object")
    reasons = pd.Series(
        ["No strong suspicious pattern matched"] * len(df),
        index=df.index,
        dtype="object",
    )

    # ------------------------------------------------------------------
    # Build detection masks
    # ------------------------------------------------------------------

    # FIX 1: exclude empty-string IPs so rows with no extractable IP
    # are not counted together and do not trigger false brute-force hits.
    valid_source = source_ip[source_ip != ""]
    ip_counts      = valid_source.value_counts()
    repeated_source = source_ip.map(ip_counts).fillna(0).astype(int) >= 3

    web_mask       = text.str.contains(WEB_ATTACK_RE, na=False)
    bot_mask       = text.str.contains(BOT_RE, na=False)
    login_mask     = text.str.contains(LOGIN_RE, na=False)
    command_mask   = text.str.contains(COMMAND_RE, na=False)   # FIX 3
    honeypot_mask  = text.str.contains(HONEYPOT_EVENT_RE, na=False)  # FIX 3

    ssh_mask = (
        protocol.str.contains("SSH", na=False)
        | text.str.contains(r"\bssh\b", case=False, na=False, regex=True)
    )
    scan_text_mask = text.str.contains(SCAN_RE, na=False)

    zeek_scan_mask = zeek_port_scan_mask(
        df,
        source_ip=source_ip,
        destination_ip=destination_ip,
        destination_port=destination_port,
    )

    brute_force_mask = (login_mask | ssh_mask) & repeated_source
    dos_mask, ddos_mask = rate_attack_masks(df, source_ip, event_time)

    # ------------------------------------------------------------------
    # Apply labels: highest priority last so it wins the final value.
    # Each tier only touches rows still marked "Normal Traffic", EXCEPT
    # DDoS (which also promotes existing DoS rows) and Web Attacks /
    # Command Execution (which are unconditional — they are the
    # highest-confidence signals and should never be downgraded).
    # ------------------------------------------------------------------

    # Tier 6 – lowest: automated tools / bots
    bot_apply = bot_mask & labels.eq("Normal Traffic")
    labels.loc[bot_apply]  = "Bots"
    reasons.loc[bot_apply] = "Automated tool or bot-like client detected"

    # Tier 5: DoS (single-source rate)
    dos_apply = dos_mask & labels.eq("Normal Traffic")
    labels.loc[dos_apply]  = "DoS"
    reasons.loc[dos_apply] = "High event rate from one source"

    # Tier 4: DDoS (multi-source rate) — also promotes existing DoS rows
    ddos_apply = ddos_mask & labels.isin(["Normal Traffic", "DoS"])
    labels.loc[ddos_apply]  = "DDoS"
    reasons.loc[ddos_apply] = "High event rate from many sources"

    # Tier 3: port scan
    scan_apply = (scan_text_mask | zeek_scan_mask) & labels.eq("Normal Traffic")
    labels.loc[scan_apply]  = "Port Scanning"
    reasons.loc[scan_apply] = (
        "Port-scan pattern detected: source contacted many ports/hosts "
        "with many failed connections"
    )

    # Tier 2: brute-force / credential stuffing
    # FIX 2: guard with labels.eq("Normal Traffic") so that rows already
    # flagged as a higher-confidence type (e.g. Web Attacks from a later
    # tier) are not silently downgraded to Brute Force.
    brute_apply = brute_force_mask & labels.eq("Normal Traffic")
    labels.loc[brute_apply]  = "Brute Force"
    reasons.loc[brute_apply] = (
        "Repeated SSH/login/authentication attempts from the same source"
    )

    # FIX 3 Tier 1b: post-exploitation command execution
    cmd_apply = command_mask & labels.eq("Normal Traffic")
    labels.loc[cmd_apply]  = "Command Execution"
    reasons.loc[cmd_apply] = "Shell command or post-exploitation activity detected"

    # Tier 1a: web / injection attacks — highest priority, always wins
    labels.loc[web_mask]  = "Web Attacks"
    reasons.loc[web_mask] = "Web exploit pattern detected"

    # FIX 3 Honeypot catch-all: if the file is a honeypot log, any
    # row still marked "Normal Traffic" (i.e. not matched by any pattern
    # above) was still recorded because something hit the sensor — flag it.
    if honeypot_format:
        hp_apply = labels.eq("Normal Traffic")
        labels.loc[hp_apply]  = "Honeypot Event"
        reasons.loc[hp_apply] = (
            "Unclassified connection recorded by honeypot sensor"
        )

    # ------------------------------------------------------------------
    # Assemble result columns
    # ------------------------------------------------------------------

    result["Extracted Source IP"]        = source_ip
    result["Extracted Destination IP"]   = destination_ip
    result["Extracted Source Port"]      = source_port
    result["Extracted Destination Port"] = destination_port
    result["Extracted Event Time"]       = event_time
    result["Predicted Attack Type"]      = labels
    result["Detection Reason"]           = reasons

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    top_ips = (
        result.loc[
            labels.ne("Normal Traffic") & labels.ne("Honeypot Event") & source_ip.ne(""),
            "Extracted Source IP",
        ]
        .value_counts()
        .head(10)
        .to_dict()
    )

    if event_time.notna().any():
        trend_df = pd.DataFrame({"time": event_time, "label": labels}).dropna()
        trend_df = trend_df[
            ~trend_df["label"].isin(["Normal Traffic", "Honeypot Event"])
        ]
        trend = (
            trend_df.groupby(pd.Grouper(key="time", freq="h")).size()
            if not trend_df.empty
            else pd.Series(dtype=int)
        )
    else:
        trend = pd.Series(dtype=int)

    meta = {
        "mode": "Generic rule-based log analysis",
        "top_ips": top_ips,
        "top_ips_label": "Top suspicious source IPs",
        "trend": trend,
        "is_honeypot": honeypot_format,   # FIX 3: expose to app.py
    }

    return result, labels, meta


# British spelling alias kept for backwards compatibility
analyse_generic_logs = analyze_generic_logs