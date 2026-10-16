"""
AI Threat Intelligence Platform - Streamlit app.

Analysis modes:
1. CICIDS/CICFlowMeter 52-feature files        -> Random Forest ML model
2. Pre-labeled web-attack feature datasets      -> existing class/prediction
   (e.g. CSIC-2010 style exports)                  columns are trusted directly
3. Aggregated time-series attack-count exports  -> volume/spike analysis
   (per-honeypot, per-port, per-country buckets)
4. Raw network/security logs                    -> generic rule-based log analysis
   (covers Zeek conn, DNS, honeypot logs, HTTP/security logs, generic CSV)
"""

import os
import re

import joblib
import pandas as pd
import streamlit as st

from src.preprocess import FEATURE_COLUMNS, preprocess_upload
from src.risk import risk_score, risk_band
from src.alerts import send_alert
from src.report import build_report

try:
    from src.generic_logs import (
        looks_like_cicids,
        is_honeypot_format,
        is_prelabeled_feature_format,
        is_timeseries_attack_format,
        analyze_generic_logs,
        analyze_prelabeled_features,
        analyze_timeseries_logs,
    )
except ImportError:
    from src.generic_logs import (
        looks_like_cicids,
        is_honeypot_format,
        is_prelabeled_feature_format,
        is_timeseries_attack_format,
        analyse_generic_logs as analyze_generic_logs,
        analyze_prelabeled_features,
        analyze_timeseries_logs,
    )


MODEL_PATH  = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

st.set_page_config(page_title="AI Threat Intelligence Platform", layout="wide")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      .app-title {
        text-align:center; font-size:52px; font-weight:800; color:#e8fff0;
        text-shadow:0 0 18px rgba(34,197,94,.55); margin:6px 0 2px 0;
      }
      .app-sub { text-align:center; color:#9ca3af; margin-bottom:18px; }
      @keyframes pulse {
        0%   { box-shadow:0 0 0 0 rgba(239,68,68,.55); }
        70%  { box-shadow:0 0 0 16px rgba(239,68,68,0); }
        100% { box-shadow:0 0 0 0 rgba(239,68,68,0); }
      }
      .risk-banner { border-radius:16px; padding:24px 30px; margin:8px 0 18px 0; }
      .risk-label  { font-size:13px; letter-spacing:3px; opacity:.85; }
      .risk-band   { font-size:44px; font-weight:800; line-height:1.05; }
      .risk-msg    { font-size:18px; margin-top:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-title">AI Threat Intelligence Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-sub">Upload a network log to detect threats and assess risk.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """
    Load model/scaler only when they exist.

    Generic-log mode, honeypot mode, pre-labeled mode, and time-series mode
    do not require the model, so the app must not stop just because
    model.pkl / scaler.pkl are absent.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def render_risk_banner(score: int, band: str):
    palette = {
        "Critical": ("#3f0a0a", "#ef4444", "#fecaca", "Immediate action required.", True),
        "High":     ("#5b1414", "#f87171", "#fecaca", "Investigate immediately.", True),
        "Medium":   ("#5c3410", "#f59e0b", "#fde68a", "Some attacks present - review needed.", False),
        "Low":      ("#0f3d24", "#22c55e", "#bbf7d0", "Network looks healthy.", False),
    }
    bg, accent, text, msg, pulse = palette.get(band, palette["Low"])
    anim = "animation:pulse 1.4s infinite;" if pulse else ""
    st.markdown(
        f"""
        <div class="risk-banner" style="background:{bg}; border:2px solid {accent}; {anim}">
          <div class="risk-label" style="color:{text};">RISK STATUS</div>
          <div class="risk-band" style="color:{accent};">{band.upper()} RISK</div>
          <div class="risk-msg" style="color:{text};">
            Risk score: <b>{score} / 100</b> &mdash; {msg}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, low_memory=False)
    # .log / .txt — try delimited parsing, fall back to raw lines
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        text = uploaded_file.read().decode("utf-8", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        return pd.DataFrame({"RawLog": lines})


def recommendations_for(counts: dict, mode: str) -> list[str]:
    recs = []
    if counts.get("Command Execution", 0):
        recs.append(
            "Isolate affected hosts — post-exploitation commands were detected. "
            "Review shell history and running processes immediately."
        )
    if counts.get("Brute Force", 0):
        recs.append(
            "Block or rate-limit repeated login sources and enforce MFA on all exposed services."
        )
    if counts.get("Port Scanning", 0):
        recs.append(
            "Review firewall rules and block reconnaissance sources scanning multiple ports/hosts."
        )
    if counts.get("DDoS", 0) or counts.get("DoS", 0):
        recs.append(
            "Apply rate limiting, upstream filtering, or a DDoS-mitigation service for high-rate sources."
        )
    if counts.get("Web Attacks", 0):
        recs.append(
            "Inspect web requests for injection / path-traversal payloads and patch exposed applications."
        )
    if counts.get("Bots", 0):
        recs.append(
            "Review automated clients and block known-malicious user agents and scanning tools."
        )
    if counts.get("Honeypot Event", 0):
        recs.append(
            "Honeypot interactions detected — cross-reference attacker IPs against your production "
            "firewall logs and pre-emptively block them."
        )
    # FIX 4 (new): time-series volume spikes
    if counts.get("Attack Surge", 0):
        recs.append(
            "Attack-volume spikes detected in this time series — cross-reference the spike windows "
            "with production traffic logs to see if they line up with real incidents, and consider "
            "alerting when future buckets exceed the same threshold."
        )
    if not recs:
        recs.append("Continue monitoring and keep log collection enabled.")
    if "Generic" in mode or "rule-based" in mode:
        recs.append(
            "For stronger ML-based accuracy on this log schema, convert logs to "
            "CICFlowMeter-style flow features and retrain the model."
        )
    return recs


def show_top_ips(meta: dict):
    top_ips = meta.get("top_ips", {})
    if top_ips:
        st.subheader(meta.get("top_ips_label", "Top suspicious source IPs"))
        st.dataframe(
            pd.Series(top_ips, name=meta.get("top_ips_value_label", "Suspicious events"))
            .rename_axis(meta.get("top_ips_axis_label", "Source IP"))
            .reset_index(),
            use_container_width=True,
        )


def show_attack_trend(meta: dict):
    trend = meta.get("trend")
    if isinstance(trend, pd.Series) and not trend.empty:
        st.subheader("Attack trend over time")
        st.line_chart(trend)


def show_preview(result_df: pd.DataFrame):
    st.subheader("Analysed log preview")
    preferred_cols = [
        "Extracted Event Time",
        "Extracted Source IP",
        "Protocol",
        "Msg",
        "Description",
        "User",
        "Client",
        "Input",
        "eventid",
        "Bucket Total Attacks",
        "Predicted Attack Type",
        "Detection Reason",
    ]
    preview_cols = [c for c in preferred_cols if c in result_df.columns]
    st.dataframe(
        result_df[preview_cols].head(50) if preview_cols else result_df.head(50),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

model, scaler = load_artifacts()

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

user_email = st.text_input(
    "Email for alerts (sent only if risk is High or above)",
    placeholder="you@example.com",
    value=st.session_state.user_email,
    key="email_input",
)
st.session_state.user_email = user_email

email_valid = bool(EMAIL_RE.match(user_email.strip())) if user_email else False
if user_email and not email_valid:
    st.warning("That doesn't look like a valid email address.")

uploaded = st.file_uploader(
    "Upload a network log",
    type=["csv"],
)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

if uploaded:
    try:
        raw = read_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()

    if raw.empty:
        st.error("The uploaded file is empty.")
        st.stop()

    raw.columns = raw.columns.astype(str).str.strip()

    # ------------------------------------------------------------------
    # Route: CICIDS/CICFlowMeter -> Random Forest
    #      | pre-labeled web-attack feature dataset -> trust existing labels
    #      | aggregated time-series attack counts -> volume/spike analysis
    #      | everything else -> generic rule-based analyser (Zeek, honeypot,
    #        HTTP, generic CSV…)
    # ------------------------------------------------------------------

    skipped_invalid_rows = 0

    if looks_like_cicids(raw, FEATURE_COLUMNS):
        st.info(
            "CICIDS/CICFlowMeter 52-feature file detected. "
            "Using Random Forest ML model."
        )

        if model is None or scaler is None:
            st.error(
                "This file matches the ML format but the model files are missing. "
                "Run `python src/train.py` first."
            )
            st.stop()

        try:
            # FIX 1: preprocess_upload previously returned (X, fmt); guard
            # against either signature so the app works with both versions.
            raw_result = preprocess_upload(raw)
            X = raw_result[0] if isinstance(raw_result, tuple) else raw_result
        except Exception as e:
            st.error(f"Could not preprocess CICIDS/CICFlowMeter file: {e}")
            st.stop()

        if X.empty:
            st.error("No valid rows remained after preprocessing.")
            st.stop()

        X_scaled = scaler.transform(X)
        preds = pd.Series(
            model.predict(X_scaled),
            index=X.index,
            name="Predicted Attack Type",
        )

        result_df = raw.loc[X.index].copy()
        result_df["Predicted Attack Type"] = preds.values

        analysis_mode = "CICIDS/CICFlowMeter — Random Forest ML model"
        meta = {
            "mode": analysis_mode,
            "top_ips": {},
            "trend": pd.Series(dtype=int),
            "is_honeypot": False,
        }

    elif is_prelabeled_feature_format(raw):
        st.info(
            "Pre-labeled web-attack feature dataset detected (existing "
            "`class`/`prediction` columns found). Using those labels "
            "directly instead of re-deriving them from raw text."
        )

        try:
            result_df, preds, meta = analyze_prelabeled_features(raw)
        except Exception as e:
            st.error(f"Pre-labeled feature analysis failed: {e}")
            st.stop()

        skipped_invalid_rows = meta.get("skipped_invalid_rows", 0)
        if skipped_invalid_rows:
            st.warning(
                f"{skipped_invalid_rows:,} of {len(raw):,} rows have no real request data "
                "(placeholder method/path values) and were excluded from the risk metrics below, "
                "though they're still included in the downloadable CSV."
            )

        analysis_mode = meta.get("mode", "Pre-labeled web-attack feature dataset")

    elif is_timeseries_attack_format(raw):
        st.info(
            "Aggregated time-series attack-count file detected (Timestamp + "
            "pre-summed counts per bucket). Using volume/spike analysis "
            "instead of per-event pattern matching."
        )

        try:
            result_df, preds, meta = analyze_timeseries_logs(raw)
        except Exception as e:
            st.error(f"Time-series analysis failed: {e}")
            st.stop()

        st.caption(
            f"{meta.get('spike_buckets', 0):,} of {meta.get('total_buckets', 0):,} "
            "time buckets flagged as statistical volume spikes (>2 std. dev. above average)."
        )

        analysis_mode = meta.get("mode", "Aggregated time-series attack analysis")

    else:
        # FIX 3: detect and announce honeypot format before running
        # FIX 2: wrap in try/except so malformed logs show a clean error
        honeypot_detected = is_honeypot_format(raw)

        if honeypot_detected:
            st.warning(
                "Honeypot sensor log detected. "
                "Using rule-based analysis with honeypot-aware classification."
            )
        else:
            st.warning(
                "Generic network/security log detected. "
                "Using rule-based log analysis (no 52-column CICIDS features found)."
            )

        try:
            result_df, preds, meta = analyze_generic_logs(raw)
        except Exception as e:
            st.error(f"Generic log analysis failed: {e}")
            st.stop()

        analysis_mode = meta.get("mode", "Generic rule-based log analysis")

    # ------------------------------------------------------------------
    # Unify preds index for counting
    # ------------------------------------------------------------------

    preds = pd.Series(preds).reset_index(drop=True)

    counts = preds.value_counts().to_dict()
    total  = int(sum(counts.values()))

    # For regular logs, Honeypot Events are not "attacks" in the same sense;
    # for honeypot files every logged event IS suspicious, so count them.
    attacks = int(total - counts.get("Normal Traffic", 0))
    if not meta.get("is_honeypot"):
        attacks -= counts.get("Honeypot Event", 0)

    score = risk_score(counts)
    band  = risk_band(score)

    # FIX 3: surface honeypot info in the analysis caption
    mode_caption = analysis_mode
    if meta.get("is_honeypot"):
        mode_caption += " · 🍯 Honeypot sensor log"
    st.caption(f"Analysis mode: {mode_caption}")

    # 1) Prominent risk banner
    render_risk_banner(score, band)

    # 2) Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total entries",    f"{total:,}")
    c2.metric("Attacks detected", f"{attacks:,}")
    c3.metric("Attack rate",      f"{(attacks / total * 100):.1f}%" if total else "0%")
    c4.metric("Risk score",       f"{score} ({band})")
    if skipped_invalid_rows:
        st.caption(
            f"({skipped_invalid_rows:,} additional rows in the file had no usable "
            "request data and are not included in the totals above.)"
        )

    st.divider()

    # 3) Attack distribution
    st.subheader("Attack distribution")
    st.bar_chart(preds.value_counts().sort_values(ascending=False))

    # 4) Extra dashboard sections
    show_top_ips(meta)
    show_attack_trend(meta)
    show_preview(result_df)

    # 5) Download analysed CSV
    st.download_button(
        "Download analysed CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name="analysed_threat_log.csv",
        mime="text/csv",
    )

    # 6) PDF report
    recommendations = recommendations_for(counts, analysis_mode)
    summary = {
        "total":            total,
        "attacks_detected": attacks,
        "pred_counts":      counts,
        "risk_score":       score,
        "risk_band":        band,
        "top_ips":          list(meta.get("top_ips", {}).items()),
        "recommendations":  recommendations,
    }
    pdf_path = build_report(summary)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download PDF report",
            f,
            file_name="threat_report.pdf",
            mime="application/pdf",
        )

    # 7) Email alert
    if band in ("High", "Critical"):
        recipient = st.session_state.user_email.strip()
        if email_valid and recipient:
            try:
                sent = send_alert(
                    score,
                    band,
                    f"{attacks} attacks in {total} entries",
                    recipient=recipient,
                )
                if sent:
                    st.success(f"Alert email sent to {recipient}.")
                else:
                    st.info("Alert not sent. Check email credentials and terminal logs.")
            except Exception as e:
                st.error(f"Could not send email: {e}")
        else:
            st.warning("Enter a valid email above to receive an alert notification.")
    else:
        st.info("Risk is below High — no alert email sent.")