"""
PDF threat report. Generates a Threat Intelligence Report with an embedded
attack-distribution chart (visual summary).
Uses fpdf2 (pip install fpdf2) and matplotlib.
"""

import os
from fpdf import FPDF

import matplotlib
matplotlib.use("Agg")          # no GUI needed, just render to a file
import matplotlib.pyplot as plt


def _safe(text) -> str:
    """fpdf2's core fonts (Helvetica) only support Latin-1. Strip/replace any
    character outside that range (smart quotes, en/em dashes, emoji, etc.)
    instead of letting them blow up rendering."""
    text = str(text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _reset_x(pdf):
    """Force the cursor back to the left margin. multi_cell() computes its
    available width from the CURRENT x position, so if a prior cell() call
    left x anywhere but the left margin, multi_cell can compute a width of
    ~0 and raise FPDFException('Not enough horizontal space...')."""
    pdf.set_x(pdf.l_margin)


def _make_chart(pred_counts: dict, path: str = "_report_chart.png"):
    """Render the attack distribution as a bar chart PNG for embedding.
    Returns the file path, or None if there is nothing to plot."""
    if not pred_counts:
        return None
    labels = list(pred_counts.keys())
    values = list(pred_counts.values())

    plt.figure(figsize=(6, 3))
    plt.bar(labels, values, color="#4C78A8")
    plt.title("Attack Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def build_report(summary: dict, out_path: str = "threat_report.pdf"):
    """
    summary expected keys:
      total, attacks_detected, pred_counts (dict),
      risk_score, risk_band,recommendations (list)
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _safe("Threat Intelligence Report"), ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, _safe(f"Total log entries: {summary.get('total', 0)}"), ln=True)
    pdf.cell(0, 8, _safe(f"Attacks detected: {summary.get('attacks_detected', 0)}"), ln=True)
    pdf.cell(0, 8,
             _safe(f"Risk score: {summary.get('risk_score', 0)} "
                   f"({summary.get('risk_band', 'Low')})"), ln=True)

    pdf.ln(4)
    _reset_x(pdf)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _safe("Attack breakdown"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    for label, n in summary.get("pred_counts", {}).items():
        _reset_x(pdf)
        pdf.cell(0, 7, _safe(f"  {label}: {n}"), ln=True)

    # --- Visual summary: embedded attack-distribution chart ---
    chart_path = _make_chart(summary.get("pred_counts", {}))
    if chart_path:
        pdf.ln(4)
        _reset_x(pdf)
        pdf.image(chart_path, w=170)

    pdf.ln(4)
    _reset_x(pdf)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _safe("Recommended actions"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    for rec in summary.get("recommendations", []):
        _reset_x(pdf)
        pdf.multi_cell(0, 7, _safe(f"  - {rec}"))

    pdf.output(out_path)

    # tidy up the temp chart image
    if chart_path and os.path.exists(chart_path):
        os.remove(chart_path)

    return out_path