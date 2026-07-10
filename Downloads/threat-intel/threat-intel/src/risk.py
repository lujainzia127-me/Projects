"""
Risk scoring: turn per-class prediction counts into a 0-100 score and a band.
Tune SEVERITY weights with your supervisor - these are sensible defaults.
"""

# Higher = more dangerous. Benign is 0.
SEVERITY = {
    "Normal Traffic": 0,
    "Port Scanning": 2,
    "Bots": 3,
    "Brute Force": 3,
    "Web Attacks": 4,
    "DoS": 4,
    "DDoS": 5,
}


def risk_score(pred_counts: dict) -> float:
    """Weighted share of malicious traffic, scaled to 0-100."""
    total = sum(pred_counts.values())
    if total == 0:
        return 0.0
    weighted = sum(SEVERITY.get(label, 3) * n for label, n in pred_counts.items())
    max_weighted = max(SEVERITY.values()) * total
    return round(100 * weighted / max_weighted, 1)


def risk_band(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


if __name__ == "__main__":
    demo = {"Normal Traffic": 900, "DDoS": 80, "Port Scanning": 20}
    s = risk_score(demo)
    print("Score:", s, "Band:", risk_band(s))
