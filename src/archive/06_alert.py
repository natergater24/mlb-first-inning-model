"""
06_alert.py
-----------
Compares model probabilities to market odds and sends alerts
when an edge exceeds the configured threshold.

Prerequisites:
  - Run 05_build_features.py first (generates todays_matchups.parquet)
  - Optionally run a model inference script (not included here — see README)
  - Set alert credentials in .env

Alert channels:
  - Email (Gmail SMTP — free)
  - SMS via Twilio (~$0.01/text)
  - Console print (always runs — no config needed)

Usage:
  python src/06_alert.py                    # Run with defaults
  python src/06_alert.py --min-edge 5.0     # Only alert on 5%+ edges
  python src/06_alert.py --dry-run          # Print alerts, don't send
"""

import os
import sys
import logging
import smtplib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
ODDS_DIR = BASE_DIR / "data" / "odds"
LOG_DIR  = BASE_DIR / "logs"

load_dotenv(BASE_DIR / "env.txt")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "alerts.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Alert config from environment
EDGE_THRESHOLD    = float(os.getenv("EDGE_THRESHOLD", "4.0"))
EMAIL_FROM        = os.getenv("ALERT_EMAIL_FROM", "")
EMAIL_TO          = os.getenv("ALERT_EMAIL_TO", "")
EMAIL_PASS        = os.getenv("ALERT_EMAIL_PASSWORD", "")
TWILIO_SID        = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN      = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM       = os.getenv("TWILIO_FROM_NUMBER", "")
TWILIO_TO         = os.getenv("TWILIO_TO_NUMBER", "")


def american_to_prob(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def compute_edge(model_prob: float, market_prob: float) -> float:
    """
    Edge = model probability minus market's true probability (vig-removed).
    Positive = we think the event is more likely than the market does.
    """
    if pd.isna(model_prob) or pd.isna(market_prob):
        return np.nan
    return (model_prob - market_prob) * 100   # return as percent


def expected_value(model_prob: float, american_odds: float) -> float:
    """
    EV per $100 wagered.
    EV = model_prob * profit_if_win - (1 - model_prob) * 100
    """
    if pd.isna(model_prob) or pd.isna(american_odds):
        return np.nan
    if american_odds > 0:
        profit = american_odds
    else:
        profit = 10000 / abs(american_odds)
    return model_prob * profit - (1 - model_prob) * 100


def find_edges(matchups: pd.DataFrame, min_edge: float = None) -> pd.DataFrame:
    """
    Given matchup table with model_prob and market odds,
    identify rows where edge exceeds threshold.

    If the model hasn't been trained yet, this uses a simple
    heuristic based on BvP-adjusted OBP vs. market implied prob.
    """
    if min_edge is None:
        min_edge = EDGE_THRESHOLD

    df = matchups.copy()

    # Use model probability if available, otherwise fall back to BvP-adjusted OBP
    if "model_prob" not in df.columns:
        log.info("No model_prob column — using bvp_obp_adj as proxy estimate.")
        # Proxy: adjusted BvP OBP (career reach base rate vs this pitcher)
        # blended with league average platoon adjustment
        if "bvp_obp_adj" in df.columns:
            df["model_prob"] = df["bvp_obp_adj"]
        elif "obp" in df.columns:
            df["model_prob"] = df["obp"]
        else:
            log.warning("No probability estimate available.")
            return pd.DataFrame()

    # Market implied probability
    if "avg_implied_prob" in df.columns:
        df["market_prob"] = df["avg_implied_prob"]
    elif "best_odds" in df.columns:
        df["market_prob"] = df["best_odds"].apply(american_to_prob)
    else:
        log.warning("No market odds available — cannot compute edge.")
        return pd.DataFrame()

    # Compute edge and EV
    df["edge_pct"] = df.apply(
        lambda r: compute_edge(r["model_prob"], r["market_prob"]), axis=1
    )
    df["ev_per_100"] = df.apply(
        lambda r: expected_value(r["model_prob"], r.get("best_odds", np.nan)), axis=1
    )

    # Filter to significant edges
    edges = df[
        df["edge_pct"].notna() &
        (df["edge_pct"] >= min_edge)
    ].copy().sort_values("edge_pct", ascending=False)

    return edges


def format_alert_text(edges: pd.DataFrame, today: str) -> str:
    """Format edge rows into a human-readable alert."""
    if len(edges) == 0:
        return f"No edges found above threshold for {today}."

    lines = [
        f"MLB FIRST AT-BAT EDGES — {today}",
        f"{'=' * 50}",
        f"Found {len(edges)} edge(s) above {EDGE_THRESHOLD}%",
        "",
    ]

    for _, row in edges.iterrows():
        batter  = row.get("batter_name", row.get("batter", "?"))
        pitcher = row.get("pitcher_name", row.get("pitcher", "?"))
        side    = row.get("side", "?")   # home or away
        edge    = row.get("edge_pct", 0)
        ev      = row.get("ev_per_100", np.nan)
        model_p = row.get("model_prob", np.nan)
        mkt_p   = row.get("market_prob", np.nan)
        odds    = row.get("best_odds", np.nan)
        bvp_pa  = row.get("bvp_pa", row.get("pa", 0))

        lines += [
            f"► {batter} vs {pitcher} ({side})",
            f"  Model: {model_p:.1%}  |  Market: {mkt_p:.1%}  |  Edge: +{edge:.1f}%",
        ]
        if not pd.isna(odds):
            lines.append(f"  Best odds: {int(odds):+d}  |  EV per $100: ${ev:.2f}")
        if not pd.isna(bvp_pa) and bvp_pa > 0:
            lines.append(f"  BvP sample: {int(bvp_pa)} PA  {'✓ reliable' if row.get('bvp_reliable') else '⚠ small sample'}")

        # Supporting stats
        stat_parts = []
        for label, col in [
            ("BvP OBP", "bvp_obp"), ("30d OBP", "obp_30d"), ("K%", "bvp_k_pct"),
            ("Exit Velo", "bvp_exit_velo"), ("Temp", "temp_f"), ("Wind", "wind_speed"),
        ]:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if col in ("temp_f",):
                    stat_parts.append(f"{label}: {val:.0f}°F")
                elif col in ("wind_speed",):
                    stat_parts.append(f"{label}: {val:.0f}mph")
                elif col in ("bvp_exit_velo",):
                    stat_parts.append(f"{label}: {val:.1f}")
                else:
                    stat_parts.append(f"{label}: {val:.3f}")
        if stat_parts:
            lines.append("  " + "  |  ".join(stat_parts))

        lines.append("")

    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("⚠ Not financial advice. Model accuracy varies. Bet responsibly.")
    return "\n".join(lines)


def send_email(subject: str, body: str) -> bool:
    """Send via Gmail SMTP."""
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASS]):
        log.info("Email not configured — skipping.")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        log.info("Email sent to %s", EMAIL_TO)
        return True
    except Exception as e:
        log.error("Email failed: %s", e)
        return False


def send_sms(body: str) -> bool:
    """Send via Twilio SMS."""
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO]):
        log.info("Twilio not configured — skipping.")
        return False
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        # Truncate SMS to 1600 chars
        msg = client.messages.create(
            body  = body[:1600],
            from_ = TWILIO_FROM,
            to    = TWILIO_TO,
        )
        log.info("SMS sent: %s", msg.sid)
        return True
    except ImportError:
        log.warning("Twilio not installed. Run: pip install twilio")
        return False
    except Exception as e:
        log.error("SMS failed: %s", e)
        return False


def run(min_edge: float = None, dry_run: bool = False):
    today = datetime.now().strftime("%Y-%m-%d")

    matchup_path = PROC_DIR / "todays_matchups.parquet"
    if not matchup_path.exists():
        log.error("No today's matchups found. Run 05_build_features.py first.")
        return

    matchups = pd.read_parquet(matchup_path)
    log.info("Loaded %d matchups for %s", len(matchups), today)

    edges = find_edges(matchups, min_edge=min_edge or EDGE_THRESHOLD)
    alert_text = format_alert_text(edges, today)

    # Always print to console
    print("\n" + alert_text)

    if len(edges) == 0:
        log.info("No edges found — no alerts sent.")
        return

    if dry_run:
        log.info("Dry run — not sending alerts.")
        return

    subject = f"MLB Edge Alert {today} — {len(edges)} matchup(s)"
    send_email(subject, alert_text)
    send_sms(f"⚾ {subject}\n\n" + alert_text[:800])

    # Save edges to file
    edges_path = PROC_DIR / f"edges_{today}.parquet"
    edges.to_parquet(edges_path, index=False)
    log.info("Edges saved: %s", edges_path)


if __name__ == "__main__":
    min_edge = None
    dry_run  = "--dry-run" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--min-edge="):
            min_edge = float(arg.split("=")[1])
        elif arg.replace(".", "").lstrip("-").isdigit():
            min_edge = float(arg)
    run(min_edge=min_edge, dry_run=dry_run)
