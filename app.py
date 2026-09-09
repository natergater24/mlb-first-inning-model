#!/usr/bin/env python3
"""
app.py — YRFI / NRFI Predictions dashboard.

One question: will a run be scored in the first inning of this game?

Landing page  : game list with model NRFI probability, first-inning sportsbook
                odds, edge and recommended bet for every game on the slate.
Detail view   : full breakdown for one game — model summary + book table,
                weather/park, both pitchers' first-inning profiles, both teams'
                projected top-5 batters, and historical YRFI context.

Run:  /Library/Frameworks/Python.framework/Versions/3.14/bin/streamlit run app.py --server.headless true
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import bet_tracker as bt

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
ODDS = ROOT / "data" / "odds"
MODELS = ROOT / "data" / "models"
PY = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"

st.set_page_config(page_title="YRFI / NRFI Predictions", page_icon="⚾", layout="wide")

TODAY = date.today().isoformat()

BOOK_META = {  # short -> (label, color)
    "DK": ("DraftKings", "#53d337"),
    "FD": ("FanDuel", "#1493ff"),
    "MGM": ("BetMGM", "#c8a24a"),
    "CZR": ("Caesars", "#1b3c5d"),
}
BOOK_BET_URL = {
    "DK": "https://sportsbook.draftkings.com/leagues/baseball/mlb",
    "FD": "https://sportsbook.fanduel.com/baseball",
    "MGM": "https://sports.betmgm.com/en/sports/baseball-23",
    "CZR": "https://sportsbook.caesars.com/us/nj/sport/baseball",
}


# ── helpers ────────────────────────────────────────────────────────────────
def fmt_stat(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if 0 <= abs(f) < 1:
        return f"{f:.{nd}f}".lstrip("0") or "0"
    return f"{f:.{nd}f}"


def fmt_pct(v, nd=0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.{nd}f}%"


def fmt_odds(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    v = int(round(float(v)))
    return f"+{v}" if v > 0 else str(v)


def american_odds(p) -> float:
    """Implied American odds for probability p."""
    p = float(p)
    if not np.isfinite(p) or p <= 0 or p >= 1:
        return np.nan
    return round(-p / (1 - p) * 100) if p > 0.5 else round((1 - p) / p * 100)


def nrfi_color(pct) -> str:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "#888"
    p = float(pct)
    if p >= 65:
        return "#1a9850"
    if p >= 55:
        return "#d9a300"
    return "#d73027"


def safe(v, default=np.nan):
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


# ── loaders ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def load_predictions() -> pd.DataFrame:
    p = PROC / "todays_yrfi_predictions.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(ttl=300)
def load_pitcher_profiles() -> pd.DataFrame:
    p = PROC / "pitcher_nrfi_profile.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_last5() -> pd.DataFrame:
    p = PROC / "pitcher_last5_starts.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_top5_teams() -> pd.DataFrame:
    p = PROC / "projected_top5_by_team.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_batter_stats() -> pd.DataFrame:
    p = PROC / "top5_batter_stats.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_bvp() -> pd.DataFrame:
    p = PROC / "bvp_full_lifetime.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_lineup_summary() -> pd.DataFrame:
    p = PROC / "lineup_position_summary.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_yrfi_outcomes() -> pd.DataFrame:
    p = PROC / "yrfi_outcomes.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_model_meta() -> dict:
    p = MODELS / "yrfi_model_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=300)
def load_game_meta() -> pd.DataFrame:
    p = PROC / "game_meta.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def predictions_mtime() -> datetime | None:
    p = PROC / "todays_yrfi_predictions.parquet"
    return datetime.fromtimestamp(p.stat().st_mtime) if p.exists() else None


def confirmed_starters(pred: pd.DataFrame, gm: pd.DataFrame) -> set[int]:
    if gm.empty:
        return set()
    g = gm[gm["game_date"].astype(str) == TODAY]
    ok = g[g["home_starter_id"].notna() & g["away_starter_id"].notna()]
    return set(ok["game_pk"].astype(int))


# ── refresh ────────────────────────────────────────────────────────────────
def run_refresh():
    steps = [
        ("Probable pitchers", ["src/02_fetch_mlb_api.py", "--probable-pitchers-only"]),
        ("Active rosters", ["src/02_fetch_mlb_api.py", "--rosters-only"]),
        ("Weather", ["src/03_fetch_weather.py"]),
        ("First-inning odds", ["src/04_fetch_odds.py"]),
        ("YRFI predictions", ["src/14_build_todays_yrfi.py"]),
    ]
    with st.status("Running daily update…", expanded=True) as status:
        for label, args in steps:
            st.write(f"→ {label}")
            r = subprocess.run(["caffeinate", "-i", PY] + args, cwd=ROOT,
                               capture_output=True, text=True)
            if r.returncode != 0:
                st.error(f"{label} failed:\n{r.stderr[-1500:]}")
                status.update(label="Refresh failed", state="error")
                return
        status.update(label="Refresh complete", state="complete")
    st.cache_data.clear()
    st.rerun()


# ── model weighting sliders ───────────────────────────────────────────────
WEIGHT_GROUPS = ["pitcher_nrfi_record", "pitcher_stuff", "pitcher_last5",
                 "lineup_offense", "lineup_recent_form", "ballpark", "weather"]
DEFAULT_LABELS = {
    "pitcher_nrfi_record": "Pitcher NRFI track record (career + season)",
    "pitcher_stuff": "Pitcher stuff (velo · K% · BB% · contact)",
    "pitcher_last5": "Pitcher recent form (last 5 starts)",
    "lineup_offense": "Lineup power & matchup (BvP · SLG · HR rate)",
    "lineup_recent_form": "Lineup recent form (L7 / L30 OBP)",
    "ballpark": "Ballpark run environment",
    "weather": "Weather (temperature · wind)",
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def render_weight_controls(meta: dict) -> dict:
    """Sliders that re-weight each feature group around the model's own
    prediction. Returns {group: multiplier}. 1.0 == the model as trained."""
    w = meta.get("weighting", {})
    labels = w.get("group_labels", DEFAULT_LABELS)
    influence = w.get("group_influence", {})
    imax = max(influence.values()) if influence else 1.0

    st.session_state.setdefault("weights", {g: 100 for g in WEIGHT_GROUPS})
    active = any(v != 100 for v in st.session_state["weights"].values())

    with st.expander(f"⚙️ Model weighting{'  —  CUSTOM WEIGHTS ACTIVE' if active else ''}",
                     expanded=active):
        st.caption("Each slider scales how much that group of stats moves the "
                   "prediction, around the model's own weighting. 100% = the model "
                   "as trained. The bar shows how much the model currently leans on "
                   "each group.")
        cols = st.columns(2)
        for i, g in enumerate(WEIGHT_GROUPS):
            c = cols[i % 2]
            infl = influence.get(g, 0.0)
            bar = int(round(100 * infl / imax)) if imax else 0
            cur = st.session_state["weights"][g]
            tag = "" if cur == 100 else f" <b style='color:#1493ff'>· {cur}%</b>"
            c.markdown(
                f"<div style='font-size:12px;font-weight:600;margin-bottom:2px'>"
                f"{labels.get(g, g)}{tag}</div>"
                f"<div style='font-size:10px;color:#999;margin-bottom:1px'>"
                f"model currently leans this much →</div>"
                f"<div style='height:6px;width:100%;background:#eee;border-radius:3px;"
                f"margin-bottom:2px'><div style='height:6px;width:{bar}%;"
                f"background:#1493ff;border-radius:3px'></div></div>",
                unsafe_allow_html=True)
            st.session_state["weights"][g] = c.slider(
                labels.get(g, g), 0, 200, cur, 10,
                key=f"w_{g}", label_visibility="collapsed",
                help="100% = model as trained · 0% = ignore this group · 200% = double its pull")
            c.write("")

        # ── live effective-weight breakdown (updates instantly with the sliders) ──
        eff = {g: (st.session_state["weights"][g] / 100.0) * influence.get(g, 0.0)
               for g in WEIGHT_GROUPS}
        tot = sum(eff.values()) or 1.0
        share = {g: 100 * v / tot for g, v in eff.items()}
        st.markdown("**Effective weighting right now** "
                    "<span style='color:#999;font-size:11px'>(slider % × the model's lean, "
                    "renormalised to 100%)</span>", unsafe_allow_html=True)
        seg = ""
        for i, g in enumerate(WEIGHT_GROUPS):
            s = share[g]
            lbl = f"{s:.0f}%" if s >= 7 else ""
            seg += (f"<div style='width:{s:.2f}%;background:{_GROUP_COLOR[i]};height:22px;"
                    f"display:inline-block;color:#fff;font-size:10px;text-align:center;"
                    f"line-height:22px;overflow:hidden' title='{labels.get(g, g)}'>{lbl}</div>")
        st.markdown(f"<div style='width:100%;border-radius:4px;overflow:hidden;"
                    f"white-space:nowrap;font-size:0'>{seg}</div>", unsafe_allow_html=True)
        leg = "  ".join(
            f"<span style='color:{_GROUP_COLOR[i]}'>■</span> "
            f"<span style='font-size:11px'>{labels.get(g, g).split('(')[0].strip()} "
            f"{share[g]:.0f}%</span>"
            for i, g in enumerate(sorted(WEIGHT_GROUPS, key=lambda x: -share[x])))
        st.markdown(f"<div style='margin-top:4px'>{leg}</div>", unsafe_allow_html=True)

        b1, b2 = st.columns([1, 4])
        if b1.button("Reset to model"):
            for g in WEIGHT_GROUPS:
                st.session_state["weights"][g] = 100
                st.session_state.pop(f"w_{g}", None)  # let the slider re-init to 100
            st.rerun()
        if active:
            b2.caption("Cards below show **model → your weighting**. Edges, EV and "
                       "recommendations use your weighting.")
    return {g: v / 100.0 for g, v in st.session_state["weights"].items()}


_GROUP_COLOR = ["#1f6f4f", "#2f7fb5", "#d98324", "#7b5ea7", "#c0568a", "#5a8f3c", "#c94c4c"]


def _recommend(yrfi_prob, nrfi_prob, yrfi_edge, nrfi_edge, best_y, best_n):
    if pd.notna(yrfi_edge) or pd.notna(nrfi_edge):
        if (yrfi_edge if pd.notna(yrfi_edge) else -99) > 4:
            return "YRFI", best_y
        if (nrfi_edge if pd.notna(nrfi_edge) else -99) > 4:
            return "NRFI", best_n
        return "NO EDGE", np.nan
    if yrfi_prob > 0.60:
        return "YRFI MODEL ONLY", np.nan
    if nrfi_prob > 0.60:
        return "NRFI MODEL ONLY", np.nan
    return "NO EDGE", np.nan


def _ev(side, odds, yrfi_prob, nrfi_prob):
    s = str(side).split()[0]
    if s not in ("YRFI", "NRFI") or pd.isna(odds):
        return np.nan
    p = yrfi_prob if s == "YRFI" else nrfi_prob
    payout = odds if odds > 0 else 10000 / abs(odds)
    return round(p * payout - (1 - p) * 100, 2)


def apply_weights(pred: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Re-derive model prob / odds / edge / recommendation for each game from the
    user's group weights. No-op when every weight is 1.0 or contributions are
    missing (model not retrained with a surrogate)."""
    pred = pred.copy()
    pred["_weights_active"] = any(abs(v - 1.0) > 1e-9 for v in weights.values())
    if "base_model_logit" not in pred.columns or not pred["_weights_active"].iloc[0]:
        return pred

    contrib_cols = {g: f"contrib_{g}" for g in weights if f"contrib_{g}" in pred.columns}

    for c in ("model_yrfi_prob", "model_nrfi_prob", "yrfi_model_odds", "nrfi_model_odds",
              "nrfi_edge_pct", "yrfi_edge_pct", "recommended_bet", "recommended_bet_odds",
              "recommended_bet_ev"):
        if c in pred.columns:
            pred[c + "_model"] = pred[c]

    adj_logit = pred["base_model_logit"].astype(float).copy()
    for g, col in contrib_cols.items():
        adj_logit = adj_logit + (weights[g] - 1.0) * pred[col].astype(float).fillna(0.0)
    yp = _sigmoid(adj_logit.values)
    pred["model_yrfi_prob"] = np.round(yp, 4)
    pred["model_nrfi_prob"] = np.round(1 - yp, 4)
    pred["yrfi_model_odds"] = [american_odds(p) for p in yp]
    pred["nrfi_model_odds"] = [american_odds(1 - p) for p in yp]

    if "yrfi_market_avg_implied" in pred.columns:
        pred["yrfi_edge_pct"] = np.where(
            pred["yrfi_market_avg_implied"].notna(),
            (pred["model_yrfi_prob"] - pred["yrfi_market_avg_implied"]) * 100, np.nan)
        pred["nrfi_edge_pct"] = np.where(
            pred["nrfi_market_avg_implied"].notna(),
            (pred["model_nrfi_prob"] - pred["nrfi_market_avg_implied"]) * 100, np.nan)

    recs, odds_, evs = [], [], []
    for _, r in pred.iterrows():
        side, o = _recommend(r["model_yrfi_prob"], r["model_nrfi_prob"],
                             r.get("yrfi_edge_pct"), r.get("nrfi_edge_pct"),
                             r.get("best_yrfi_odds"), r.get("best_nrfi_odds"))
        recs.append(side)
        odds_.append(o)
        evs.append(_ev(side, o, r["model_yrfi_prob"], r["model_nrfi_prob"]))
    pred["recommended_bet"] = recs
    pred["recommended_bet_odds"] = odds_
    pred["recommended_bet_ev"] = evs
    return pred


# ═══════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════
def page_landing(pred: pd.DataFrame):
    gm = load_game_meta()
    conf = confirmed_starters(pred, gm)
    pred = pred.copy()
    pred["_confirmed"] = pred["game_pk"].astype(int).isin(conf)

    # header
    left, right = st.columns([3, 1])
    with left:
        st.title("⚾ YRFI / NRFI Predictions")
        st.caption(f"{datetime.now():%A %B %-d, %Y}  ·  {datetime.now():%-I:%M %p}")
    with right:
        mt = predictions_mtime()
        if mt:
            age_min = (datetime.now() - mt).total_seconds() / 60
            col = "#1a9850" if age_min < 30 else "#d9a300" if age_min < 120 else "#d73027"
            st.markdown(
                f"<div style='text-align:right'>updated "
                f"<span style='color:{col};font-weight:700'>"
                f"{mt:%-I:%M %p}</span><br><span style='color:{col};font-size:12px'>"
                f"{age_min:.0f} min ago</span></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh slate", use_container_width=True):
            run_refresh()

    if pred.empty:
        st.warning("No predictions for today. Click **Refresh slate**.")
        return

    strong_nrfi = int(((pred.get("nrfi_edge_pct", pd.Series(dtype=float))) > 4).sum())
    strong_yrfi = int(((pred.get("yrfi_edge_pct", pd.Series(dtype=float))) > 4).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games today", len(pred))
    m2.metric("Strong NRFI edges", strong_nrfi)
    m3.metric("Strong YRFI edges", strong_yrfi)
    m4.metric("Confirmed starters", f"{pred['_confirmed'].sum()}/{len(pred)}")

    st.divider()

    # filters
    f1, f2, f3 = st.columns([2, 2, 2])
    view = f1.radio("Show", ["All Games", "NRFI Edge", "YRFI Edge", "No Edge"],
                    horizontal=True, label_visibility="collapsed")
    sort_by = f2.selectbox("Sort by", ["Edge %", "Game Time", "Home Team", "Away Team"],
                           label_visibility="collapsed")
    only_conf = f3.toggle("Confirmed starters only")

    df = pred.copy()
    df["_abs_edge"] = df[["nrfi_edge_pct", "yrfi_edge_pct"]].abs().max(axis=1) \
        if "nrfi_edge_pct" in df.columns else 0.0
    if only_conf:
        df = df[df["_confirmed"]]
    if view == "NRFI Edge":
        df = df[df.get("nrfi_edge_pct", 0) > 4]
    elif view == "YRFI Edge":
        df = df[df.get("yrfi_edge_pct", 0) > 4]
    elif view == "No Edge":
        df = df[~((df.get("nrfi_edge_pct", 0) > 4) | (df.get("yrfi_edge_pct", 0) > 4))]

    if sort_by == "Edge %":
        df = df.sort_values("_abs_edge", ascending=False, na_position="last")
    elif sort_by == "Home Team":
        df = df.sort_values("home_team")
    elif sort_by == "Away Team":
        df = df.sort_values("away_team")
    else:
        df = df.sort_values("game_pk")

    if df.empty:
        st.info("No games match this filter.")
        return

    for _, r in df.iterrows():
        game_card(r)


def logo_img(team_id, size=22) -> str:
    if team_id is None or (isinstance(team_id, float) and np.isnan(team_id)):
        return ""
    return (f"<img src='https://www.mlbstatic.com/team-logos/{int(team_id)}.svg' "
            f"width='{size}' height='{size}' "
            f"style='vertical-align:middle;margin-right:6px'>")


def _fmt_checked(iso) -> str:
    if not iso or (isinstance(iso, float) and np.isnan(iso)):
        return "not checked"
    try:
        dt = datetime.fromisoformat(str(iso))
        return dt.strftime("%b %-d, %-I:%M %p")
    except ValueError:
        return str(iso)


def _better_value_side(r) -> str:
    """Which of NRFI / YRFI is the better value pick."""
    ne, ye = r.get("nrfi_edge_pct"), r.get("yrfi_edge_pct")
    if pd.notna(ne) and pd.notna(ye):
        return "NRFI" if ne >= ye else "YRFI"
    return "NRFI" if float(r["model_nrfi_prob"]) >= float(r["model_yrfi_prob"]) else "YRFI"


def _bet_url(r, side: str, bk: str) -> tuple[str, bool]:
    """(url, is_instant) — the per-selection betslip deeplink if SGO gave one,
    else the book's MLB lobby."""
    dl = r.get(f"{side}_link_{bk}")
    if isinstance(dl, str) and dl.startswith("http"):
        return dl, True
    return BOOK_BET_URL.get(bk, "#"), False


def _book_line_table(r) -> str:
    """Compact HTML table: NRFI / YRFI rows across DK / FD / MGM / CZR.
    Each listed line links to that book — ⚡ = instant add-to-betslip deeplink,
    ↗ = book's MLB page (no deeplink offered for that book)."""
    head = ""
    for bk, (label, color) in BOOK_META.items():
        head += (f"<th style='padding:2px 8px;font-size:11px'>"
                 f"<span style='color:{color};font-weight:700'>{bk}</span></th>")
    body = ""
    for side, label in (("nrfi", "NRFI"), ("yrfi", "YRFI")):
        cells = ""
        for bk, (blabel, color) in BOOK_META.items():
            o = r.get(f"{side}_odds_{bk}")
            txt = fmt_odds(o)
            if txt == "—":
                cells += ("<td style='padding:2px 8px;text-align:center;font-size:13px;"
                          "color:#ccc'>—</td>")
                continue
            url, instant = _bet_url(r, side, bk)
            mark = "⚡" if instant else ""
            tip = "add to betslip instantly" if instant else "open book"
            cells += (f"<td style='padding:2px 8px;text-align:center;font-size:13px'>"
                      f"<a href='{url}' target='_blank' title='{tip}' "
                      f"style='color:{color};text-decoration:none;font-weight:600'>"
                      f"{txt}{mark}</a></td>")
        body += (f"<tr><td style='padding:2px 8px;font-weight:700;font-size:12px'>"
                 f"{label}</td>{cells}</tr>")
    return (f"<table style='border-collapse:collapse'><tr>"
            f"<th></th>{head}</tr>{body}</table>")


def game_card(r):
    pk = int(r["game_pk"])
    nrfi = float(r["model_nrfi_prob"]) * 100
    yrfi = float(r["model_yrfi_prob"]) * 100
    rec = str(r.get("recommended_bet", "NO EDGE"))
    edge = r.get("nrfi_edge_pct") if "NRFI" in rec else r.get("yrfi_edge_pct") if "YRFI" in rec else np.nan
    better = _better_value_side(r)
    conf_note = "✅ Confirmed" if r["_confirmed"] else "⏳ probable"

    badge = ("✅ BET NRFI", "#1a9850") if rec == "NRFI" else \
            ("✅ BET YRFI", "#1493ff") if rec == "YRFI" else \
            ("📈 MODEL: NRFI", "#5b8a72") if rec == "NRFI MODEL ONLY" else \
            ("📈 MODEL: YRFI", "#6a8fb5") if rec == "YRFI MODEL ONLY" else \
            ("⚪ NO EDGE", "#888")

    def team_line(side):
        tid = r.get(f"{side}_team_id")
        abbr = r[f"{side}_team"]
        record = r.get(f"{side}_team_record") or ""
        pit = r.get(f"{side}_pitcher_name") or "TBD"
        sc = nrfi_color(r.get(f"{side}_pitcher_seas_nrfi_pct"))
        return (f"<div style='line-height:1.5;white-space:nowrap'>{logo_img(tid)}"
                f"<b style='font-size:15px'>{abbr}</b> "
                f"<span style='color:#888;font-size:12px'>{record}</span> &nbsp;·&nbsp; "
                f"<span style='font-size:13px'>{pit}</span> "
                f"<span style='color:{sc};font-size:12px'>"
                f"(NRFI {fmt_pct(r.get(f'{side}_pitcher_seas_nrfi_pct'))}, "
                f"L5 {fmt_pct(r.get(f'{side}_pitcher_l5_nrfi_pct'))})</span></div>")

    with st.container(border=True):
        left, mid, right = st.columns([5, 2, 4])

        # ── teams stacked + park / weather ───────────────────────────────
        with left:
            dome = bool(r.get("is_dome"))
            wx = "🏠 Dome" if dome else (
                f"{fmt_stat(r.get('temp_f'),0)}°F · {fmt_stat(r.get('wind_speed'),0)} mph")
            st.markdown(
                team_line("away") + team_line("home") +
                f"<div style='color:#888;font-size:12px;margin-top:2px'>"
                f"@ {r.get('park_name','')} · {wx} · <span>{conf_note}</span></div>",
                unsafe_allow_html=True)

        # ── model YRFI / NRFI odds, better value bolded ──────────────────
        with mid:
            wa = bool(r.get("_weights_active"))
            def orow(side, odds, prob, om_prob=None):
                b = side == better
                w = "800" if b else "400"
                c = "#111" if b else "#888"
                mark = " ◄" if b else ""
                was = (f"<span style='color:#bbb;font-size:10px'> "
                       f"(model {om_prob:.0f}%)</span>" if wa and om_prob is not None else "")
                return (f"<div style='font-size:14px;color:{c};font-weight:{w}'>"
                        f"{side}&nbsp;&nbsp;{fmt_odds(odds)}"
                        f"<span style='font-size:11px'>&nbsp;{prob:.0f}%{mark}</span>"
                        f"{was}</div>")
            om_n = float(r["model_nrfi_prob_model"]) * 100 if wa and "model_nrfi_prob_model" in r else None
            om_y = float(r["model_yrfi_prob_model"]) * 100 if wa and "model_yrfi_prob_model" in r else None
            st.markdown(
                f"<div style='font-size:11px;color:#888;letter-spacing:.5px'>"
                f"{'⚙️ YOUR WEIGHTING' if wa else 'MODEL'}</div>"
                + orow("NRFI", r.get("nrfi_model_odds"), nrfi, om_n)
                + orow("YRFI", r.get("yrfi_model_odds"), yrfi, om_y),
                unsafe_allow_html=True)

        # ── book lines + checked timestamp ──────────────────────────────
        with right:
            st.markdown(
                "<div style='font-size:11px;color:#888;letter-spacing:.5px'>"
                "BOOK LINES <span style='font-weight:400'>· ⚡ adds bet to slip · ↗ opens book"
                "</span></div>"
                + _book_line_table(r)
                + f"<div style='font-size:10px;color:#999;margin-top:2px'>"
                f"checked {_fmt_checked(r.get('odds_checked_at'))} · ⚡ works on Caesars & BetMGM · "
                f"DraftKings & FanDuel open their MLB lobby (no working instant link)</div>",
                unsafe_allow_html=True)

        # ── recommendation strip ────────────────────────────────────────
        b1, b2, b3 = st.columns([2, 4, 2])
        b1.markdown(f"<span style='background:{badge[1]};color:#fff;border-radius:5px;"
                    f"padding:3px 10px;font-weight:700;font-size:13px'>{badge[0]}</span>",
                    unsafe_allow_html=True)
        parts = []
        if pd.notna(edge):
            ecol = "#1a9850" if edge > 4 else "#888"
            parts.append(f"<span style='color:{ecol};font-weight:700'>{edge:+.1f}% "
                         f"{'NRFI' if 'NRFI' in rec else 'YRFI'} edge</span>")
        else:
            parts.append("<span style='color:#888'>no market edge</span>")
        ev = r.get("recommended_bet_ev")
        if pd.notna(ev):
            parts.append(f"EV/$100 <b>${ev:+.2f}</b>")
        side_l = "nrfi" if "NRFI" in rec else "yrfi"
        bb = r.get(f"best_{side_l}_book")
        bo = r.get(f"best_{side_l}_odds")
        if isinstance(bb, str):
            blabel = BOOK_META.get(bb, (bb, "#1493ff"))[0]
            bl = r.get(f"best_{side_l}_link")
            instant = isinstance(bl, str) and bl.startswith("http")
            burl = bl if instant else BOOK_BET_URL.get(bb, "#")
            mark = "⚡" if instant else "↗"
            parts.append(f"best <a href='{burl}' target='_blank' "
                         f"style='color:#1493ff;font-weight:700;text-decoration:none'>"
                         f"{blabel} {fmt_odds(bo)}{mark}</a>")
        b2.markdown("<div style='font-size:13px;padding-top:4px'>"
                    + " &nbsp;·&nbsp; ".join(parts) + "</div>", unsafe_allow_html=True)
        if b3.button("View details →", key=f"det_{pk}", use_container_width=True):
            st.session_state["game"] = pk
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  DETAIL VIEW
# ═══════════════════════════════════════════════════════════════════════════
def page_detail(pk: int, pred: pd.DataFrame):
    row = pred[pred["game_pk"].astype(int) == pk]
    if row.empty:
        st.session_state.pop("game", None)
        st.rerun()
    r = row.iloc[0]

    if st.button("← Back to All Games"):
        st.session_state.pop("game", None)
        st.rerun()

    dome = bool(r.get("is_dome"))
    wx = "🏠 Dome" if dome else f"{fmt_stat(r.get('temp_f'),0)}°F · wind {fmt_stat(r.get('wind_speed'),0)} mph"
    st.header(f"{r['away_team']} @ {r['home_team']}")
    st.caption(f"{TODAY}  ·  {r.get('park_name','')}  ·  {wx}")

    section_prediction(r)
    section_track_bet(r)
    section_weather_park(r)
    profiles = load_pitcher_profiles()
    last5 = load_last5()
    section_pitcher(r, profiles, last5, side="away")
    section_pitcher(r, profiles, last5, side="home")
    section_batters(r, side="away")
    section_batters(r, side="home")
    section_history(r)


def section_prediction(r):
    st.subheader("1 · Model Prediction Summary")
    nrfi = float(r["model_nrfi_prob"])
    yrfi = float(r["model_yrfi_prob"])
    if bool(r.get("_weights_active")) and "model_nrfi_prob_model" in r:
        st.info(f"⚙️ **Custom weighting active** — model's own prediction was "
                f"NRFI {float(r['model_nrfi_prob_model'])*100:.1f}% / "
                f"YRFI {float(r['model_yrfi_prob_model'])*100:.1f}%. "
                f"Adjust or reset in **⚙️ Model weighting** at the top.")
    c1, c2 = st.columns(2)
    c1.markdown(f"### 🟢 {nrfi*100:.1f}% — No Run in the 1st")
    c1.markdown(f"Model NRFI odds: **`{fmt_odds(r.get('nrfi_model_odds'))}`**")
    c2.markdown(f"### 🔴 {yrfi*100:.1f}% — Yes Run in the 1st")
    c2.markdown(f"Model YRFI odds: **`{fmt_odds(r.get('yrfi_model_odds'))}`**")

    st.markdown(
        f"<div style='display:flex;height:26px;border-radius:5px;overflow:hidden;font-size:12px'>"
        f"<div style='width:{nrfi*100:.1f}%;background:#1a9850;color:#fff;text-align:center;"
        f"line-height:26px'>NRFI {nrfi*100:.0f}%</div>"
        f"<div style='width:{yrfi*100:.1f}%;background:#d73027;color:#fff;text-align:center;"
        f"line-height:26px'>YRFI {yrfi*100:.0f}%</div></div>", unsafe_allow_html=True)
    st.write("")

    rows = []
    best_n_edge = best_y_edge = -99
    for bk, (label, _) in BOOK_META.items():
        no_ = r.get(f"nrfi_odds_{bk}"); yo_ = r.get(f"yrfi_odds_{bk}")
        ni = r.get(f"nrfi_implied_{bk}"); yi = r.get(f"yrfi_implied_{bk}")
        n_edge = (nrfi - ni) * 100 if pd.notna(ni) else np.nan
        y_edge = (yrfi - yi) * 100 if pd.notna(yi) else np.nan
        if pd.notna(n_edge):
            best_n_edge = max(best_n_edge, n_edge)
        if pd.notna(y_edge):
            best_y_edge = max(best_y_edge, y_edge)
        rows.append({"Book": label,
                     "NRFI Odds": fmt_odds(no_), "NRFI Impl%": fmt_pct(ni*100, 1) if pd.notna(ni) else "—",
                     "NRFI Edge%": f"{n_edge:+.1f}" if pd.notna(n_edge) else "—",
                     "YRFI Odds": fmt_odds(yo_), "YRFI Impl%": fmt_pct(yi*100, 1) if pd.notna(yi) else "—",
                     "YRFI Edge%": f"{y_edge:+.1f}" if pd.notna(y_edge) else "—"})
    tbl = pd.DataFrame(rows)

    def hl(row):
        s = [""] * len(row)
        try:
            if row["NRFI Edge%"] != "—" and abs(float(row["NRFI Edge%"]) - best_n_edge) < 0.05:
                s = ["background-color:#d6f0d6;color:#0a3d0a"] * len(row)
            elif row["YRFI Edge%"] != "—" and abs(float(row["YRFI Edge%"]) - best_y_edge) < 0.05:
                s = ["background-color:#d6e6f5;color:#0a2a4a"] * len(row)
        except ValueError:
            pass
        return s

    st.dataframe(tbl.style.apply(hl, axis=1), hide_index=True, use_container_width=True)

    rec = str(r.get("recommended_bet", "NO EDGE"))
    bet_side = rec.split()[0] if rec.split()[0] in ("NRFI", "YRFI") else _better_value_side(r)
    st.caption(f"Add **{bet_side}** to betslip  ·  ⚡ = adds the exact selection instantly "
               f"(Caesars, BetMGM)  ·  ↗ = opens the book's MLB page "
               f"(DraftKings & FanDuel — no working instant link for this market)")
    bcols = st.columns(len(BOOK_META))
    for i, bk in enumerate(BOOK_META):
        url, instant = _bet_url(r, bet_side.lower(), bk)
        odds = fmt_odds(r.get(f"{bet_side.lower()}_odds_{bk}"))
        bcols[i].link_button(
            f"{'⚡' if instant else '↗'} {bk} {bet_side} {odds if odds!='—' else ''}".strip(),
            url, use_container_width=True)

    if rec in ("NRFI", "YRFI"):
        side = rec
        bb = r.get(f"best_{side.lower()}_book"); bo = r.get(f"best_{side.lower()}_odds")
        bl = r.get(f"best_{side.lower()}_link")
        ev = r.get("recommended_bet_ev")
        edge = r.get(f"{side.lower()}_edge_pct")
        st.success(f"🎯 **Best Bet: {side} at {BOOK_META.get(bb,(bb,))[0]} {fmt_odds(bo)}** "
                   f"— Edge: {edge:+.1f}% — EV per \\$100: \\${ev:+.2f}")
        if isinstance(bl, str) and bl.startswith("http"):
            st.link_button(f"⚡ Add {side} to {BOOK_META.get(bb,(bb,))[0]} betslip", bl)
    elif "MODEL ONLY" in rec:
        st.info(f"📈 Model leans **{rec.split()[0]}** ({max(nrfi,yrfi)*100:.1f}%), "
                f"but no market edge ≥ 4%.")
    else:
        st.caption("⚪ No actionable edge — model and market agree.")


def section_weather_park(r):
    st.subheader("2 · Weather & Park Context")
    c1, c2 = st.columns(2)
    with c1:
        if bool(r.get("is_dome")):
            st.markdown("🏠 **Dome stadium — weather not a factor**")
        else:
            wc = r.get("wind_out_component", 0) or 0
            wdir = "blowing out ➡️" if wc > 2 else "blowing in ⬅️" if wc < -2 else "crosswind ↕️"
            st.markdown(
                f"- Temperature: **{fmt_stat(r.get('temp_f'),0)}°F**\n"
                f"- Wind: **{fmt_stat(r.get('wind_speed'),0)} mph** ({wdir})\n"
                f"- Humidity: **{fmt_stat(r.get('humidity'),0)}%**")
    with c2:
        rf = float(r.get("park_run_factor") or 100)
        hrf = float(r.get("park_hr_factor") or 100)
        env = "above-average run environment" if rf > 102 else \
              "pitcher-friendly park" if rf < 98 else "neutral park"

        def _factor_words(v):
            if v >= 108:
                return "well above average", "much easier to homer here", "#c0392b"
            if v >= 103:
                return "above average", "homer-friendly", "#e07b39"
            if v <= 92:
                return "well below average", "hard to homer here", "#1a7f5a"
            if v <= 97:
                return "below average", "suppresses home runs", "#2f9e6e"
            return "roughly league-average", "neutral for home runs", "#777"
        hw, hmean, hcol = _factor_words(hrf)

        st.markdown(
            f"- Park: **{r.get('park_name','')}**\n"
            f"- Run factor: **{fmt_stat(rf,0)}** ({env})\n"
            f"- HR factor: **{fmt_stat(hrf,0)}** — "
            f"<span style='color:{hcol};font-weight:600'>{hw}</span> ({hmean})",
            unsafe_allow_html=True)
        st.caption("HR factor: 100 = MLB average park for home runs. Above 100 means "
                   "this park yields **more** HRs than average (dimensions / altitude / air), "
                   "below 100 means **fewer**. It nudges the model's first-inning run "
                   "probability up in homer-friendly parks.")
        uz = r.get("umpire_zone_adj", 0) or 0
        utxt = "larger zone — favors pitcher" if uz > 0 else \
               "smaller zone — favors hitter" if uz < 0 else "league-average zone / not yet assigned"
        st.markdown(f"- Umpire zone: **{utxt}**")


def _profile_row(p: dict, keyset: tuple) -> list:
    return [p.get(k) for k in keyset]


def section_pitcher(r, profiles: pd.DataFrame, last5: pd.DataFrame, side: str):
    name = r.get(f"{side}_pitcher_name") or "TBD"
    pid = r.get(f"{side}_pitcher_id")
    st.subheader(f"{'3' if side=='away' else '4'} · {side.title()} Pitcher — {name}")
    if profiles.empty or pd.isna(pid):
        st.caption("No profile available.")
        return
    pr = profiles[profiles["pitcher_id"] == int(pid)]
    if pr.empty:
        st.caption("No first-inning history for this pitcher yet.")
        return
    p = pr.iloc[0].to_dict()

    cols = ["GS", "NRFI%", "YRFI%", "1st ERA", "OBP", "K%", "BB%", "HardHit%", "Velo", "Primary"]
    def r_(gs, nrfi, yrfi, era, obp, k, bb, hh, velo, prim):
        gs = "—" if gs is None or (isinstance(gs, float) and np.isnan(gs)) else str(int(gs))
        return [gs, fmt_pct(nrfi), fmt_pct(yrfi), fmt_stat(era, 2), fmt_stat(obp),
                fmt_stat(k), fmt_stat(bb), fmt_stat(hh), fmt_stat(velo, 1),
                (prim or "—")]
    tbl = pd.DataFrame([
        r_(p.get("total_starts"), p.get("nrfi_pct"), p.get("yrfi_pct"), p.get("first_inn_era"),
           p.get("first_inn_obp_allowed"), p.get("first_inn_k_rate"), p.get("first_inn_bb_rate"),
           p.get("first_inn_hard_hit_allowed"), p.get("first_inn_avg_velo"), p.get("primary_pitch")),
        r_(p.get("seas_starts"), p.get("seas_nrfi_pct"), p.get("seas_yrfi_pct"),
           p.get("seas_first_inn_era"), p.get("seas_first_inn_obp_allowed"),
           p.get("seas_first_inn_k_rate"), p.get("seas_first_inn_bb_rate"), None, None, None),
        r_(p.get("l5_starts"), p.get("l5_nrfi_pct"), p.get("l5_yrfi_pct"), p.get("l5_first_inn_era"),
           p.get("l5_first_inn_obp_allowed"), None, None, None, p.get("l5_avg_velo"), None),
        r_(p.get("home_starts"), p.get("home_nrfi_pct"),
           100 - p.get("home_nrfi_pct") if pd.notna(p.get("home_nrfi_pct")) else None,
           p.get("home_first_inn_era"), None, None, None, None, None, None),
        r_(p.get("away_starts"), p.get("away_nrfi_pct"),
           100 - p.get("away_nrfi_pct") if pd.notna(p.get("away_nrfi_pct")) else None,
           p.get("away_first_inn_era"), None, None, None, None, None, None),
    ], index=["Career", "This Season", "Last 5", "Home", "Away"], columns=cols)

    career_nrfi = p.get("nrfi_pct") or 0
    def hl(row):
        if row.name == "This Season":
            return ["background-color:#d6e6f5;color:#0a2a4a"] * len(row)
        if row.name == "Last 5":
            try:
                v = float(str(row["NRFI%"]).rstrip("%"))
                good = v >= career_nrfi
                return [f"background-color:{'#d6f0d6;color:#0a3d0a' if good else '#f5dede;color:#5a1010'}"] * len(row)
            except ValueError:
                return [""] * len(row)
        return [""] * len(row)
    st.dataframe(tbl.style.apply(hl, axis=1), use_container_width=True)

    if not last5.empty:
        l5 = last5[last5["pitcher_id"] == int(pid)].sort_values("start_date", ascending=False)
        if not l5.empty:
            d = pd.DataFrame({
                "Date": pd.to_datetime(l5["start_date"]).dt.strftime("%b %-d"),
                "Opp": l5["opponent"],
                "Home/Away": np.where(l5["is_home"], "H", "A"),
                "1st Inn R": l5["inning_1_runs_allowed"].astype(int),
                "1st Inn Pitches": l5["inning_1_pitches"].apply(lambda v: fmt_stat(v, 0)),
                "Result": l5["result"],
                "Velo": l5["velo_on_day"].apply(lambda v: fmt_stat(v, 1)),
            })
            st.dataframe(d, hide_index=True, use_container_width=True)

    arsenal = []
    for b, lbl in (("fb", "Fastball"), ("sl", "Slider"), ("cb", "Curve"), ("ch", "Change/Split")):
        u = p.get(f"first_inn_{b}_usage")
        if u is None or (isinstance(u, float) and np.isnan(u)) or u == 0:
            continue
        arsenal.append({"Pitch": lbl + (" ⭐" if p.get("primary_pitch") == b else ""),
                        "Usage%": fmt_pct((u or 0) * 100, 1),
                        "Whiff%": fmt_pct((p.get(f"first_inn_{b}_whiff") or 0) * 100, 1)})
    if arsenal:
        st.caption("First-inning arsenal")
        st.dataframe(pd.DataFrame(arsenal), hide_index=True, use_container_width=True)


def section_batters(r, side: str):
    home_pitcher = r.get("home_pitcher_name")
    away_pitcher = r.get("away_pitcher_name")
    opp_pitcher = home_pitcher if side == "away" else away_pitcher
    opp_pid = r.get("home_pitcher_id") if side == "away" else r.get("away_pitcher_id")
    team = r[f"{side}_team"]
    st.subheader(f"{'5' if side=='away' else '6'} · {team} Top of Order vs {opp_pitcher or 'TBD'}")
    st.caption("Based on lineup-position frequency this season — lineup not yet confirmed")

    top5 = load_top5_teams()
    bstats = load_batter_stats()
    bvp = load_bvp()
    lsum = load_lineup_summary()

    trow = top5[top5["team_abbr"] == team]
    if trow.empty:
        # alias
        trow = top5[top5["team_abbr"].isin(_alias_variants(team))]
    if trow.empty:
        st.caption("No projected lineup for this team.")
        return
    tr = trow.iloc[0]

    def _i(v):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(v))

    cols = st.columns(5)
    for i in range(1, 6):
        pid = tr.get(f"projected_pos_{i}_player_id")
        pname = tr.get(f"projected_pos_{i}_player_name")
        conf = tr.get(f"projected_pos_{i}_confidence_pct")
        with cols[i - 1]:
            st.markdown(f"**#{i} {pname or 'TBD'}**")
            st.caption(f"bats here {fmt_pct(conf)} of games")
            if pd.isna(pid):
                continue
            pid = int(pid)
            bs = bstats[bstats["player_id"] == pid]
            bs = bs.iloc[0].to_dict() if not bs.empty else {}

            # ── Table 1 (primary): batter vs THIS pitcher ──────────────────
            st.markdown(f"<div style='font-size:11px;font-weight:700;color:#1493ff'>"
                        f"vs {opp_pitcher or 'pitcher'}</div>", unsafe_allow_html=True)
            bv = pd.DataFrame()
            if not bvp.empty and pd.notna(opp_pid):
                bv = bvp[(bvp["batter"] == pid) & (bvp["pitcher"] == int(opp_pid))]
            if not bv.empty and float(bv.iloc[0]["pa"]) >= 1:
                b0 = bv.iloc[0]
                pa = int(b0["pa"])
                vtbl = pd.DataFrame({
                    "": ["PA", "AVG", "OBP", "K%", "BB%", "HR"],
                    "vs SP": [str(pa), fmt_stat(b0.get("avg")), fmt_stat(b0.get("obp")),
                              fmt_stat(b0.get("k_pct")), fmt_stat(b0.get("bb_pct")),
                              str(int(b0.get("hr") or 0))]})
                st.dataframe(vtbl, hide_index=True, use_container_width=True)
                if pa < 5:
                    st.caption(f"⚠️ small sample ({pa} PA)")
            else:
                st.caption("🚫 **No matchup data** — batter has not faced this pitcher.")

            # ── Table 2: recent & season windows (OBP / OPS / R / AB) ──────
            st.markdown("<div style='font-size:11px;font-weight:700;color:#888'>"
                        "recent & season</div>", unsafe_allow_html=True)
            w = pd.DataFrame({
                "": ["AB", "R", "OBP", "OPS"],
                "L7": [_i(bs.get("l7_ab")), _i(bs.get("l7_r")),
                       fmt_stat(bs.get("l7_obp")), fmt_stat(bs.get("l7_ops"))],
                "L30": [_i(bs.get("l30_ab")), _i(bs.get("l30_r")),
                        fmt_stat(bs.get("l30_obp")), fmt_stat(bs.get("l30_ops"))],
                "Seas": [_i(bs.get("seas_ab")), _i(bs.get("seas_r")),
                         fmt_stat(bs.get("seas_obp")), fmt_stat(bs.get("seas_ops"))],
            })
            st.dataframe(w, hide_index=True, use_container_width=True)
            st.caption(
                f"Season: {_i(bs.get('seas_g'))}G · {_i(bs.get('seas_hr'))}HR · "
                f"{_i(bs.get('seas_rbi'))}RBI · {fmt_stat(bs.get('seas_avg'))}/"
                f"{fmt_stat(bs.get('seas_obp'))}/{fmt_stat(bs.get('seas_slg'))}")
            if safe(bs.get("seas_g"), 0) and bs["seas_g"] < 30:
                st.caption("⚠️ Limited season sample")

            # lineup position bar
            ls = lsum[lsum["player_id"] == pid]
            if not ls.empty:
                l0 = ls.iloc[0]
                tg = l0.get("total_games_played") or 1
                bars = []
                for pos in range(1, 10):
                    g = l0.get(f"games_pos_{pos}")
                    g = 0 if g is None or (isinstance(g, float) and np.isnan(g)) else float(g)
                    if g > 0:
                        bars.append(f"#{pos}: {100*g/tg:.0f}%")
                if bars:
                    st.caption("Lineup spots: " + "  ·  ".join(bars))

    if side == "home":
        st.info("🔬 **Coming — per-pitch batted-ball model (separate from the current model).** "
                "The prediction above is driven by the aggregate stats you see here. A future "
                "layer will grade each batter on batted-ball success against *similar pitches* — "
                "grouped by velocity band for fastballs and by pitch type for breaking / offspeed "
                "— and surface it here alongside (not inside) the current model. More detail to come.")


def section_history(r):
    st.subheader("7 · Historical YRFI Context")
    y = load_yrfi_outcomes()
    if y.empty:
        st.caption("No historical data.")
        return
    cur = y[y["game_year"] == y["game_year"].max()]
    league = cur["yrfi"].mean() * 100

    park = r.get("park_name", "")
    yv = y[y["venue_name"] == park]
    park_rate = yv[yv["game_year"] == yv["game_year"].max()]["yrfi"].mean() * 100 if len(yv) else np.nan

    hp = r.get("home_pitcher_id"); ap = r.get("away_pitcher_id")
    hp_rate = _pitcher_yrfi(y, hp, home=True)
    ap_rate = _pitcher_yrfi(y, ap, home=False)

    h2h = y[((y["home_team"] == r["home_team"]) & (y["away_team"] == r["away_team"])) |
            ((y["home_team"] == r["away_team"]) & (y["away_team"] == r["home_team"]))]
    h2h_rate = h2h["yrfi"].mean() * 100 if len(h2h) else np.nan

    model_yrfi = float(r["model_yrfi_prob"]) * 100
    rows = [
        ("Model prediction (this game)", model_yrfi),
        ("League YRFI rate (this season)", league),
        (f"{park} (this season)", park_rate),
        (f"{r.get('home_pitcher_name','home SP')} 1st-inn YRFI allowed", hp_rate),
        (f"{r.get('away_pitcher_name','away SP')} 1st-inn YRFI allowed", ap_rate),
        (f"{r['away_team']} vs {r['home_team']} historically", h2h_rate),
    ]
    for label, val in rows:
        if pd.isna(val):
            st.write(f"{label}: —")
            continue
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<div style='width:340px;font-size:13px'>{label}</div>"
            f"<div style='flex:1;background:#eee;border-radius:3px;height:16px'>"
            f"<div style='width:{min(val,100):.0f}%;background:#d73027;height:16px;border-radius:3px'></div>"
            f"</div><div style='width:48px;font-size:12px'>{val:.0f}%</div></div>",
            unsafe_allow_html=True)


def _pitcher_yrfi(y, pid, home: bool):
    if pd.isna(pid):
        return np.nan
    pid = int(pid)
    cur = y[y["game_year"] == y["game_year"].max()]
    if home:
        g = cur[cur["home_starter_id"] == pid]
        return (g["first_inn_runs_away"] > 0).mean() * 100 if len(g) else np.nan
    g = cur[cur["away_starter_id"] == pid]
    return (g["first_inn_runs_home"] > 0).mean() * 100 if len(g) else np.nan


def _alias_variants(abbr: str) -> list[str]:
    m = {"ARI": ["AZ"], "AZ": ["ARI"], "OAK": ["ATH"], "ATH": ["OAK"],
         "CWS": ["CHW"], "CHW": ["CWS"]}
    return m.get(abbr, [])


# ═══════════════════════════════════════════════════════════════════════════
# Bet tracker
# ═══════════════════════════════════════════════════════════════════════════
def _money(v) -> str:
    v = float(v or 0)
    return f"{'−' if v < 0 else '+'}${abs(v):,.0f}"


def tracker_bar():
    """Top-of-page bankroll tracker + bet log. Shown on every view."""
    df = bt.load_bets()

    # Auto-grade finished games once per browser session (button forces a recheck).
    if not st.session_state.get("_bets_graded"):
        df, n = bt.grade_open_bets(df)
        st.session_state["_bets_graded"] = True
        if n:
            st.toast(f"Graded {n} bet{'s' if n != 1 else ''}")

    # $/unit for NEW bets — defaults to the most recent bet's size; older bets
    # keep the unit size they were logged at.
    default_us = bt.current_unit_size(df)
    unit_size = float(st.session_state.setdefault("unit_size", default_us))
    s = bt.tracker_stats(df)
    col = "#1a9850" if s["net_units"] > 0 else "#d73027" if s["net_units"] < 0 else "#555"

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown(
            f"<div style='font-size:26px;font-weight:800;line-height:1.25'>"
            f"📊 {s['wins']}-{s['losses']} "
            f"<span style='color:{col}'>&nbsp;|&nbsp; {s['net_units']:+.1f}u "
            f"&nbsp;|&nbsp; {_money(s['net_dollars'])}</span></div>"
            f"<div style='font-size:12px;color:#888'>combined record · "
            f"{s['open']} open · ROI {s['roi_pct']:+.1f}%</div>",
            unsafe_allow_html=True)
    with c2:
        st.session_state["unit_size"] = st.number_input(
            "$ / unit (new bets)", min_value=1.0, value=unit_size, step=5.0)
    with c3:
        st.write("")
        if st.button("↻ Grade bets", use_container_width=True):
            _, n = bt.grade_open_bets()
            st.toast(f"Graded {n} bet{'s' if n != 1 else ''}" if n
                     else "No bets ready to grade yet")
            st.rerun()

    by_book = bt.record_by_book(df)
    if not by_book.empty:
        with st.expander("📚 Record by sportsbook"):
            st.dataframe(
                by_book.style.format({"Units": "{:+.2f}", "$": _money}),
                hide_index=True, use_container_width=True)

    with st.expander(f"🧾 Bet log ({len(df)})"):
        render_bet_log(df, ctx="log")
        st.caption("Saved to `data/bet_log.csv` — published to the live site each "
                   "morning by `launch.sh`. On the hosted site, log/edit bets from "
                   "the local app; changes there won't survive the next redeploy.")
    st.divider()


def render_bet_log(df: pd.DataFrame, ctx: str = "log"):
    if df.empty:
        st.caption("No bets logged yet — open a game and use **Track a Bet**.")
        return
    for _, b in df.sort_values("placed_at", ascending=False).iterrows():
        icon = {"won": "✅", "lost": "❌", "open": "🕗"}.get(str(b["status"]), "•")
        line = (f"{icon} **{b['away_team']} @ {b['home_team']}** · {b['side']} "
                f"`{bt.american_str(b['odds'])}` · {float(b['units']):g}u · {b['book']}")
        if str(b["status"]) in ("won", "lost"):
            ru = float(b["result_units"]) if pd.notna(b["result_units"]) else 0.0
            fir = b.get("first_inn_runs")
            line += f" · **{ru:+.2f}u** · 1st-inn runs: {fir}"
        if bt.is_editable(b):
            with st.expander(line):
                _edit_bet_form(b, ctx)
        else:
            st.markdown(line)


def _edit_bet_form(b: pd.Series, ctx: str = "log"):
    real_id = str(b["bet_id"])
    k = f"{ctx}_{real_id}"
    c1, c2, c3, c4 = st.columns([1, 1.3, 1, 1])
    side = c1.selectbox("Side", ["NRFI", "YRFI"],
                        index=0 if str(b["side"]).upper() == "NRFI" else 1,
                        key=f"eb_side_{k}")
    book = c2.selectbox("Book", bt.BOOKS,
                        index=bt.BOOKS.index(b["book"]) if b["book"] in bt.BOOKS
                        else len(bt.BOOKS) - 1, key=f"eb_book_{k}")
    odds = c3.number_input("Odds", value=float(b["odds"]), step=5.0,
                           key=f"eb_odds_{k}")
    units = c4.number_input("Units", min_value=0.0, value=float(b["units"]),
                            step=0.5, key=f"eb_units_{k}")
    s1, s2 = st.columns(2)
    if s1.button("💾 Save", key=f"eb_save_{k}", use_container_width=True):
        bt.update_bet(real_id, side=side.upper(), book=book,
                      odds=float(odds), units=float(units))
        st.toast("Bet updated")
        st.rerun()
    if s2.button("🗑 Delete", key=f"eb_del_{k}", use_container_width=True):
        bt.delete_bet(real_id)
        st.toast("Bet deleted")
        st.rerun()


def section_track_bet(r):
    st.subheader("💰 Track a Bet")
    pk = int(r["game_pk"])
    rec = str(r.get("recommended_bet", "")).split()
    default_side = rec[0] if rec and rec[0] in ("NRFI", "YRFI") else _better_value_side(r)

    c1, c2, c3, c4 = st.columns([1, 1.3, 1, 1])
    side = c1.selectbox("Side", ["NRFI", "YRFI"],
                        index=0 if default_side == "NRFI" else 1, key=f"tb_side_{pk}")
    book = c2.selectbox("Book", bt.BOOKS, key=f"tb_book_{pk}")
    code = {v: k for k, v in bt.BOOK_CODE_TO_NAME.items()}.get(book)
    try:
        prefill = float(r.get(f"{side.lower()}_odds_{code}")) if code else 100.0
    except (TypeError, ValueError):
        prefill = 100.0
    # key varies with side/book so the field re-prefills when either changes
    odds = c3.number_input("Odds (American)", value=prefill, step=5.0,
                           key=f"tb_odds_{pk}_{side}_{book}")
    units = c4.number_input("Units staked", min_value=0.0, value=1.0, step=0.5,
                            key=f"tb_units_{pk}")
    us = float(st.session_state.get("unit_size", bt.DEFAULT_UNIT_SIZE))
    st.caption(f"Stake is in **units** · 1 unit = \\${us:g} (set at the top) · "
               f"this bet risks {units:g}u = \\${units * us:,.0f}")
    if st.button("➕ Log this bet", key=f"tb_log_{pk}", type="primary"):
        bt.add_bet(game_pk=pk, game_date=str(r.get("game_date", TODAY)),
                   away_team=r["away_team"], home_team=r["home_team"],
                   side=side, book=book, odds=float(odds), units=float(units),
                   unit_size=us)
        st.session_state["_bets_graded"] = False
        st.toast(f"Logged {side} {bt.american_str(odds)} · {units:g}u")
        st.rerun()

    mine = bt.load_bets()
    if not mine.empty:
        mine = mine[pd.to_numeric(mine["game_pk"], errors="coerce") == pk]
    if not mine.empty:
        st.caption("Bets on this game")
        render_bet_log(mine, ctx=f"game{pk}")


# ═══════════════════════════════════════════════════════════════════════════
def main():
    pred = load_predictions()
    meta = load_model_meta()

    tracker_bar()

    if not pred.empty:
        weights = render_weight_controls(meta)
        pred = apply_weights(pred, weights)

    if "game" in st.session_state and not pred.empty:
        page_detail(int(st.session_state["game"]), pred)
    else:
        page_landing(pred)

    if meta:
        with st.sidebar:
            st.markdown("### Model")
            for m in meta.get("metrics", []):
                if m["set"] == "test_2024":
                    st.metric("Test AUC (2024)", m["auc"])
                    st.caption(f"Brier {m['brier']} · ECE {m['ece']} · "
                               f"acc {m['accuracy']}")
            st.caption(f"Trained {meta.get('trained_at','?')}")
            with st.expander("Caveats"):
                for c in meta.get("caveats", []):
                    st.caption("• " + c)
                st.caption("• ⚡ instant-betslip links work for Caesars & BetMGM only "
                           "(and are NJ-scoped US URLs — outside the US they may not "
                           "resolve to your account). DraftKings gives no deeplink for "
                           "this market; FanDuel's feed returns a generic selection id "
                           "that FanDuel rejects — both open the book's MLB lobby (↗).")


if __name__ == "__main__":
    main()
