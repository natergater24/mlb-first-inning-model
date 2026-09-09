#!/usr/bin/env python3
"""
bet_tracker.py — YRFI/NRFI bet logging, MLB-API grading, bankroll tracker.

Persistence is LOCAL-FIRST: data/bet_log.csv lives in the repo and is read /
written on the local filesystem. launch.sh commits + pushes it every morning
so the deployed (Streamlit Cloud) app shows the same history. All storage goes
through load_bets() / save_bets(), so a GitHub-API backend can be swapped in
later without touching app.py.

Grading uses the free, keyless MLB Stats API linescore — the same source
src/10_build_yrfi_dataset.py uses for ground truth.

Row schema (one row per bet; multiple bets per game / per book allowed):
    bet_id         12-hex id
    placed_at      ISO-8601 UTC
    game_pk        MLB game id
    game_date      YYYY-MM-DD
    game_start     ISO-8601 UTC first-pitch (from the MLB API) or blank
    away_team      abbr
    home_team      abbr
    side           NRFI | YRFI
    book           DraftKings | FanDuel | BetMGM | Caesars | Other
    odds           American (e.g. 100, -120)
    units          float — stake in units
    unit_size      dollars per unit at the time this bet was placed (frozen, so
                   older bets keep their value when the current unit size changes)
    status         open | won | lost
    first_inn_runs total 1st-inning runs at grade time
    graded_at      ISO-8601 UTC
    result_units   +units*profit on a win, -units on a loss
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
BET_LOG = ROOT / "data" / "bet_log.csv"

COLUMNS = [
    "bet_id", "placed_at", "game_pk", "game_date", "game_start",
    "away_team", "home_team", "side", "book", "odds", "units",
    "unit_size", "status", "first_inn_runs", "graded_at", "result_units",
]

DEFAULT_UNIT_SIZE = 25.0  # dollars per unit

BOOKS = ["DraftKings", "FanDuel", "BetMGM", "Caesars", "Other"]
# app.py BOOK_META short code -> full name used here
BOOK_CODE_TO_NAME = {"DK": "DraftKings", "FD": "FanDuel",
                     "MGM": "BetMGM", "CZR": "Caesars"}

SCHED_URL = "https://statsapi.mlb.com/api/v1/schedule"
LINESCORE_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/linescore"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── persistence ───────────────────────────────────────────────────────────
def load_bets() -> pd.DataFrame:
    """Return the full bet log (empty, well-formed frame if the file is absent)."""
    if BET_LOG.exists():
        df = pd.read_csv(BET_LOG, dtype={"bet_id": str})
    else:
        df = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    # Rows written before per-bet unit size existed keep the default.
    df["unit_size"] = pd.to_numeric(df["unit_size"], errors="coerce").fillna(DEFAULT_UNIT_SIZE)
    return df[COLUMNS].copy()


def current_unit_size(df: pd.DataFrame | None = None) -> float:
    """The unit size of the most recently placed bet (fallback: the default)."""
    if df is None:
        df = load_bets()
    if df.empty:
        return DEFAULT_UNIT_SIZE
    latest = df.sort_values("placed_at").iloc[-1]["unit_size"]
    try:
        return float(latest)
    except (TypeError, ValueError):
        return DEFAULT_UNIT_SIZE


def save_bets(df: pd.DataFrame) -> None:
    BET_LOG.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(BET_LOG, index=False)


# ── odds math ─────────────────────────────────────────────────────────────
def profit_multiple(american) -> float:
    """Profit in units per unit staked for a winning bet at these American odds."""
    a = float(american)
    return a / 100.0 if a > 0 else 100.0 / abs(a)


def american_str(v) -> str:
    try:
        a = int(round(float(v)))
    except (TypeError, ValueError):
        return "—"
    return f"+{a}" if a > 0 else str(a)


# ── MLB API helpers ───────────────────────────────────────────────────────
def game_start_iso(game_pk) -> str | None:
    """First-pitch time (ISO-8601 UTC) from the MLB schedule endpoint."""
    try:
        d = requests.get(SCHED_URL,
                         params={"sportId": 1, "gamePk": int(game_pk)},
                         timeout=15).json()
        return d["dates"][0]["games"][0]["gameDate"]
    except Exception:
        return None


def first_inning_runs(game_pk) -> int | None:
    """
    Total runs in the 1st inning, or None if the 1st inning is not yet complete
    (so an open NRFI bet is never graded early).
    """
    try:
        d = requests.get(LINESCORE_URL.format(pk=int(game_pk)), timeout=15).json()
    except Exception:
        return None
    innings = d.get("innings") or []
    if not innings:
        return None
    cur = d.get("currentInning") or 0
    state = str(d.get("inningState") or "")
    # The 1st inning is complete once play has moved past it.
    first_done = cur > 1 or (cur == 1 and state == "End")
    i1 = innings[0]
    total = int(i1.get("home", {}).get("runs") or 0) + \
        int(i1.get("away", {}).get("runs") or 0)
    if not first_done:
        # Grade early only if a run already scored (the outcome is settled).
        return total if total >= 1 else None
    return total


# ── add / edit / delete ───────────────────────────────────────────────────
def add_bet(*, game_pk, game_date, away_team, home_team, side, book,
            odds, units, unit_size=DEFAULT_UNIT_SIZE, game_start=None) -> str:
    df = load_bets()
    bet_id = uuid.uuid4().hex[:12]
    if game_start is None:
        game_start = game_start_iso(game_pk)
    row = {
        "bet_id": bet_id, "placed_at": _now_iso(),
        "game_pk": int(game_pk), "game_date": str(game_date),
        "game_start": game_start or "",
        "away_team": away_team, "home_team": home_team,
        "side": str(side).upper(), "book": book,
        "odds": float(odds), "units": float(units),
        "unit_size": float(unit_size),
        "status": "open", "first_inn_runs": pd.NA,
        "graded_at": pd.NA, "result_units": pd.NA,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_bets(df)
    return bet_id


def update_bet(bet_id, **fields) -> None:
    df = load_bets()
    m = df["bet_id"] == bet_id
    if not m.any():
        return
    for k, v in fields.items():
        if k in df.columns:
            df.loc[m, k] = v
    save_bets(df)


def delete_bet(bet_id) -> None:
    df = load_bets()
    save_bets(df[df["bet_id"] != bet_id])


def is_editable(row) -> bool:
    """A bet may be edited while it is open and its game has not started."""
    if str(row.get("status")) != "open":
        return False
    gs = row.get("game_start")
    if gs is None or str(gs).strip() in ("", "<NA>", "nan", "NaT"):
        return True
    try:
        start = datetime.fromisoformat(str(gs).replace("Z", "+00:00"))
        return datetime.now(timezone.utc) < start
    except ValueError:
        return True


# ── grading ───────────────────────────────────────────────────────────────
def grade_open_bets(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    """Grade every open bet whose 1st inning is complete. Returns (df, n_graded)."""
    if df is None:
        df = load_bets()
    open_idx = df.index[df["status"] == "open"]
    graded = 0
    runs_cache: dict[int, int | None] = {}
    for idx in open_idx:
        pk = int(df.at[idx, "game_pk"])
        if pk not in runs_cache:
            runs_cache[pk] = first_inning_runs(pk)
        runs = runs_cache[pk]
        if runs is None:
            continue
        side = str(df.at[idx, "side"]).upper()
        won = (runs == 0) if side == "NRFI" else (runs >= 1)
        units = float(df.at[idx, "units"])
        odds = float(df.at[idx, "odds"])
        df.at[idx, "status"] = "won" if won else "lost"
        df.at[idx, "first_inn_runs"] = runs
        df.at[idx, "graded_at"] = _now_iso()
        df.at[idx, "result_units"] = (round(units * profit_multiple(odds), 4)
                                      if won else -units)
        graded += 1
    if graded:
        save_bets(df)
    return df, graded


# ── tracker ───────────────────────────────────────────────────────────────
def _net_dollars(rows: pd.DataFrame) -> float:
    """Sum result_units * that bet's own unit_size (so past bets keep the unit
    size they were placed at even after the current unit size changes)."""
    ru = pd.to_numeric(rows["result_units"], errors="coerce").fillna(0.0)
    us = pd.to_numeric(rows["unit_size"], errors="coerce").fillna(DEFAULT_UNIT_SIZE)
    return float((ru * us).sum())


def tracker_stats(df: pd.DataFrame) -> dict:
    graded = df[df["status"].isin(["won", "lost"])]
    wins = int((graded["status"] == "won").sum())
    losses = int((graded["status"] == "lost").sum())
    net_units = float(pd.to_numeric(graded["result_units"], errors="coerce").sum())
    staked = float(pd.to_numeric(graded["units"], errors="coerce").sum())
    return {
        "wins": wins,
        "losses": losses,
        "open": int((df["status"] == "open").sum()),
        "net_units": round(net_units, 2),
        "net_dollars": round(_net_dollars(graded), 2),
        "roi_pct": round(100 * net_units / staked, 1) if staked else 0.0,
    }


def record_by_book(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sportsbook W-L, net units and net dollars (dollars use each bet's own
    unit size). One row per book that has at least one bet."""
    if df.empty:
        return pd.DataFrame(columns=["Book", "W-L", "Units", "$", "Open"])
    rows = []
    for book, g in df.groupby("book"):
        graded = g[g["status"].isin(["won", "lost"])]
        rows.append({
            "Book": book,
            "W-L": f"{int((graded['status'] == 'won').sum())}-"
                   f"{int((graded['status'] == 'lost').sum())}",
            "Units": round(float(pd.to_numeric(graded["result_units"],
                                               errors="coerce").sum()), 2),
            "$": round(_net_dollars(graded), 2),
            "Open": int((g["status"] == "open").sum()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("$", ascending=False).reset_index(drop=True)
