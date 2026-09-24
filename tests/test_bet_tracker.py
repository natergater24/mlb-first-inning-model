import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from bet_tracker import daily_net_by_book, COLUMNS


def _bet(game_date, book, status, result_units, unit_size=25.0):
    return {c: pd.NA for c in COLUMNS} | {
        "game_date": game_date, "book": book, "status": status,
        "result_units": result_units, "unit_size": unit_size,
    }


def test_daily_net_by_book_sums_per_day_per_book():
    df = pd.DataFrame([
        _bet("2026-09-20", "DraftKings", "won", 1.0, 25.0),   # +25
        _bet("2026-09-20", "DraftKings", "lost", -1.0, 25.0), # -25
        _bet("2026-09-20", "FanDuel", "won", 2.0, 10.0),      # +20
        _bet("2026-09-21", "DraftKings", "won", 0.5, 25.0),   # +12.5
    ])
    out = daily_net_by_book(df)
    assert out.loc["2026-09-20", "DraftKings"] == 0.0
    assert out.loc["2026-09-20", "FanDuel"] == 20.0
    assert out.loc["2026-09-21", "DraftKings"] == 12.5


def test_daily_net_by_book_all_column_sums_across_books():
    df = pd.DataFrame([
        _bet("2026-09-20", "DraftKings", "won", 1.0, 25.0),
        _bet("2026-09-20", "FanDuel", "lost", -2.0, 10.0),
    ])
    out = daily_net_by_book(df)
    assert out.loc["2026-09-20", "All"] == 25.0 - 20.0


def test_daily_net_by_book_ignores_open_bets():
    df = pd.DataFrame([
        _bet("2026-09-20", "DraftKings", "open", pd.NA, 25.0),
    ])
    out = daily_net_by_book(df)
    assert out.empty


def test_daily_net_by_book_fills_missing_book_day_with_zero():
    df = pd.DataFrame([
        _bet("2026-09-20", "DraftKings", "won", 1.0, 25.0),
        _bet("2026-09-21", "FanDuel", "won", 1.0, 25.0),
    ])
    out = daily_net_by_book(df)
    assert out.loc["2026-09-20", "FanDuel"] == 0.0
    assert out.loc["2026-09-21", "DraftKings"] == 0.0


def test_daily_net_by_book_empty_input():
    out = daily_net_by_book(pd.DataFrame(columns=COLUMNS))
    assert out.empty


def test_daily_net_by_book_sorted_by_date():
    df = pd.DataFrame([
        _bet("2026-09-22", "DraftKings", "won", 1.0, 25.0),
        _bet("2026-09-20", "DraftKings", "won", 1.0, 25.0),
    ])
    out = daily_net_by_book(df)
    assert list(out.index) == ["2026-09-20", "2026-09-22"]
