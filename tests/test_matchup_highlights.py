import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from matchup_highlights import notable_bvp_for_game


def _game(game_pk=1, away="TB", home="NYY", away_pid=100, home_pid=200):
    return pd.Series({
        "game_pk": game_pk, "away_team": away, "home_team": home,
        "away_pitcher_id": away_pid, "away_pitcher_name": "Away Pitcher",
        "home_pitcher_id": home_pid, "home_pitcher_name": "Home Pitcher",
    })


def _top5(team, batters):
    """batters: list of (player_id, player_name), up to 5."""
    row = {"team_abbr": team}
    for i, (pid, name) in enumerate(batters, start=1):
        row[f"projected_pos_{i}_player_id"] = pid
        row[f"projected_pos_{i}_player_name"] = name
    return pd.DataFrame([row])


def _bvp(rows):
    """rows: list of dicts with batter, pitcher, pa, obp, avg, hr."""
    return pd.DataFrame(rows)


def test_finds_hot_batter_vs_home_pitcher():
    game = _game(home_pid=200)
    top5 = pd.concat([_top5("TB", [(1, "Hot Hitter")]), _top5("NYY", [])], ignore_index=True)
    bvp = _bvp([{"batter": 1, "pitcher": 200, "pa": 20, "obp": 0.500, "avg": 0.400, "hr": 3}])
    out = notable_bvp_for_game(game, top5, bvp)
    assert len(out) == 1
    assert out[0]["batter_name"] == "Hot Hitter"
    assert out[0]["pitcher_name"] == "Home Pitcher"
    assert out[0]["team"] == "TB"
    assert out[0]["hot"] is True


def test_finds_cold_batter_vs_away_pitcher():
    game = _game(away_pid=100)
    top5 = pd.concat([_top5("TB", []), _top5("NYY", [(2, "Cold Hitter")])], ignore_index=True)
    bvp = _bvp([{"batter": 2, "pitcher": 100, "pa": 18, "obp": 0.150, "avg": 0.100, "hr": 0}])
    out = notable_bvp_for_game(game, top5, bvp)
    assert len(out) == 1
    assert out[0]["hot"] is False


def test_includes_matchups_from_both_sides_of_the_same_game():
    game = _game(away_pid=100, home_pid=200)
    top5 = pd.concat([
        _top5("TB", [(1, "Away Hitter")]),
        _top5("NYY", [(2, "Home Hitter")]),
    ], ignore_index=True)
    bvp = _bvp([
        {"batter": 1, "pitcher": 200, "pa": 20, "obp": 0.500, "avg": 0.400, "hr": 1},
        {"batter": 2, "pitcher": 100, "pa": 18, "obp": 0.150, "avg": 0.100, "hr": 0},
    ])
    out = notable_bvp_for_game(game, top5, bvp)
    assert len(out) == 2
    assert {m["batter_name"] for m in out} == {"Away Hitter", "Home Hitter"}


def test_excludes_batter_below_pa_threshold():
    game = _game()
    top5 = _top5("TB", [(1, "Thin Sample")])
    bvp = _bvp([{"batter": 1, "pitcher": 200, "pa": 5, "obp": 0.600, "avg": 0.500, "hr": 2}])
    out = notable_bvp_for_game(game, top5, bvp)
    assert out == []


def test_excludes_batter_with_unremarkable_obp():
    game = _game()
    top5 = _top5("TB", [(1, "Average Joe")])
    bvp = _bvp([{"batter": 1, "pitcher": 200, "pa": 20, "obp": 0.310, "avg": 0.270, "hr": 1}])
    out = notable_bvp_for_game(game, top5, bvp)
    assert out == []


def test_excludes_batter_with_no_bvp_history():
    game = _game()
    top5 = _top5("TB", [(1, "No History")])
    bvp = _bvp([{"batter": 999, "pitcher": 200, "pa": 20, "obp": 0.500, "avg": 0.400, "hr": 1}])
    out = notable_bvp_for_game(game, top5, bvp)
    assert out == []


def test_ranked_by_pa_descending():
    game = _game()
    top5 = _top5("TB", [(1, "Big Sample"), (2, "Small Sample")])
    bvp = _bvp([
        {"batter": 1, "pitcher": 200, "pa": 20, "obp": 0.450, "avg": 0.400, "hr": 1},
        {"batter": 2, "pitcher": 200, "pa": 40, "obp": 0.450, "avg": 0.400, "hr": 1},
    ])
    out = notable_bvp_for_game(game, top5, bvp)
    assert [m["batter_name"] for m in out] == ["Small Sample", "Big Sample"]


def test_respects_n_limit():
    game = _game()
    batters = [(i, f"Batter {i}") for i in range(1, 6)]
    top5 = _top5("TB", batters)
    bvp = _bvp([{"batter": i, "pitcher": 200, "pa": 15 + i, "obp": 0.500, "avg": 0.400, "hr": 1}
               for i in range(1, 6)])
    out = notable_bvp_for_game(game, top5, bvp, n=2)
    assert len(out) == 2


def test_empty_inputs_return_empty_list():
    assert notable_bvp_for_game(_game(), pd.DataFrame(), pd.DataFrame()) == []
