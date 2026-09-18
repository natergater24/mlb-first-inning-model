import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yrfi_features import _shrink

def test_shrink_pulls_thin_sample_toward_prior():
    # Mason Barnett's real 2026-09-18 case: 25.0% seas_nrfi_pct over 4 starts
    result = _shrink(rate=25.0, n=4, prior=70.0, k=12)
    assert result == round((4 * 25.0 + 12 * 70.0) / (4 + 12), 4)
    assert 55 < result < 62  # pulled well above the raw 25%, not all the way to 70

def test_shrink_barely_moves_large_sample():
    # An established pitcher: 68% over 30 starts should stay close to 68
    result = _shrink(rate=68.0, n=30, prior=70.0, k=12)
    assert 67.0 < result < 69.0

def test_shrink_returns_exact_prior_for_debut_pitcher():
    # No profile row at all -> n is None/0 -> shrunk value IS the league prior
    assert _shrink(rate=None, n=None, prior=70.0, k=12) == 70.0
    assert _shrink(rate=None, n=0, prior=70.0, k=12) == 70.0

def test_shrink_never_returns_nan():
    import math
    result = _shrink(rate=float("nan"), n=float("nan"), prior=70.0, k=12)
    assert not math.isnan(result)
