import numpy as np
import pandas as pd

from scripts.walk_forward_backtest import window_boundaries


def test_window_boundaries_span_full_timestamp_range():
    timestamps = pd.date_range("2022-01-01", periods=1_000, freq="h").to_numpy()

    boundaries = window_boundaries(timestamps)

    assert len(boundaries) == 4
    # The last window's test cutoff is open-ended and must reach near the end
    # of the range -- not stop at the 14th unique timestamp.
    last_train_cutoff = boundaries[-1][0]
    assert last_train_cutoff > timestamps[len(timestamps) // 2]


def test_window_boundaries_are_expanding_and_ordered():
    timestamps = pd.date_range("2022-01-01", periods=1_000, freq="h").to_numpy()

    boundaries = window_boundaries(timestamps)

    train_cutoffs = [b[0] for b in boundaries]
    assert train_cutoffs == sorted(train_cutoffs)
    assert len(set(train_cutoffs)) == len(train_cutoffs)

    for train_cutoff, calibration_cutoff, test_cutoff in boundaries:
        assert train_cutoff < calibration_cutoff
        if test_cutoff is not None:
            assert calibration_cutoff < test_cutoff

    assert boundaries[-1][2] is None


def test_window_boundaries_requires_minimum_timestamps():
    import pytest

    with pytest.raises(ValueError):
        window_boundaries(np.arange(5))
