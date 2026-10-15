import numpy as np
import pytest

from src.labels import IGNORE
from src.metrics import evaluate


def test_hand_computed_example():
    strata = np.array(["A"] * 4 + ["B"] * 4)
    y_sw = np.array([1, 0, 0, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.4, 0.8, 0.1, 0.2, 0.3])
    y_dur = np.array([0, IGNORE, IGNORE, 2, 1, IGNORE, IGNORE, IGNORE])
    dur_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    table, overall, cm = evaluate(strata, y_sw, p, y_dur, dur_pred)

    # A: predicted switches at 0 and 2 -> TP=1, FP=1, FN=1 -> P=R=F1=0.5
    assert table.loc["A", "sw_f1"] == pytest.approx(0.5)
    # B: one switch, predicted correctly, no false alarms -> F1 = 1
    assert table.loc["B", "sw_f1"] == pytest.approx(1.0)
    assert overall["macro_sw_f1"] == pytest.approx(0.75)
    assert overall["sigma_universality_sw_f1"] == pytest.approx(0.25)
    # duration scored only at true switches: A gets 1/2 right, B 1/1
    assert table.loc["A", "dur_accuracy"] == pytest.approx(0.5)
    assert table.loc["B", "dur_accuracy"] == pytest.approx(1.0)
    assert cm.values.sum() == 3


def test_rejects_ignored_positions():
    with pytest.raises(AssertionError):
        evaluate(np.array(["A"]), np.array([IGNORE]), np.array([0.5]), np.array([IGNORE]), np.array([0]))
