import pytest
from pytest import MonkeyPatch

from pyHarm.StopCriterion.ABCStopCriterion import ABCStopCriterion


@pytest.fixture
def mock_abcstopcriterion(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ABCStopCriterion, "__abstractmethods__", set())
    monkeypatch.setattr(ABCStopCriterion, "factory_keyword", lambda self: "monkey_abcstopcriterion")
    monkeypatch.setattr(ABCStopCriterion, "getStopCriterionStatus", lambda self: True)


@pytest.mark.all
@pytest.mark.unit
def test_abcstopcriterion(mock_abcstopcriterion) -> None:
    bounds = [0.0, 1.0]
    ds_min = 3.0
    SC = ABCStopCriterion(bounds, ds_min)
    assert SC.ds_min == ds_min
    assert SC.puls_inf == bounds[0]
    assert SC.puls_sup == bounds[1]
    assert SC.getStopCriterionStatus() == True
