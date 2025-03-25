import pytest
from pytest import MonkeyPatch

from pyHarm.StepSizeRules.ABCStepSizeRule import ABCStepSizeRule

BOUNDS = [1e-1, 1e2]
DS_RETURNED = 1.0


@pytest.fixture
def mock_abcstepsizerule(monkeypatch: MonkeyPatch) -> ABCStepSizeRule:
    monkeypatch.setattr(ABCStepSizeRule, "__abstractmethods__", set())
    monkeypatch.setattr(ABCStepSizeRule, "factory_keyword", "mock_abcstepsizerule")
    monkeypatch.setattr(ABCStepSizeRule, "getStepSize", lambda self, ds, sollist: DS_RETURNED)
    SSR = ABCStepSizeRule(BOUNDS)
    return SSR


@pytest.mark.all
@pytest.mark.unit
def test_abcstepsizerule(mock_abcstepsizerule: ABCStepSizeRule) -> None:
    assert mock_abcstepsizerule.ds_min == BOUNDS[0]
    assert mock_abcstepsizerule.ds_max == BOUNDS[1]
    ## test ProjectInBounds method :
    # case 1 : ds < ds_min :
    assert mock_abcstepsizerule.ProjectInBounds(BOUNDS[0] - 1.0) == BOUNDS[0]
    # case 2 : ds > ds_max :
    assert mock_abcstepsizerule.ProjectInBounds(BOUNDS[1] + 1.0) == BOUNDS[1]
    # case 3 : ds_min < ds < ds_max :
    assert (
        mock_abcstepsizerule.ProjectInBounds((BOUNDS[1] - BOUNDS[0]) / 2.0)
        == (BOUNDS[1] - BOUNDS[0]) / 2.0
    )
