import numpy as np
import pytest
from pytest import MonkeyPatch

from pyHarm.Solver import FirstSolution, SystemSolution
from pyHarm.StopCriterion.StopCriterionBounds import StopCriterionBounds

BOUNDS = [0.0, 1.0]
DS_MIN = 3.0


@pytest.mark.all
@pytest.mark.unit
def test_StopCriterionBounds_getStopCriterionStatus() -> None:
    SC = StopCriterionBounds(BOUNDS, DS_MIN)
    assert SC.factory_keyword == "bounds"
    # case 1 -> consecutive ds_min
    sol = FirstSolution(np.array([0.5]))
    sol.ds = 3.0
    sollist = [sol] * 2
    assert SC.getStopCriterionStatus(sol, sollist) == True
    # case 2 -> FirstSolution out of bounds
    sol = FirstSolution(np.array([-1.0]))
    sol.ds = 5.0
    sollist = []
    assert SC.getStopCriterionStatus(sol, sollist) == False
    # case 2 -> SystemSolution out of bounds
    sol = SystemSolution(np.array([-1.0]))
    sol.ds = 5.0
    sollist = []
    assert SC.getStopCriterionStatus(sol, sollist) == True
