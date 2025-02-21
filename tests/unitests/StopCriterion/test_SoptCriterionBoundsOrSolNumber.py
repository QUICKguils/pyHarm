import numpy as np
import pytest
from pytest import MonkeyPatch

from pyHarm.Solver import FirstSolution, SystemSolution
from pyHarm.StopCriterion.StopCriterionBoundsOrSolNumber import (
    StopCriterionBoundsOrSolNumber,
)

BOUNDS = [0.0, 1.0]
DS_MIN = 3.0


@pytest.mark.all
@pytest.mark.unit
def test_StopCriterionBounds_getStopCriterionStatus() -> None:
    assert StopCriterionBoundsOrSolNumber.factory_keyword == "solnumber"
    assert StopCriterionBoundsOrSolNumber.default == {"max_solutions": 100}
    SC = StopCriterionBoundsOrSolNumber(BOUNDS, DS_MIN, **{"max_solutions": 3})
    # case 1 -> metaclass True
    sol = SystemSolution(np.array([0.5]))
    sol.ds = DS_MIN
    sollist = [sol] * 1
    assert SC.getStopCriterionStatus(sol, sollist) == False
    sollist += [sol] * 1
    assert SC.getStopCriterionStatus(sol, sollist) == True
    # sol.ds = 3.5
    sol.ds = DS_MIN + 0.1
    sollist += [sol]
    assert SC.getStopCriterionStatus(sol, sollist) == True
