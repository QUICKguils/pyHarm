import pytest 
from pytest import MonkeyPatch
from pyHarm.StopCriterion.StopCriterionBoundsOrSolNumber import StopCriterionBoundsOrSolNumber
from pyHarm.Solver import SystemSolution,FirstSolution
import numpy as np

BOUNDS = [0.,1.]
DS_MIN = 3.

@pytest.mark.all
@pytest.mark.unit
def test_StopCriterionBounds_getStopCriterionStatus()->None:
    assert StopCriterionBoundsOrSolNumber.factory_keyword == 'solnumber'
    assert StopCriterionBoundsOrSolNumber.default == {"max_solutions" : 100}
    SC = StopCriterionBoundsOrSolNumber(BOUNDS,DS_MIN,**{"max_solutions" : 3})
    # case 1 -> metaclass True
    sol = SystemSolution(np.array([.5]))
    sol.ds = DS_MIN
    sollist = [sol]*1
    assert SC.getStopCriterionStatus(sol,sollist) == False
    sollist += [sol]*1
    assert SC.getStopCriterionStatus(sol,sollist) == True
    # sol.ds = 3.5
    sol.ds = DS_MIN + .1
    sollist += [sol]
    assert SC.getStopCriterionStatus(sol,sollist) == True