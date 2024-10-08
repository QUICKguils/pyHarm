from pyHarm.Solver import SystemSolution
import numpy as np 
from pytest import MonkeyPatch
from pyHarm.Correctors.CorrectorNoContinuation import Corrector_no_continuation
import pytest

SOLSTART = np.array([0.,0.])
SOL = SystemSolution(SOLSTART)
SOLX = np.array([0., 1.])

@pytest.mark.all
@pytest.mark.unit
def test_CorrectorNoContinuation() -> None :
    C = Corrector_no_continuation()
    assert C.ClosureEquation(SOLX, SOL, [SOL]) == (SOLX - SOLSTART)[-1]
    dRdx,dRdom = C.ClosureJacobian(SOLX, SOL, [SOL])
    assert dRdx.size == SOLX.size - 1
    assert dRdom == 1.

