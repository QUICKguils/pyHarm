import numpy as np
import pytest
from pytest import MonkeyPatch

from pyHarm.Correctors.CorrectorArcLength import Corrector_arc_length
from pyHarm.Solver import SystemSolution

SOLSTART = np.array([0.0, 1.0])
SOL = SystemSolution(SOLSTART)
SOLPREC = SystemSolution(SOLSTART)
SOLPREC.x_pred = np.array([0.0, 0.0])
SOLX = np.array([1.0, 0.0])
SOL.precedent_solution = SOLPREC


@pytest.mark.all
@pytest.mark.unit
def test_CorrectorArcLength() -> None:
    C = Corrector_arc_length()
    Rcont = C.ClosureEquation(SOLX, SOL, [SOL])
    assert isinstance(Rcont, float)
    assert Rcont == 1.0
    dRdx, dRdom = C.ClosureJacobian(SOLX, SOL, [SOL])
    assert dRdx == 2 * (SOLX - SOLPREC.x)[:-1].T
    assert dRdom == 2 * (SOLX - SOLPREC.x)[-1]
