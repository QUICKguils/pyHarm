import numpy as np
import pytest
from pytest import MonkeyPatch

from pyHarm.Correctors.CorrectorPseudoArcLength import Corrector_pseudo_arc_length
from pyHarm.Solver import SystemSolution

SOLSTART = np.array([0.0, 1.0])
SOL = SystemSolution(SOLSTART)
SOLPREC = SystemSolution(SOLSTART)
SOLPREC.x_pred = np.array([0.0, 0.0])
SOLX = np.array([1.0, 0.0])
SOL.precedent_solution = SOLPREC


@pytest.mark.all
@pytest.mark.unit
def test_CorrectorPseudoArcLength() -> None:
    C = Corrector_pseudo_arc_length()
    Rcont = C.ClosureEquation(SOLX, SOL, [SOL])
    assert isinstance(Rcont, float)
    assert Rcont == 0.0
    dRdx, dRdom = C.ClosureJacobian(SOLX, SOL, [SOL])
    assert dRdx == -SOLSTART[:-1]
    assert dRdom == -SOLSTART[-1]
