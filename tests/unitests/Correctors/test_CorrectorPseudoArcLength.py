from pyHarm.Solver import SystemSolution
import numpy as np 
from pytest import MonkeyPatch
from pyHarm.Correctors.CorrectorPseudoArcLength import Corrector_pseudo_arc_length
import pytest

SOLSTART = np.array([0.,1.])
SOL = SystemSolution(SOLSTART)
SOLPREC = SystemSolution(SOLSTART)
SOLPREC.x_pred = np.array([0.,0.])
SOLX = np.array([1., 0.])
SOL.precedent_solution = SOLPREC

@pytest.mark.all
@pytest.mark.unit
def test_CorrectorPseudoArcLength() -> None :
    C = Corrector_pseudo_arc_length()
    Rcont = C.ClosureEquation(SOLX, SOL, [SOL])
    assert isinstance(Rcont, float)
    assert Rcont == 0.
    dRdx,dRdom = C.ClosureJacobian(SOLX, SOL, [SOL])
    assert dRdx == -SOLSTART[:-1]
    assert dRdom == -SOLSTART[-1]

