import pytest 
from pytest import MonkeyPatch
from pyHarm.Predictors.PredictorSecant import PredictorSecant
from pyHarm.Solver import FirstSolution, SystemSolution
import numpy as np

SIGN_DS = 1
DS = 1.

SOL_FIRSTA = FirstSolution(np.array([0.,1.]))
SOL_FIRSTA.flag_accepted = True
SOL_FIRSTA.flag_J = True
SOL_FIRSTA.flag_J_f = True
SOL_FIRSTA.J_f = np.eye(2)

SOL_ACCEPT = SystemSolution(np.array([1.,2.]))
SOL_ACCEPT.flag_accepted = True
SOL_ACCEPT.flag_J = True
SOL_ACCEPT.flag_J_f = True
SOL_ACCEPT.J_f = np.eye(2)
SOL_ACCEPT.precedent_solution = SOL_FIRSTA

@pytest.mark.all
@pytest.mark.unit
def test_PredictorSecant_factory_keyword()->None:
    assert PredictorSecant.factory_keyword == 'secant' 


@pytest.mark.all
@pytest.mark.unit
def test_PredictorSecant_predict() -> None :
    P = PredictorSecant(SIGN_DS,norm='om',bifurcation_detection=False)
    SOLLIST = [SOL_FIRSTA]
    xpred, lstpt, sign_ds = P.predict(SOLLIST, DS, k_imposed=0)
    assert xpred[-1] == 2.
    assert np.allclose(xpred[:-1],SOL_FIRSTA.x[:-1])
    assert lstpt == SOL_FIRSTA
    assert sign_ds == SIGN_DS
    SOLLIST = [SOL_FIRSTA,SOL_ACCEPT]
    xpred, lstpt, sign_ds = P.predict(SOLLIST, DS, k_imposed=None)
    assert lstpt == SOL_ACCEPT
    assert xpred[-1] == 3.
    assert xpred[0] == 2.