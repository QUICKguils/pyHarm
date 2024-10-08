import pytest 
from pytest import MonkeyPatch
from pyHarm.Predictors.PredictorTangent import PredictorTangent
from pyHarm.Solver import FirstSolution, SystemSolution
import numpy as np

SIGN_DS = 1
DS = 1.

SOL_FIRSTA = FirstSolution(np.array([0.,1.]))
SOL_FIRSTA.flag_accepted = True
SOL_FIRSTA.flag_J = True
SOL_FIRSTA.flag_J_f = True
SOL_FIRSTA.J_f = np.ones((2,2))


@pytest.mark.all
@pytest.mark.unit
def test_PredictorTangent_factory_keyword()->None:
    assert PredictorTangent.factory_keyword == 'tangent' 


@pytest.mark.all
@pytest.mark.unit
def test_PredictorTangent_predict() -> None :
    P = PredictorTangent(SIGN_DS,norm='om',bifurcation_detection=False)
    SOLLIST = [SOL_FIRSTA]
    xpred, lstpt, sign_ds = P.predict(SOLLIST, DS, k_imposed=0)
    assert xpred[-1] == 2.
    assert xpred[0] == pytest.approx(-1.)
    assert lstpt == SOL_FIRSTA
    assert sign_ds == SIGN_DS