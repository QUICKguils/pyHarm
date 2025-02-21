import numpy as np
import pytest
from pytest import MonkeyPatch

from pyHarm.Predictors.PredictorPreviousSolution import PredictorPreviousSolution
from pyHarm.Solver import FirstSolution, SystemSolution

SIGN_DS = 1
DS = 1.0
SOL_ACCEPT = FirstSolution(np.array([0.0, 1.0]))
SOL_ACCEPT.flag_accepted = True
SOL_ACCEPT.flag_J = True
SOL_ACCEPT.flag_J_f = True
SOL_ACCEPT.J_f = np.eye(2)


@pytest.mark.all
@pytest.mark.unit
def test_PredictorPreviousSolution_factory_keyword() -> None:
    assert PredictorPreviousSolution.factory_keyword == "previous"


@pytest.mark.all
@pytest.mark.unit
def test_PredictorPreviousSolution_predict() -> None:
    P = PredictorPreviousSolution(SIGN_DS, norm="om", bifurcation_detection=False)
    SOLLIST = [SOL_ACCEPT]
    xpred, lstpt, sign_ds = P.predict(SOLLIST, DS, k_imposed=0)
    assert xpred[-1] == 2.0
    assert np.allclose(xpred[:-1], SOL_ACCEPT.x[:-1])
    assert lstpt == SOL_ACCEPT
    assert sign_ds == SIGN_DS
