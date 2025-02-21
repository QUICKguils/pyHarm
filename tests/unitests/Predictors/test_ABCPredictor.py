import numpy as np
import pytest
from pytest import MonkeyPatch

from pyHarm.Predictors.ABCPredictor import ABCPredictor
from pyHarm.Solver import FirstSolution, SystemSolution

SIGN_DS = 1
DIRECTION_TEST = np.array([1.0, 0.0, 1.0, 5.0])

SOL_FIRST = FirstSolution(np.array([0.0]))
SOL_FIRST.flag_accepted = True
SOL_ACCEPT = SystemSolution(np.array([0.0]))
SOL_ACCEPT.flag_accepted = True
SOL_NOTACC = SystemSolution(np.array([0.0]))
SOL_NOTACC.flag_accepted = False


@pytest.fixture
def mock_abcpredictor(monkeypatch: MonkeyPatch) -> ABCPredictor:
    monkeypatch.setattr(ABCPredictor, "__abstractmethods__", set())
    monkeypatch.setattr(ABCPredictor, "factory_keyword", "mock_abspredictor")
    monkeypatch.setattr(
        ABCPredictor,
        "predict",
        lambda self, sollist, ds: (np.array([0.0]), SystemSolution(np.array([0.0])), 1),
    )
    return ABCPredictor(SIGN_DS)


@pytest.mark.all
@pytest.mark.unit
def test_ABCPredictor__init__(mock_abcpredictor: ABCPredictor) -> None:
    assert mock_abcpredictor.sign_ds == SIGN_DS
    assert mock_abcpredictor.predictor_options == ABCPredictor.default_options


@pytest.mark.all
@pytest.mark.unit
def test_ABCPredictor_norm_dir(mock_abcpredictor: ABCPredictor) -> None:
    assert np.allclose(
        mock_abcpredictor.norm_dir(DIRECTION_TEST),
        DIRECTION_TEST / np.linalg.norm(DIRECTION_TEST),
    )
    mock_abcpredictor.predictor_options["norm"] = "om"
    assert mock_abcpredictor.norm_dir(DIRECTION_TEST)[-1] == 1.0


@pytest.mark.all
@pytest.mark.unit
def test_ABCPredictor_getPointerToSolution(mock_abcpredictor: ABCPredictor) -> None:
    sollist = [SOL_ACCEPT, SOL_NOTACC]
    assert mock_abcpredictor.getPointerToSolution(sollist, k_imposed=0) == SOL_ACCEPT
    assert mock_abcpredictor.getPointerToSolution(sollist, k_imposed=1) == SOL_NOTACC
    assert mock_abcpredictor.getPointerToSolution(sollist, k_imposed=None) == SOL_ACCEPT


@pytest.mark.all
@pytest.mark.unit
def test_ABCPredictor_bifurcation_detect(mock_abcpredictor: ABCPredictor) -> None:
    SOL_FIRST.flag_J = True
    SOL_FIRST.flag_J_f = True
    SOL_FIRST.J_f = np.eye(2)
    mock_abcpredictor.bifurcation_detect(SOL_FIRST)
    assert SOL_FIRST.flag_bifurcation == False
    SOL_ACCEPT.flag_J = True
    SOL_ACCEPT.flag_J_f = True
    SOL_ACCEPT.J_f = np.eye(2)
    SOL_ACCEPT.precedent_solution = SOL_NOTACC
    SOL_NOTACC.det_J_x = 1.0
    SOL_NOTACC.det_J_f = 2.0
    mock_abcpredictor.bifurcation_detect(SOL_ACCEPT)
    assert SOL_ACCEPT.flag_bifurcation == False
    mock_abcpredictor.predictor_options["flag_print"] = False
    SOL_NOTACC.det_J_x = -1.0
    SOL_NOTACC.det_J_f = 2.0
    mock_abcpredictor.bifurcation_detect(SOL_ACCEPT)
    assert SOL_ACCEPT.flag_bifurcation == True
    assert mock_abcpredictor.sign_ds == -SIGN_DS
    SOL_NOTACC.det_J_x = -1.0
    SOL_NOTACC.det_J_f = -2.0
    mock_abcpredictor.bifurcation_detect(SOL_ACCEPT)
    assert SOL_ACCEPT.flag_bifurcation == True
    assert mock_abcpredictor.sign_ds == -SIGN_DS
    mock_abcpredictor.predictor_options["bifurcation_detect"] = False
    assert mock_abcpredictor.bifurcation_detect(SOL_ACCEPT) == None
