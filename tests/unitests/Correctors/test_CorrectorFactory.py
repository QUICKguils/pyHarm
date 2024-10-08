import pytest 
from pytest import MonkeyPatch
import numpy as np
from pyHarm.Correctors.ABCCorrector import ABCCorrector
from pyHarm.Correctors.FactoryCorrector import Corrector_dico, generateCorrector
from pyHarm.Solver import SystemSolution

@pytest.mark.all
@pytest.mark.unit
def test_Corrector_dico() -> None:
    for key,corrector in Corrector_dico.items():
        assert corrector.factory_keyword == key


@pytest.mark.all
@pytest.mark.unit
def test_generateCorrector()->None:
    class FakeCor(ABCCorrector):
        factory_keyword='fakecor'
        def ClosureEquation(self, *args):
            pass
        def ClosureJacobian(self, *args):
            pass
    Corrector_dico[FakeCor.factory_keyword] = FakeCor
    C = generateCorrector(FakeCor.factory_keyword, dict())
    assert isinstance(C,ABCCorrector)