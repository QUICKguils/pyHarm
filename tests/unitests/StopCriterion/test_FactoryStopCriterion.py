import pytest 
from pytest import MonkeyPatch
from pyHarm.Solver import SystemSolution
from pyHarm.StopCriterion.FactoryStopCriterion import Stopper_dico, generateStopCriterion
from pyHarm.StopCriterion.ABCStopCriterion import ABCStopCriterion
@pytest.mark.all
@pytest.mark.unit
def test_Stopper_dico() -> None :
    for key,stopper in Stopper_dico.items() : 
        assert key == stopper.factory_keyword

BOUNDS = [0.,1.]
DS_MIN = 3.
@pytest.mark.all
@pytest.mark.unit
def test_generateStopCriterion()->None:
    class FakeStopper(ABCStopCriterion) : 
        factory_keyword = 'fakestopper'
        def getStopCriterionStatus(self, *args, **kwargs) -> bool:
            return True
    Stopper_dico[FakeStopper.factory_keyword] = FakeStopper
    SC = generateStopCriterion(FakeStopper.factory_keyword,BOUNDS,DS_MIN,dict())
    assert isinstance(SC,FakeStopper)