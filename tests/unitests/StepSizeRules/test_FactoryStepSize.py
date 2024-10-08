import pytest
from pytest import MonkeyPatch
from pyHarm.Solver import SystemSolution
from pyHarm.StepSizeRules.FactoryStepSize import StepSizer_dico, generateStepSizeRule
from pyHarm.StepSizeRules.ABCStepSizeRule import ABCStepSizeRule

BOUNDS = [1e-1,1e2]
DS = 1.0

@pytest.mark.all
@pytest.mark.unit
def test_StepSizer_dico() -> None :
    for key,stepsizer in StepSizer_dico.items() :
        assert stepsizer.factory_keyword == key

@pytest.mark.all
@pytest.mark.unit
def test_generateStepSizeRule() -> None:
    class FakeStep(ABCStepSizeRule) : 
        factory_keyword = 'fakestep'
        def getStepSize(self, *args, **kwargs) -> float:
            return True
    StepSizer_dico[FakeStep.factory_keyword] = FakeStep
    SSR = generateStepSizeRule(FakeStep.factory_keyword, BOUNDS, dict())
    assert isinstance(SSR, ABCStepSizeRule)
    assert isinstance(SSR, FakeStep)
    assert SSR.getStepSize() == True
