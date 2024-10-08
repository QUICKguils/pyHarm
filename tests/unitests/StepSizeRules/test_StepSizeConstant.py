import pytest
from pyHarm.StepSizeRules.StepSizeConstant import StepSizeConstant

BOUNDS = [1e-1,1e2]
DS = 1.0

@pytest.mark.all
@pytest.mark.unit
def test_StepSizeConstant() -> None :
    SSR = StepSizeConstant(BOUNDS)
    assert StepSizeConstant.factory_keyword == 'constant'
    assert SSR.getStepSize(DS, list()) == DS