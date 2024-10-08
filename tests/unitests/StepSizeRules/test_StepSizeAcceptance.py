import pytest
from pyHarm.StepSizeRules.StepSizeAcceptance import StepSizeAcceptance
from pyHarm.Solver import SystemSolution
import numpy as np

BOUNDS = [1e-1,1e2]
DS = 1.0

@pytest.mark.all
@pytest.mark.unit
def test_StepSizeAcceptance() -> None :
    SSR = StepSizeAcceptance(BOUNDS)
    assert StepSizeAcceptance.factory_keyword == 'acceptance'
    assert SSR.stepsize_options == StepSizeAcceptance.default_options
    assert SSR.consecutive_accept == StepSizeAcceptance.default_options["consecutive_accept"]
    kwargs = {'stepsize_options':{"consecutive_accept":2}}
    SSR = StepSizeAcceptance(BOUNDS,**kwargs)
    assert SSR.consecutive_accept == kwargs['stepsize_options']['consecutive_accept']
    ## test get_stepsize() and the many cases : 
    # Creation of fake solutions : 
    sol_accept = SystemSolution(np.array([0.]))
    sol_accept.flag_accepted = True
    sol_notacc = SystemSolution(np.array([0.]))
    sol_notacc.flag_accepted = False
    # case 1  : last solution not accepted 
    assert SSR.getStepSize(DS, [sol_notacc]) == DS/2.
    # case 2  : last solution not accepted 
    assert SSR.getStepSize(DS, [sol_accept]*kwargs['stepsize_options']['consecutive_accept']) == DS*2.
    # case 3 : not enough solutions : 
    assert SSR.getStepSize(DS, [sol_accept]) == DS


