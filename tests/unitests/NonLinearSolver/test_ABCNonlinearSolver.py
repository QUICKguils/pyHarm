import numpy as np
import pytest

from pyHarm.NonLinearSolver.ABCNonLinearSolver import ABCNLSolver
from pyHarm.Solver import SystemSolution

fakeSystemSolution = SystemSolution(xs=np.array([1.0]))
residual = lambda res: np.zeros(
    1,
)
jacobian = lambda res: np.zeros((1, 1))
solver_options = {"a": 1.0, "b": 2.0}
factory_keyword = "fakeABCNLSolver"


class fakeABCNLSolver(ABCNLSolver):
    factory_keyword = factory_keyword

    def Solve(self, system_solution: SystemSolution) -> SystemSolution:
        return system_solution

    def __post_init__(self):
        self.post_init_ran = True
        pass


@pytest.mark.all
@pytest.mark.unit
def test_ABCNLSolver_init():
    solver = fakeABCNLSolver(residual, jacobian, solver_options)
    assert solver.residual == residual
    assert solver.jacobian == jacobian
    assert solver.solver_options == solver_options
    assert solver.post_init_ran == True


@pytest.mark.all
@pytest.mark.unit
def test_ABCNLSolver_factory_keyword():
    solver = fakeABCNLSolver(residual, jacobian, solver_options)
    assert solver.factory_keyword == factory_keyword


@pytest.mark.all
@pytest.mark.unit
def test_ABCNLSolver_Solve():
    solver = fakeABCNLSolver(residual, jacobian, solver_options)
    assert solver.Solve(fakeSystemSolution) == fakeSystemSolution
