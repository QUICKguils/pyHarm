from pyHarm.NonLinearSolver.FactoryNonLinearSolver import generateNonLinearSolver, Solver_dico
from pyHarm.NonLinearSolver.ABCNonLinearSolver import ABCNLSolver
from pyHarm.Solver import FirstSolution
import pytest
import numpy as np

residual = lambda x,_ : x
jacobian = lambda x,_ : np.array([[1.]])
solver_options = dict()

@pytest.mark.all
@pytest.mark.unit
def test_factoryNonLinearSolvers_dico():
    for key, val in Solver_dico.items():
        assert issubclass(val,ABCNLSolver)
        assert isinstance(key, str)

@pytest.mark.all
@pytest.mark.unit
def test_factoryNonLinearSolvers_factory():
    for key, val in Solver_dico.items():
        solver = generateNonLinearSolver(key, residual, jacobian, solver_options)
        assert isinstance(solver, val)
