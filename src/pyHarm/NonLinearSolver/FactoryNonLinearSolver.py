# Copyright 2024 SAFRAN SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the factory of ABCNLSolver.

Attributes:
    Solver_dico (dict): Dictionary containing all the available ABCNLSolvers as values, and their factory_keyword attribute as key.
"""

from pyHarm.NonLinearSolver.ABCNonLinearSolver import ABCNLSolver
from pyHarm.NonLinearSolver.ScipyRoot import Solver_ScipyRoot
from pyHarm.NonLinearSolver.MoorePenrose import Solver_MoorePenrose
from pyHarm.NonLinearSolver.NewtonRaphson import Solver_NewtonRaphson

Solver_dico = {Solver_ScipyRoot.factory_keyword:     Solver_ScipyRoot,
               Solver_MoorePenrose.factory_keyword:  Solver_MoorePenrose,
               Solver_NewtonRaphson.factory_keyword:  Solver_NewtonRaphson}
"""dict: Dictionary containing all the available ABCNLSolvers as values, and their factory_keyword attribute as key."""

def generateNonLinearSolver(name_nonlinearsolver, residual, jacobian, nonlinearsolver_options) -> ABCNLSolver:
    """
    Factory function that creates a ABCNLSolver object.

    Args:
        name_nonlinearsolver (str): type of nonlinear solver to instantiate.
        residual (Callable): function that returns the residual of the system.
        jacobian (Callable): function that returns the jacobians of the system.
        nonlinearsolver_options (dict): dictionary containing the supplementary options for the nonlinear solver.

    Returns:
        ABCNLSolver: Instance of the required ABCNLSolver class.
    """
    E = Solver_dico[name_nonlinearsolver](residual, jacobian, nonlinearsolver_options)
    return E