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

from pyHarm.Correctors.ABCCorrector import ABCCorrector
import numpy as np
from pyHarm.Solver import SystemSolution


class Corrector_arc_length(ABCCorrector):
    """
    Corrector that uses an arc length parametrisation equation.
    """

    factory_keyword : str = "arc_length"
    """str: name of the class to call in the factory in order to create an instance of the class."""


    def ClosureEquation(self, solx:np.ndarray,sol:SystemSolution,sollist:list[SystemSolution],**kwargs) -> np.ndarray:
        """Computes the residual contribution of the correction equation.

        Args:
            solx (np.ndarray): actual displacement vector.
            sol (SystemSolution): actual SystemSolution that contains the starting point.
            sollist (list[SystemSolution]): list of SystemSolutions from previous analysis steps.

        Returns:
            np.ndarray: Residual of the correction equation.
        """
        ds = np.linalg.norm(sol.precedent_solution.x_pred - sol.precedent_solution.x)
        R_cont =  (np.linalg.norm(solx[:-1]-sol.precedent_solution.x[:-1]) ** 2 + (solx[-1]-sol.precedent_solution.x[-1]) ** 2 - ds ** 2)
        return R_cont

    def ClosureJacobian(self, solx:np.ndarray,sol:SystemSolution,sollist:list[SystemSolution],**kwargs) -> tuple[np.ndarray,np.ndarray]:
        """Computes the jacobian contribution of the correction equation.

        Args:
            solx (np.ndarray): actual displacement vector.
            sol (SystemSolution): actual SystemSolution that contains the starting point.
            sollist (list[SystemSolution]): list of SystemSolutions from previous analysis steps.

        Returns:
            tuple[np.ndarray,np.ndarray]: Jacobians of the correction equation.
        """
        dRdx = (2. * (solx[:-1] - sol.precedent_solution.x[:-1]).T)
        dRdom = (2. * (solx[-1] - sol.precedent_solution.x[-1]) )
        return dRdx, dRdom