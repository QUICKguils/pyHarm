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


class Corrector_pseudo_arc_length(ABCCorrector):
    """Corrector corresponding to the pseudo_arc_length method where the equations are solved imposing that the solution is ortogonal to the prediction direction."""

    factory_keyword : str = "pseudo_arc_length"
    """str: name of the class to call in the factory in order to create a instance of the class."""

    def ClosureEquation(self, solx:np.ndarray,sol:SystemSolution,sollist:list[SystemSolution],**kwargs) -> np.ndarray:
        """Compute the residual contribution of the correction equation.

        Args:
            solx (np.ndarray): actual displacement vector.
            sol (SystemSolution): actual SystemSolution that contains the starting point.
            sollist (list[SystemSolution]): list of SystemSolutions from previous analysis steps.

        Returns:
            np.ndarray: Residual of the correction equation.
        """
        R_cont =  np.dot((sol.precedent_solution.x_pred - sol.precedent_solution.x),solx-sol.precedent_solution.x_pred)
        return R_cont

    def ClosureJacobian(self, solx:np.ndarray,sol:SystemSolution,sollist:list[SystemSolution],**kwargs) -> tuple[np.ndarray,np.ndarray]:
        """Compute the jacobian contribution of the correction equation.

        Args:
            solx (np.ndarray): actual displacement vector.
            sol (SystemSolution): actual SystemSolution that contains the starting point.
            sollist (list[SystemSolution]): list of SystemSolutions from previous analysis steps.

        Returns:
            tuple[np.ndarray,np.ndarray]: Jacobians of the correction equation.
        """
        dRdx = np.transpose((sol.precedent_solution.x_pred - sol.precedent_solution.x)[:-1])
        dRdom = (sol.precedent_solution.x_pred - sol.precedent_solution.x)[-1]
        return dRdx, dRdom