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

import abc
import numpy as np
from pyHarm.Solver import SystemSolution

class ABCCorrector(abc.ABC):
    """This is the abstract class ruling the corrector class. The system is responsible for adding the correction residual equation to the augmented system.
    
    """
    @property
    @abc.abstractmethod
    def factory_keyword(self)->str:
        """
        str: name of the class to call in the factory in order to create an instance of the class.
        """
        ...
    
    def __init__(self,**kwargs):
        pass

    @abc.abstractmethod
    def ClosureEquation(self, solx:np.ndarray, sol:SystemSolution,sollist:list[SystemSolution]) -> np.ndarray:
        """Computes the residual contribution of the correction equation.

        Args:
            solx (np.ndarray): actual displacement vector.
            sol (SystemSolution): actual SystemSolution that contains the starting point.
            sollist (list[SystemSolution]): list of SystemSolutions from previous analysis steps.
        """
        pass

    @abc.abstractmethod
    def ClosureJacobian(self, solx:np.ndarray, sol:SystemSolution,sollist:list[SystemSolution]) -> tuple[np.ndarray,np.ndarray]:
        """Computes the jacobian contribution of the correction equation.

        Args:
            solx (np.ndarray): actual displacement vector.
            sol (SystemSolution): actual SystemSolution that contains the starting point.
            sollist (list[SystemSolution]): list of SystemSolutions from previous analysis steps.
        """
        pass