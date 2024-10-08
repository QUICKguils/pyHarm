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
from pyHarm.Solver import SystemSolution

class ABCNLSolver(abc.ABC):
    """This is the abstract class ruling the solver class. The system is responsible of solving the system starting at a given starting point.
    
    Args:
        solver_options (dict): dictionary containing other options for creation of the solver class.
        residual (Callable): function that returns the residual vector of the system to be solved.
        jacobian (Callable): function that returns the jacobian matrix of the system to be solved.
    """
    @property
    @abc.abstractmethod
    def factory_keyword(self)->str:
        """
        keyword that is used to call the creation of this class in the system factory.
        """
        ...
        
    def __init__(self, residual, jacobian, solver_options):
        self.solver_options = solver_options
        self.residual = residual
        self.jacobian = jacobian
        self.__post_init__()

    def __post_init__(self):
        pass
    
    @abc.abstractmethod
    def Solve(self,system_solution:SystemSolution) -> SystemSolution:
        """Runs the solver.

        Args:
            system_solution (SystemSolution): SystemSolution that contains the starting point.
        """
        pass
