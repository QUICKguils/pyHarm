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


######## abstract Stopper ########
class ABCStopCriterion(abc.ABC):
    """Abstract class for the stopping criterion. Any stopping criterion code
    shall inherit from this class.
    
    Args:
        bounds (list[float, float]): A list of upper and lower bounds.
        ds_min: The minimum step size.
        **kwargs: Additional keyword arguments.

    Raises:
        NotImplementedError: If the subclass lacks the factory_keyword class attribute.
        TypeError: If the factory_keyword is not a string value.
    """
    @property
    @abc.abstractmethod
    def factory_keyword(self)->str:
        """
        Returns:
            str: keyword that is used to call the creation of this class in the system factory.
        """
        ...
        
    
    def __init__(self,bounds:list[float,float], ds_min:float,**kwargs) -> None:
        self.puls_inf = bounds[0]
        self.puls_sup = bounds[1]
        self.epsilon_bounds = (self.puls_inf+self.puls_sup) / 2 * 1E-5
        self.bound_inf = self.puls_inf - self.epsilon_bounds
        self.bound_sup = self.puls_sup + self.epsilon_bounds
        self.ds_min = ds_min
        pass

    @abc.abstractmethod
    def getStopCriterionStatus(self,sol:SystemSolution,sollist:list,**kwargs) -> bool:
        """Abstract method to get the stop criterion status.

        Args:
            sol (SystemSolution): A SystemSolution object.
            sollist (list): A list of solutions.
            **kwargs: Additional keyword arguments.
        
        Raises:
            None.
        """
        pass