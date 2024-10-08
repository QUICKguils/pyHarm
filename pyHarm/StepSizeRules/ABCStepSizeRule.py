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

######## abstract StepSizer ########
class ABCStepSizeRule(abc.ABC):
    """This is the abstract class ruling the stepsize rules class. The step size rules are responsible for adjusting the step size of the analysis depending on the given inputs.
    
    Args:
        bounds (list[float,float]): List containing the step size bounds [min_step, max_step].

    Attributes:
        ds_min (float): min step size.
        ds_max (float): max step size.
    """
    @property
    @abc.abstractmethod
    def factory_keyword(self)->str:
        """
        Returns:
            str: keyword that is used to call the creation of this class in the system factory.
        """
        ...

    def __init__(self,bounds:list[float,float],**kwargs):
        self.ds_min=bounds[0]
        self.ds_max = bounds[1]
        pass

    
    @abc.abstractmethod
    def getStepSize(self,ds:float,sollist:list[SystemSolution],**kwargs) -> float:
        """Returns the step size to be used for the prediction step of the analysis.
        """
        pass

    def ProjectInBounds(self,ds:float) -> float:
        """Projects the step-size onto the bounds if the step-size is out of the required bounds.

        Args: 
            ds (float): step-size.

        Returns:
            float: step-size projected onto the bounds if it is out of the bounds.
        """
        return np.max([self.ds_min,np.min([ds,self.ds_max])])
