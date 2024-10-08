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

from pyHarm.StepSizeRules.ABCStepSizeRule import ABCStepSizeRule 
from pyHarm.Solver import SystemSolution
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
import numpy as np


class StepSizeAcceptance(ABCStepSizeRule):
    """
    Step size is divided by 2 if last solution is not accepted or multiplied by 2 if a number of consecutive accepted solutions is reached.

    Attributes:
        default_options (dict): contains default step size options concerning consecutive_accept if not provided during creation.
        consecutive_accept (int): number of consecutive accept before increasing step size.

    """
    name = "accepted step size adaptation"
    factory_keyword : str = "acceptance"
    default_options = {"consecutive_accept":5}
    def __init__(self,bounds:list[float,float],**kwargs):
        super().__init__(bounds)
        self.stepsize_options = getCustomOptionDictionary(kwargs.get("stepsize_options",dict()),self.default_options)
        self.consecutive_accept = self.stepsize_options["consecutive_accept"]
    
    def getStepSize(self, ds:float, sollist:list[SystemSolution], **kwargs) -> float:
        """Returns the step size to be used for the prediction step of the analysis.

        Args: 
            ds (float): Current step size.
            sollist (list[SystemSolution]): list of SystemSolution returned during the analysis.

        Returns:
            float: updated step size.
        """
        if ((not sollist[-1].flag_accepted) and (ds>self.ds_min)): 
            ds/=2
        try :
            acc = np.array([sol.flag_accepted for sol in sollist[-self.consecutive_accept::]])
            if np.sum(acc) == self.consecutive_accept and ds<self.ds_max: 
                ds*=2
        except:
            pass
        if ds<self.ds_min or ds>self.ds_max: ds = self.ProjectInBounds(ds) #shouldn't be necessary
        return ds