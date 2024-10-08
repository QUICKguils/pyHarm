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


class StepSizeConstant(ABCStepSizeRule):
    """
    Step size is kept constant.
    """
    factory_keyword : str = "constant"
    name = "constant step size"

    def getStepSize(self, ds:float, sollist:list[SystemSolution], **kwargs) -> float:
        """Returns the step size to be used for the prediction step of the analysis.

        Args: 
            ds (float): Current step size.
            sollist (list[SystemSolution]): list of SystemSolution returned during the analysis.

        Returns:
            float: updated step size.
        """
        return ds