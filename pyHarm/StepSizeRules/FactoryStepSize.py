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
Module that contains the factory of ABCStepSizeRule objects. 

Attributes:
    StepSizer_dico (dict): Dictionary containing the factory keywords and their associated class of ABCStepSizeRule objects.
"""
from pyHarm.StepSizeRules.ABCStepSizeRule import ABCStepSizeRule 
from pyHarm.StepSizeRules.StepSizeConstant import StepSizeConstant 
from pyHarm.StepSizeRules.StepSizeAcceptance import StepSizeAcceptance 


StepSizer_dico = {StepSizeConstant.factory_keyword:                            StepSizeConstant,
                  StepSizeAcceptance.factory_keyword:                          StepSizeAcceptance}
"""dict: Dictionary containing the factory keywords and their associated class of ABCStepSizeRule objects."""


def generateStepSizeRule(name_stepsize, bounds:list[float,float], stepsize_options) -> ABCStepSizeRule:
    """
    Factory function that creates a ABCStepSizeRule object.

    Args:
        name_stepsize (str): Type of the stepsize rule object that is to be instanciated.
        bounds (list[float,float]): list containing the bounds of the step-size [min_step, max_step].
        stepsize_options (dict): dictionary containing complementary keywords argument that is passed to the initialisation of the object.

    Returns:
        ABCStepSizeRule: Instance of the required ABCStepSizeRule class.
    """
    E = StepSizer_dico[name_stepsize](bounds, **stepsize_options)
    return E