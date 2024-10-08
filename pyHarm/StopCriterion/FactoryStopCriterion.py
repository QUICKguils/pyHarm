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
Factory module responsible for creating the ABCStopCriterion needed.

Attributes:
    Stopper_dico (dict[str,ABCStopCriterion]): Dictionary containing all the StopCriterions available.
"""

from pyHarm.StopCriterion.ABCStopCriterion import ABCStopCriterion
from pyHarm.StopCriterion.StopCriterionBounds import StopCriterionBounds
from pyHarm.StopCriterion.StopCriterionBoundsOrSolNumber import StopCriterionBoundsOrSolNumber


Stopper_dico = {StopCriterionBounds.factory_keyword:                                StopCriterionBounds,
                StopCriterionBoundsOrSolNumber.factory_keyword:                             StopCriterionBoundsOrSolNumber}
"""dict[str,ABCStopCriterion]: Dictionary containing all the StopCriterions available."""

def generateStopCriterion(name_stopcriterion:str, bounds:list[float,float], ds_min:float, stopcriterion_options) -> ABCStopCriterion:
    """
    Factory function that creates a ABCSystem object.

    Args:
        name_stopcriterion (str): Type of the ABCStopCriterion object that is to be instantiated.
        bounds (list[float,float]): angular frequency bounds [puls_inf, puls_sup].
        ds_min (float): minimum step size.
        stopcriterion_options (dict): dictionary containing options to pass to the instantiated ABCStopCriterion.

    Returns:
        ABCStopCriterion: Instance of the required ABCStopCriterion class.
    """
    E = Stopper_dico[name_stopcriterion](bounds, ds_min, **stopcriterion_options)
    return E