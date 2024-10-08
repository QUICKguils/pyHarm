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

"""This module is the factory of ABCAnalysis subclasses.

Attributes:
    NonLinearStudy_kind (dict[str, ABCAnalysis]): Dictionary containing ABCAnalysis subclasses as values and their factory_keyword attribute as key.
"""
from pyHarm.Analysis.ABCAnalysis import ABCAnalysis
from pyHarm.Systems.ABCSystem import ABCSystem
from pyHarm.Analysis.FRF_NonLinear import FRF_NonLinear
from pyHarm.Analysis.Linear_Analysis import Linear_Analysis

NonLinearStudy_kind = {FRF_NonLinear.factory_keyword:           FRF_NonLinear,
                       Linear_Analysis.factory_keyword:         Linear_Analysis}
"""dict[str, ABCAnalysis]: Dictionary containing ABCAnalysis subclasses as values and their factory_keyword attribute as key."""

def generateNonLinearAnalysis(name_nonlinearstudy, datas:dict, system:ABCSystem, **kwargs) -> ABCAnalysis:
    """
    Factory function that creates a ABCAnalysis object.

    Args:
        name_nonlinearstudy (str): type of ABCAnalysis to instantiate.
        datas (dict): dictionary containing the definition of the analysis.
        system (ABCSystem): System associated with the analysis.

    Returns:
        ABCAnalysis: Instance of the required ABCAnalysis class.
    """
    E = NonLinearStudy_kind[name_nonlinearstudy](datas, system, **kwargs)
    return E