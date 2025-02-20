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
This module is the factory of the ABCElement excluding the Substructure kind.

Attributes:
    L_Elem (list[ABCElement]): List of available ABCElement subclasses available for creation.
    ElementDictionary (dict[str, ABCElement]): Dictionary of available ABCElement as values and their factory_keyword as key.
"""

from pyHarm.CoordinateSystem import CoordinateSystem

# --- Elements
# ------ NodeToNodeElements
from pyHarm.Elements.ABCElement import ABCElement
from pyHarm.Elements.NodeToNodeElements.GeneralOrderElement import GeneralOrderElement
from pyHarm.Elements.NodeToNodeElements.LinearDamper import LinearDamper
from pyHarm.Elements.NodeToNodeElements.LinearSpring import LinearSpring
from pyHarm.Elements.NodeToNodeElements.CubicSpring import CubicSpring
from pyHarm.Elements.NodeToNodeElements.PenaltyBilateralGap import PenaltyBilateralGap
from pyHarm.Elements.NodeToNodeElements.PenaltyUnilateralGap import PenaltyUnilateralGap
from pyHarm.Elements.NodeToNodeElements.Jenkins import Jenkins
from pyHarm.Elements.NodeToNodeElements.Penalty3D import Penalty3D
from pyHarm.Elements.NodeToNodeElements.GeneralOrderForcing import GeneralOrderForcing
from pyHarm.Elements.NodeToNodeElements.CosinusForcing import CosinusForcing
from pyHarm.Elements.NodeToNodeElements.SinusForcing import SinusForcing
# --------- NodeToNodeElements/DLFTElements
from pyHarm.Elements.NodeToNodeElements.DLFTElements.DLFTUniGap import DLFTUniGap
from pyHarm.Elements.NodeToNodeElements.DLFTElements.DLFTFriction import DLFTFriction
from pyHarm.Elements.NodeToNodeElements.DLFTElements.DLFT3D import DLFT3D

# --- SubstructureMatrixElements
from pyHarm.Elements.SubstructureMatrixElements.Substructure import Substructure
from pyHarm.Elements.SubstructureMatrixElements.GeneralOrderMatrixElement import GOMatrix
from pyHarm.Elements.SubstructureMatrixElements.LinearHystMatrixElement import LinearHystMatrix

# --- Forcing 

L_Elem = [GeneralOrderElement,
          LinearDamper,
          LinearSpring,
          CubicSpring,
          PenaltyBilateralGap,
          PenaltyUnilateralGap,
          Jenkins,
          Penalty3D,
          DLFTUniGap,
          DLFT3D,
          DLFTFriction,
          GeneralOrderForcing,
          CosinusForcing,
          SinusForcing,
          Substructure,
          GOMatrix,
          LinearHystMatrix]
"""list[ABCElement]: List of available ABCElement subclasses available for creation."""

ElementDictionary = {e.factory_keyword:e for e in L_Elem}
"""dict[str, ABCElement]: Dictionary of available ABCElement as values and their factory_keyword as key."""

def generateElement(nh, nti, name, data, dict_CS:dict[str,CoordinateSystem]) -> ABCElement:
    """
    Factory function that creates an ABCElement object.

    Args:
        nh (int): number of harmonics.
        nti (int): number of time steps.
        name (str): type of kinematic condition to instantiate.
        data (dict): dictionary containing the definition of the element.
        dict_CS (dict[str,CoordinateSystem]): dictionary containing all the local and global coordinate systems.

    Returns:
        ABCElement: Instance of the required ABCElement class.
    """
    typeE = data["type"]
    if "coordinatesystem" in data : 
        CS = dict_CS[data["coordinatesystem"]]
    else : 
        CS = dict_CS["global"]
    E = ElementDictionary[typeE](nh, nti, name, data, CS)
    return E