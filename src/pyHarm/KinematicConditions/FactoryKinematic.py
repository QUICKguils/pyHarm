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

"""Module that contains the factory of the ABCKinematic subclasses.

Attributes:
    Kinematic_dico (dict): Dictionary containing ABCKinematic as values and their factory_keyword attribute as key.

"""
from pyHarm.CoordinateSystem import CoordinateSystem
from pyHarm.KinematicConditions.ABCKinematic import ABCKinematic
from pyHarm.KinematicConditions.GODisplacement import GODisplacement
from pyHarm.KinematicConditions.AccelImposed import AccelImposed
from pyHarm.KinematicConditions.SpeedImposed import SpeedImposed
from pyHarm.KinematicConditions.DispImposed import DispImposed
from pyHarm.KinematicConditions.BaseProjection import BaseProjection

Kinematic_dico = {
    GODisplacement.factory_keyword :        GODisplacement,
    AccelImposed.factory_keyword:           AccelImposed,
    SpeedImposed.factory_keyword:           SpeedImposed,
    DispImposed.factory_keyword:           DispImposed,
    BaseProjection.factory_keyword:           BaseProjection,
}
"""dict: Dictionary containing ABCKinematic as values and their factory_keyword attribute as key."""


def generateKinematic(nh, nti, name, data, dict_CS:dict[str,CoordinateSystem]) -> ABCKinematic:
    """
    Factory function that creates a ABCKinematic object.

    Args:
        nh (int): number of harmonics.
        nti (int): number of time steps.
        name (str): type of kinematic condition to instantiate.
        data (dict): dictionary containing the definition of the kinematic condition.
        dict_CS (dict[str,CoordinateSystem]): dictionary containing all the local and global coordinate systems.

    Returns:
        ABCKinematic: Instance of the required ABCKinematic class.
    """
    typeK = data["type"]
    if "coordinatesystem" in data : 
        CS = dict_CS[data["coordinatesystem"]]
    else : 
        CS = dict_CS["global"]
    K = Kinematic_dico[typeK](nh, nti, name, data, CS)
    return K