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

""""
This module is responsible for the creation of the ABCSubstructure instances used in pyHarm.
"""

from pyHarm.Substructures.ABCSubstructure import ABCSubstructure
from pyHarm.Substructures.OnlyDofs import OnlyDofs
from pyHarm.Substructures.Substructure import Substructure
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary

SubstructureDico = {
    OnlyDofs.factory_keyword:       OnlyDofs,
    Substructure.factory_keyword:   Substructure
}
"""dict[str,ABCSubstructure]: List of available ABCSubstructure subclasses available for creation."""

def generate_substructure(nh, name, data) -> ABCSubstructure:
    """
    Factory function that creates an ABCSubstructure object.

    Args:
        nh (int): number of harmonics.
        name (str): type of substructure to instantiate.
        data (dict): dictionary containing the definition of the substructure.

    Returns:
        ABCSubstructure: Instance of the required ABCSubstructure class.
    """
    default = {'type':'substructure','reader':'generic'}
    data = getCustomOptionDictionary(data,default)
    return SubstructureDico[data['type']](nh, name, data)