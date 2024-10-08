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
This is the factory of the ABCCorrector subclasses. 

Attributes:
    Corrector_dico (dict[str, ABCCorrector]): Dictionary containing the ABCCorrector subclasses as values and their factory_keyword as key.
"""

from pyHarm.Correctors.CorrectorArcLength import Corrector_arc_length
from pyHarm.Correctors.CorrectorPseudoArcLength import Corrector_pseudo_arc_length
from pyHarm.Correctors.CorrectorNoContinuation import Corrector_no_continuation
from pyHarm.Correctors.ABCCorrector import ABCCorrector


Corrector_dico = {Corrector_no_continuation.factory_keyword:                      Corrector_no_continuation,
                  Corrector_arc_length.factory_keyword:                          Corrector_arc_length,
                  Corrector_pseudo_arc_length.factory_keyword:                   Corrector_pseudo_arc_length}
"""dict[str, ABCCorrector]: Dictionary containing the ABCCorrector subclasses as values and their factory_keyword as key."""


def generateCorrector(name_corrector, corrector_options) -> ABCCorrector:
    """
    Factory function that creates an ABCCorrector object.

    Args:
        name_corrector (str): Type of the ABCCorrector object that is to be instantiated.
        corrector_options (dict): supplementary options passed to the corrector.

    Returns:
        ABCCorrector: Instance of the required ABCCorrector class.
    """
    E = Corrector_dico[name_corrector](**corrector_options)
    return E