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
Module that contains the factory of reducers.

Attributes:
    Reductor_dico (dict): Dictionary containing available ABCReductors as values and their factoryu keyword as key.
"""
from pyHarm.Reductors.ABCReductor import ABCReductor
from pyHarm.Reductors.NoReductor import NoReductor
from pyHarm.Reductors.StaticReductor import StaticReductor
from pyHarm.Reductors.GlobalHarmonicReductor import GlobalHarmonicReductor
from pyHarm.Reductors.LocalHarmonicReductor import LocalHarmonicReductor
from pyHarm.Reductors.AllgowerPreconditioner import AllgowerPreconditioner
from pyHarm.Reductors.KrackPreconditioner import KrackPreconditioner
from pyHarm.Reductors.NLdofsReductor import NLdofsReductor

import pandas as pd

Reductor_dico = {
    NoReductor.factory_keyword:                     NoReductor,
    StaticReductor.factory_keyword:                 StaticReductor,
    NLdofsReductor.factory_keyword:                 NLdofsReductor,
    GlobalHarmonicReductor.factory_keyword:         GlobalHarmonicReductor,
    LocalHarmonicReductor.factory_keyword:          LocalHarmonicReductor,
    AllgowerPreconditioner.factory_keyword:         AllgowerPreconditioner,
    KrackPreconditioner.factory_keyword:            KrackPreconditioner,
}
"""dict: Dictionary containing availabe ABCReductors as values and their factoryu keyword as key."""

def generateReductor(data:dict,expl_dofs:pd.DataFrame) -> ABCReductor:
    """
    Factory function that creates a ABCReductor object.

    Args:
        expl_dofs (pd.DataFrame): explicit dofs DataFrame built by the ABCSystem.
        data (dict): dictionary containing the inputs that are needed to create a system.

    Returns:
        ABCReductor: Instance of the required ABCReductor class.
    """
    typeReductor = data["type"]
    reductor = Reductor_dico[typeReductor](data,expl_dofs)
    return reductor