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
This module contains the basic MatrixElement class being a abstract class derived from the ABCElement class 
It mostly implements a part of the required abstract methods : the methods that generates the indices from the input datas.
"""
from abc import abstractmethod
from pyHarm.Elements.ABCElement import ABCElement
import pandas as pd
import numpy as np

class MatrixElement(ABCElement) : 

    def __init_data__(self, name, data, CS):
        self.indices = []
        self.name = name
        self.CS = CS
        self.sub = data["connect"]
        self.data = data
        self._generateMatrices(data)

    def __str__(self):
        return f"Matrix Element of type {self.factory_keyword} applied on {self.subs} substructure"

    def generateIndices(self,ed:pd.DataFrame) :
        """From the explicit dof DataFrame, generates the index of dofs concerned by the connector.
        
        Args:
            expl_dofs (pd.DataFrame): explicit dof DataFrame from the studied system.

        Attributes:
            indices (np.ndarray): index of the dofs that the connector needs.
        """
        self.indices = np.array(
            ed[ed['sub']==self.sub].index
        )

    @abstractmethod
    def _generateMatrices(self,data):
        ...
