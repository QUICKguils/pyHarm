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


from pyHarm.Elements.NodeToNodeElements.GeneralOrderForcing import GeneralOrderForcing
import numpy as np

class CosinusForcing(GeneralOrderForcing): 
    """
    This element is an external forcing that applies a pure cosine signal onto the first harmonic. 
    
    Attributes:
        amp (float): amplitude value of the forcing.
    """
    factory_keyword : str = "CosinusForcing"
    """str: keyword that is used to call the creation of this class in the system factory."""

    def __post_init__(self,):
        self.dto = 0
        self.ho = 1
        self.phi = 0.
        self.loadvec = self._loadingdofs()
        self.amp = self.data["amp"]
        self.nabo = np.linalg.matrix_power(self.nabla,self.dto)
        self.flag_elemtype = -1 # it does not contribute to any of the system matrices