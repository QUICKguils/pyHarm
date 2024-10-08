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

from pyHarm.Elements.SubstructureMatrixElements.GeneralOrderMatrixElement import GOMatrix
import numpy as np 


class LinearHystMatrix(GOMatrix) : 

    factory_keyword = 'linear_hysteretic'

    def _generateMatrices(self, data):
        self.dto = 1
        self.dom = 0
        self.kronMat = np.kron(np.linalg.matrix_power(np.sign(self.nabla),self.dto),data["matrix"])
