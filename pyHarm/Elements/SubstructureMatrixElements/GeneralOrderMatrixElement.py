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

from pyHarm.Elements.SubstructureMatrixElements.MatrixElement import MatrixElement
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
import numpy as np
from numba import njit

@njit(cache=True)
def GOMatrixResidual(x,om,kronMat,dom):
    R = (om ** (dom) * kronMat) @ x
    return R

@njit(cache=True)
def GOMatrixJacobian(x,om,kronMat,dom):
    dJdx = (om ** (dom) * kronMat)
    dJdom = np.zeros(x.shape)
    if dom != 0 :
        dJdom = (dom * om ** (dom-1) * kronMat) @ x
    return dJdx,dJdom


class GOMatrix(MatrixElement):

    factory_keyword='GOMatrix'
    default_GOMatrix = {'dto':0}

    def _generateMatrices(self, data):
        data = getCustomOptionDictionary(data,self.default_GOMatrix)
        if 'dom' not in data :
            data['dom'] = data['dto']
        self.dto = data['dto']
        self.dom = data['dom']
        self.kronMat = np.kron(np.linalg.matrix_power(self.nabla,self.dto),data["matrix"])
        self.flag_elemtype = self.dto

    def adim(self, lc, wc):
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Attributes:
            kronM (float): modified mass matrix according to the characteristic parameters.
            kronC (float): modified damping matrix according to the characteristic parameters.
            kronG (float): modified gyroscopic matrix according to the characteristic parameters.
            kronK (float): modified rigidity matrix according to the characteristic parameters.
            flag_adim (bool): True if equations are to be adimensionalized.
        """
        self.kronMat = self.kronMat * lc * (wc**self.dom)
        self.flag_adim = True

    def evalResidual(self,xg,om):
        """Computes the residual.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.

        Returns:
            np.ndarray: residual vector.
        """
        x = xg[self.indices]
        self.R = GOMatrixResidual(x,om,self.kronMat,self.dom)
        return self.R
    
    def evalJacobian(self,xg,om):
        """Computes the jacobians.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.

        Returns:
            np.ndarray: Jacobian with respect to displacement.
            np.ndarray: Jacobian with respect to angular frequency.
        """
        x = xg[self.indices]
        self.J, self.dJdom = GOMatrixJacobian(x,om,self.kronMat,self.dom)
        return self.J, self.dJdom