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
import numpy as np
from numba import njit


@njit(cache=True)
def SubstructureResidual(x,om,kronM,kronC,kronG,kronK):
    R = (om ** 2 * kronM + om * kronC+ om**2 * kronG + kronK) @ x
    return R

@njit(cache=True)
def SubstructureJacobian(x,om,kronM,kronC,kronG,kronK):
    dJdx = (om ** 2 * kronM + om * kronC+ om**2 * kronG + kronK)
    dJdom = (2. * om * kronM + kronC + 2 * om * kronG) @ x
    return dJdx,dJdom


class Substructure(MatrixElement):

    factory_keyword='substructure'

    def _generateMatrices(self, data):
        self.size_mat = data["matrix"]["M"].shape
        self.kronM = np.kron(np.linalg.matrix_power(self.nabla,2),data["matrix"]["M"])
        self.kronC = np.kron(np.linalg.matrix_power(self.nabla,1),data["matrix"]["C"])
        self.kronG = np.kron(np.linalg.matrix_power(self.nabla,1),data["matrix"]["G"])
        self.kronK = np.kron(np.linalg.matrix_power(self.nabla,0),data["matrix"]["K"])

    def get_Mass(self,):
        """
        Extracts Mass matrix from the harmonic domain Mass matrix

        Returns :
            np.ndarray: Mass matrix
        """
        M = -self.kronM[self.size_mat[0]:2*self.size_mat[0], self.size_mat[0]:2*self.size_mat[0]]
        return M

    def get_Rigidity(self,):
        """
        Extracts Rigidity matrix from the harmonic domain Rigidity matrix

        Returns :
            np.ndarray: Rigidity matrix
        """
        K = self.kronK[self.size_mat[0]:2*self.size_mat[0], self.size_mat[0]:2*self.size_mat[0]]
        return K

    def adim(self, lc, wc):
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Attributes:
            kronM (float): modified mass matrix according to the characteristic parameters.
            kronC (float): modified damping matrix according to the characteristic parameters.
            kronG (float): modified gyroscopic matrix according to the characteristic parameters.
            kronK (float): modified rigidity matrix according to the characteristic parameters.
            flag_adim (bool): True if equations are to be adimensionalized.
        """
        self.kronM = self.kronM * lc * (wc**2)
        self.kronC = self.kronC * lc * (wc**1)
        self.kronG = self.kronG * lc * (wc**2)
        self.kronK = self.kronK * lc
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
        self.R = SubstructureResidual(x,om,\
                                      self.kronM,self.kronC,self.kronG,self.kronK)
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
        self.J, self.dJdom = SubstructureJacobian(x,om,\
                                      self.kronM,self.kronC,self.kronG,self.kronK)
        return self.J, self.dJdom

    