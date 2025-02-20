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

from pyHarm.Systems.ABCSystem import ABCSystem
import numpy as np
from pyHarm.Elements.ABCElement import ABCElement
import copy

class System(ABCSystem):
    """Subclass of ABCSystem that is in charge of assembling the residual and jacobians contributions coming from the ABCElements.
    """
    
    factory_keyword : str = "Base"
    """str: keyword that is used to call the creation of this class in the system factory."""

    
    def Residual(self, q:np.ndarray) -> np.ndarray:
        """
        Method that computes the residual vector of the whole system when given a reduced size displacement vector.

        Args:
            q (np.ndarray): reduced size displacement vector (no kinematic conditions).

        Returns:
            np.ndarray: reduced size residual vector.
        """
        x = self._expand_q(q) # expand x to its full size
        x += self._complete_x(self.LC, x) # complete x by applying the kinematic condtions
        Rlin = np.zeros(self.ndofs)
        Rg = np.zeros(self.ndofs)
        Rlin += self._residual(self.LE_extforcing,x)
        Rlin += self._residual(self.LE_linear,x)
        Rg += Rlin
        Rg += self._residual(self.LE_nonlinear_nodlft,x)
        Rg += self._residual(self.LE_nonlinear_dlft,x,**{"Rglin":Rlin})
        Rg += self._complete_R(self.LC, Rg, x) # complete R by transporting the residuals of kinematic dofs
        Rg = self.kick_kc_dofs @ Rg # Reduce R by cutting the kinematic dofs lines
        return Rg
    
    def Jacobian(self, q:np.ndarray) -> tuple[np.ndarray]:
        """
        Method that computes the jacobian matrices of the whole system when given a reduced size displacement vector.

        Args:
            q (np.ndarray): reduced size displacement vector (no kinematic conditions).

        Returns:
            tuple(np.ndarray): reduced size jacobian matrices.
        """
        x = self._expand_q(q)
        x += self._complete_x(self.LC, x)
        # - Initialize the matrices
        dJdxlin = np.zeros((self.ndofs, self.ndofs))
        dJdomlin = np.zeros((self.ndofs, 1))
        dJdx = np.zeros((self.ndofs, self.ndofs))
        dJdom = np.zeros((self.ndofs, 1))

        # - External forcing
        djdx,djdom = self._jacobian(self.LE_extforcing,x)
        dJdxlin += copy.copy(djdx)
        dJdomlin += copy.copy(djdom)

        # - Linear elements
        djdx,djdom = self._jacobian(self.LE_linear,x)
        dJdxlin += copy.copy(djdx)
        dJdomlin += copy.copy(djdom)

        # - Add the linear Jacobian to the full jacobian
        dJdx+=dJdxlin
        dJdom+=dJdomlin
        
        # - Calculation of the nonlinear elements
        djdx,djdom = self._jacobian(self.LE_nonlinear_nodlft,x)
        dJdx+=copy.copy(djdx)
        dJdom+=copy.copy(djdom)
        
        # - Calculation of the DLFT nonlinear elements (requires the residual)
        if len(self.LE_nonlinear_dlft) != 0 :
            Rlin = np.zeros(self.ndofs)
            Rlin += self._residual(self.LE_extforcing,x)
            Rlin += self._residual(self.LE_linear,x)
            djdx,djdom = self._jacobian(self.LE_nonlinear_dlft,x,**{"Rglin":Rlin,"dJgdxlin":dJdxlin,"dJgdomlin":dJdomlin})
            dJdx+=copy.copy(djdx)
            dJdom+=copy.copy(djdom)
            
        jxadd, jomadd = self._complete_J(self.LC, dJdx, dJdom, x)
        dJdx += jxadd
        dJdom += jomadd
        dJdx = self.kick_kc_dofs @ dJdx @ self.kick_kc_dofs.T
        dJdom = self.kick_kc_dofs @ dJdom
        return dJdx,dJdom