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

from pyHarm.Elements.NodeToNodeElements.NodeToNodeElement import NodeToNodeElement
import numpy as np
import copy


class DLFTElement(NodeToNodeElement):
    """
    This element is modification of the ABCElement in order to take into account the required modifications needed when using DLFT elements. 
    
    Attributes:
        flag_nonlinear (bool): if True, the element is nonlinear.
        flag_AFT (bool): if True, the element requires an alternating frequency/time domain procedure for computing residuals.
        flag_DLFT (bool): if True, the element uses the dynamic Lagrangian method for computing the residuals.
    """
    
    factory_keyword : str = "DLFTElement"
    """str: keyword that is used to call the creation of this class in the system factory."""

    def __flag_update__(self,):
        self.flag_nonlinear = True
        self.flag_AFT = True
        self.flag_DLFT = True

    def adim(self,):
        pass
        
    def _evalJaco_DF(self, xg, om, Rglin, dJgdxlin, dJgdomlin, step):
        """Computes the jacobian using finite difference method for DLFT Elements.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.
            Rglin (float): Residual of all the linear contributions.
            dJgdxlin (float): Jacobian with respect to displacement of all the linear contributions.
            dJgdomlin (float): Jacobian with respect to angular frequency of all the linear contributions.
            step (float): step size for the finite difference method.

        Returns:
            dJdx (np.ndarray): jacobian with respect to displacement vector.
            dJdom (np.ndarray): jacobian with respect to angular frequency.
        """
        R_init = self.evalResidual(xg, om, Rglin)
        dJdx = np.zeros((len(self.indices), len(self.indices)))
        dJdom = np.zeros((len(self.indices),1))
        for kk,idid in enumerate(self.indices) : 
            x_m = copy.copy(xg)
            x_m[idid] += step
            R_idid = self.evalResidual(x_m, om, (Rglin+step*dJgdxlin[idid,:]))
            dJdx[:,kk] = (R_idid - R_init) / step
        R_om = self.evalResidual(xg, om+step, (Rglin+step*dJgdomlin[:,0]))
        dJdom[:,0] = (R_om - R_init) / step
        self.J = copy.copy(dJdx)
        self.dJdom = copy.copy(dJdom)
        return dJdx, dJdom
