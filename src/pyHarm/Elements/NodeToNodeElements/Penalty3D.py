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
from pyHarm.Elements.NodeToNodeElements.Jenkins import JenkinsResidual_jax
from pyHarm.Elements.NodeToNodeElements.PenaltyUnilateralGap import PenalGapResidual_jax
import numpy as np
import copy
from jax import jacfwd, jit

@jit
def evalResidual_jax(x, om, Pdir, Pslave, Pmaster, mu, N0, g, k_n, k_t, DFT, DTF):

    # Normal contact
    R_N, f_time_n = PenalGapResidual_jax(x, om, Pdir[[0],:,:], Pslave, Pmaster, g, k_n, DFT, DTF, N0)
    
    # Tangential contact
    R_T = JenkinsResidual_jax(x, om, Pdir[[1,2],:,:], Pslave, Pmaster, mu, N0, k_t, DFT, DTF, f_time_n)

    return R_N + R_T

jaxf = jacfwd(evalResidual_jax)
jaxf = jit(jaxf)


class Penalty3D(NodeToNodeElement) : 
    """
    This element is a combination of an unilateral gap element and a jenkins element on the perpendicular directions when contact is made. 
    
    Attributes:
        g (float): gap value.
        k_n (foat): linear spring value for the unilateral gap.
        k_t (foat): linear spring value for the jenkins.
        mu (float): friction coefficient.
        N0 (float): normal preload on the element.
        jac (str): if "analytical" uses analytical expression of jacobian, if "jax" uses automatic differentiation, if "DF" uses finite difference.
    """
    factory_keyword : str = "Penalty3D"
    """str: keyword that is used to call the creation of this class in the system factory."""
    def __post_init__(self,):
        self.mu = self.data["mu"]
        self.N0 = self.data["N0"]
        self.k_n = self.data["k_n"]
        self.k_t = self.data["k_t"]
        self.g = self.data["g"]
        self.nabo = np.linalg.matrix_power(self.nabla,0)

    def __flag_update__(self,):
        self.flag_nonlinear = True
        self.flag_AFT = True
    
    def adim(self, lc, wc):
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Attributes:
            k_n (float): modified linear coefficient for the unilateral gap to apply according to the characteristic parameters.
            k_n (float): modified linear coefficient for the jenkins to apply according to the characteristic parameters.
            g (float): modified gap value according to the characteristic parameters.
            flag_adim (bool): True if equations are to be adimensionalized
        """
        self.k_n = self.k_n * lc
        self.k_t = self.k_t * lc
        self.g = self.g / lc
        self.flag_adim = True

    def evalResidual(self, xg, om):
        """Computes the residual.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.

        Returns:
            np.ndarray: residual vector.
        """
        x = xg[self.indices] 
        self.R = evalResidual_jax(x, om, self.Pdir, self.Pslave, self.Pmaster, self.mu, self.N0, self.g, self.k_n, self.k_t, self.D["ft"],self.D["tf"])
        return self.R 
    
    def _evalJaco_ana(self, xg, om):
        x = xg[self.indices]
        dJdom = np.zeros((len(x),))
        dJdx = jaxf(x, om, self.Pdir, self.Pslave, self.Pmaster, self.mu, self.N0, self.g, self.k_n, self.k_t, self.D["ft"],self.D["tf"])    
        self.J = copy.copy(dJdx)
        self.dJdom = copy.copy(dJdom)
        return dJdx, dJdom
    
    def evalJacobian(self, xg, om):
        """Computes the jacobians.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.

        Returns:
            np.ndarray: Jacobian with respect to displacement.
            np.ndarray: Jacobian with respect to angular frequency.
        """
        dJdx, dJdom = self._evalJaco_ana(xg, om)
        self.J = dJdx
        self.dJdom = dJdom
        return self.J,self.dJdom