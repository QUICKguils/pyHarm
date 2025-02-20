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

from pyHarm.Elements.NodeToNodeElements.DLFTElements.DLFTElement import DLFTElement
from pyHarm.Elements.NodeToNodeElements.DLFTElements.DLFTFriction import DLFTFrictionResidual_jax
from pyHarm.Elements.NodeToNodeElements.DLFTElements.DLFTUniGap import DLFTUniGapResidual_jax
from numba import njit
import numpy as np
import copy
from jax import jacfwd, jit

@jit
def evalResidual_jax(x, om, Rlin, nbSub, Pdir, Pslave, Pmaster, mu, N0, g, eps, DFT, DTF):
    """:meta private:"""

    # Normal contact
    R_N, f_time_n = DLFTUniGapResidual_jax(x, om, Rlin, nbSub, Pdir[[0],:,:], Pslave, Pmaster, g, eps, DFT, DTF, N0)
    
    # Tangential contact
    R_T, _ = DLFTFrictionResidual_jax(x, om, Rlin, nbSub, Pdir[[1,2],:,:], Pslave, Pmaster, mu, N0, eps, DFT, DTF, f_time_n)

    return R_N + R_T

jaxf = jacfwd(evalResidual_jax)
jaxf = jit(jaxf)
""":meta private:"""


class DLFT3D(DLFTElement) : 
    """
    This element is a combination of the unilateral DFLT gap element and the DLFT friction element in the perpendicular directions. 
    
    Attributes:
        N0 (float): normal force applied at contact.
        g (float): gap value.
        eps (foat): penalty parameter of the DLFT method.
        mu (foat): friction coefficient.
        jac (str): if "analytical" uses analytical expression of jacobian, if "jax" uses automatic differentiation, if "DF" uses finite difference.
    """

    factory_keyword : str = "DLFT3D"
    """str: keyword that is used to call the creation of this class in the system factory."""

    def __post_init__(self,):
        self.mu = self.data["mu"]
        self.N0 = self.data["N0"]
        self.eps = self.data["eps"]
        self.g = self.data["g"]
        self.nabo = np.linalg.matrix_power(self.nabla,0)
    
    def adim(self, lc, wc):
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Attributes:
            eps (float): modified penalty coefficient to apply according to the characteristic parameters.
            g (float): modified gap value according to the characteristic parameters.
            flag_adim (bool): True if equations are to be adimensionalized
        """
        self.eps = self.eps * lc
        self.g = self.g / lc
        self.flag_adim = True

    def evalResidual(self, xg, om, Rglin=None):
        """Computes the residual.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.
            Rglin (np.ndarray): Residual vector of all the linear element contributions.

        Returns:
            np.ndarray: residual vector.
        """
        x = xg[self.indices] 
        Rlin = Rglin[self.indices]
        self.R = evalResidual_jax(x, om, Rlin, float(self.nbSub), self.Pdir, self.Pslave, self.Pmaster,\
                             self.mu, self.N0, self.g, self.eps,\
                             self.D["ft"],self.D["tf"])
        _, f_time_n = DLFTUniGapResidual_jax(x, om, Rlin, float(self.nbSub), self.Pdir[[0],:,:], self.Pslave, self.Pmaster,\
                            self.g, self.eps, self.D["ft"],self.D["tf"], self.N0)
        self.sep = (f_time_n-self.N0)>0
        _, slip = DLFTFrictionResidual_jax(x, om, Rlin, float(self.nbSub), self.Pdir[[1,2],:,:], self.Pslave, self.Pmaster, self.mu, self.N0, self.eps, self.D["ft"], self.D["tf"], f_time_n)
        self.slip = slip
        return self.R 
    
    def _evalJaco_ana(self, xg, om, Rglin=None):
        x = xg[self.indices]
        Rlin = Rglin[self.indices]
        dJdom = np.zeros((len(x),))
        dJdx = jaxf(x, om, Rlin, float(self.nbSub), self.Pdir, self.Pslave, self.Pmaster,\
                             self.mu, self.N0, self.g, self.eps,\
                             self.D["ft"],self.D["tf"])    
        self.J = copy.copy(dJdx)
        self.dJdom = copy.copy(dJdom)
        return dJdx, dJdom
    
    def evalJacobian(self, xg, om, Rglin=None, dJgdxlin=None, dJgdomlin=None):
        """Computes the jacobians.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.
            Rglin (np.ndarray): Residual vector of all the linear element contributions.
            dJgdxlin (np.ndarray): Jacobian with respect to displacement of all the linear element contributions.
            dJgdomlin (np.ndarray): Jacobian with respect to angular frequency of all the linear element contributions.

        Returns:
            np.ndarray: Jacobian with respect to displacement.
            np.ndarray: Jacobian with respect to angular frequency.
        """
        dJdx, dJdom = self._evalJaco_ana(xg, om, Rglin)
        self.J = dJdx
        self.dJdom = dJdom
        return self.J,self.dJdom