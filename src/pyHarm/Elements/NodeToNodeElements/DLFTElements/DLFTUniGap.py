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
from numba import njit
import numpy as np
import copy
import jax
import jax.numpy as jnp

@jax.jit
def DLFTUniGapResidual_jax(x, om, Rlin, nbSub, Pdir, Pslave, Pmaster, g, eps, DFT, DTF, N0):
    R = jnp.zeros((len(x),))
    nti = jnp.shape(DFT)[1]
    for dir1 in range(jnp.shape(Pdir)[0]):
        rlin_part = -(1/nbSub*Pdir[dir1,:,:] @ (Pslave - Pmaster) @ Rlin)
        eps_part = eps * (((Pdir[dir1,:,:] @ (Pslave - Pmaster) @ x)) @ DFT - g) @ DTF
        lambda_opt_t = ( rlin_part + eps_part ) @ DFT
        lambda_opt_t = lambda_opt_t * ((lambda_opt_t-N0)<0)
        R += (Pslave - Pmaster).T @ Pdir[dir1,:,:].T @ (lambda_opt_t @ DTF)
    return R, lambda_opt_t

jaxf = jax.jacfwd(DLFTUniGapResidual_jax)
jaxf = jax.jit(jaxf)

@njit(cache=True)
def DLFTUniGapResidual(x, om, Rlin, nbSub, Pdir, Pslave, Pmaster, g, eps, DFT, DTF, N0):
    R = np.zeros((len(x),))
    nti = DFT.shape[1]
    for dir1 in range(Pdir.shape[0]):
        rlin_part = -(1/nbSub*Pdir[dir1,:,:] @ (Pslave - Pmaster) @ Rlin)
        eps_part = eps * (((Pdir[dir1,:,:] @ (Pslave - Pmaster) @ x)) @ DFT - g) @ DTF             
        lambda_opt_t = ( rlin_part + eps_part ) @ DFT
        lambda_opt_t = lambda_opt_t * ((lambda_opt_t-N0)<0)
        R += (Pslave - Pmaster).T @ Pdir[dir1,:,:].T @ (lambda_opt_t @ DTF)
    return R


@njit(cache=True)
def DLFTUniGapJacobian(x, om, Rlin, dJdxlin, dJdomlin, nbSub, Pdir, Pslave, Pmaster, g, eps, DFT, DTF):
    dJdx = np.zeros((len(x),len(x)))
    dJdom = np.zeros((len(x),))
    nti = DFT.shape[1]
    for dir1 in range(Pdir.shape[0]):
        rlin_part = -(1/nbSub * Pdir[dir1,:,:] @ (Pslave - Pmaster) @ Rlin)
        eps_part = eps * (((Pdir[dir1,:,:] @ (Pslave - Pmaster) @ x)) @ DFT - g) @ DTF             
        lambda_opt_t = ( rlin_part + eps_part ) @ DFT
        overshoot = np.where(lambda_opt_t<0)
        dlam_opt_t = ((1/nbSub * dJdxlin @ (Pslave - Pmaster).T @ Pdir[dir1,:,:].T) + eps*((Pslave-Pmaster).T@Pdir[dir1,:,:].T)) @ DFT
        for o in overshoot :
            dlam_opt_t[:,o] = 0
        dlam_h = dlam_opt_t@DTF@Pdir[dir1,:,:]
        dJdx += ((Pslave-Pmaster).T @ dlam_h.T)
    return dJdx,dJdom

class DLFTUniGap(DLFTElement) : 
    """
    This element is the DLFT unilateral gap element. 
    
    Attributes:
        N0 (float): normal force applied at contact.
        g (float): gap value.
        eps (foat): penalty parameter of the DLFT method.
        jac (str): if "analytical" uses analytical expression of jacobian, if "jax" uses automatic differentiation, if "DF" uses finite difference.
    """
    factory_keyword : str = "DLFTUniGap"
    """str: keyword that is used to call the creation of this class in the system factory."""
    def __post_init__(self,):
        self.g = self.data["g"]
        self.eps = self.data["eps"]
        self.N0 = self.data["N0"]
        self.nabo = np.linalg.matrix_power(self.nabla,0)
        if "jac" in self.data.keys():
            self.jac = self.data["jac"]
        else:
            self.jac = "analytical"

    def _evalJaco_ana(self, xg, om, Rglin, dJgdxlin, dJgdomlin,*args):
        x = xg[self.indices]
        Rlin = Rglin[self.indices]
        dJdxlin = dJgdxlin[tuple(np.meshgrid(self.indices,self.indices))]
        dJdomlin = dJgdomlin[self.indices]
        dJdx,dJdom = DLFTUniGapJacobian(x, om, Rlin, dJdxlin, dJdomlin, float(self.nbSub),\
                                        self.Pdir, self.Pslave, self.Pmaster,\
                                        self.g, self.eps,\
                                        self.D["ft"],self.D["tf"])
        self.J = copy.copy(dJdx)
        self.dJdom = copy.copy(dJdom)
        return dJdx, dJdom
    
    def _evalJaco_jax(self, xg, om, Rglin=None, dJgdxlin=None, dJgdomlin=None):
        x = xg[self.indices]
        Rlin = Rglin[self.indices]
        self.dJdom = np.zeros((len(x),))
        self.dJdx = jaxf(x, om, Rlin, float(self.nbSub), self.Pdir, self.Pslave, self.Pmaster,\
                             self.g, self.eps,\
                             self.D["ft"],self.D["tf"], self.N0)    
        return self.dJdx, self.dJdom
    
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
        self.R,_ = DLFTUniGapResidual(x, om, Rlin, float(self.nbSub), self.Pdir, self.Pslave, self.Pmaster,\
                             self.g, self.eps,\
                             self.D["ft"],self.D["tf"], self.N0)
        return self.R
    
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
        if self.jac == "analytical":
            dJdx, dJdom = self._evalJaco_ana(xg, om, Rglin, dJgdxlin, dJgdomlin, 1e-5)
        elif self.jac == "jax":
            dJdx, dJdom = self._evalJaco_jax(xg, om, Rglin, dJgdxlin, dJgdomlin)
        self.J = dJdx
        self.dJdom = dJdom
        return self.J,self.dJdom
    
