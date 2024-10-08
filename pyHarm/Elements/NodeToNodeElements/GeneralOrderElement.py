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

from numba import njit
import numpy as np
from pyHarm.Elements.NodeToNodeElements.NodeToNodeElement import NodeToNodeElement

@njit(cache=True)
def GOEResidual_noAFT(x,om,Pdir,Pslave,Pmaster,dto,xo,nabo,k):
    R = np.zeros((len(x),))
    for direction in range(Pdir.shape[0]):
        dofs = Pdir[direction,:,:] @ (Pslave - Pmaster) @ x
        force = ((om**dto * nabo) @ (k * dofs ** xo))
        f_harm = (Pslave - Pmaster).T @ Pdir[direction,:,:].T @ force
        R += f_harm
    return R

@njit(cache=True)
def GOEJacobian_noAFT(x,om,Pdir,Pslave,Pmaster,dto,xo,nabo,k):
    dJdx = np.zeros((len(x),len(x)))
    dJdom = np.zeros((len(x),))
    for direction in range(Pdir.shape[0]):
        dofs = Pdir[direction,:,:] @ (Pslave - Pmaster) @ x
        djdx = (Pslave - Pmaster).T @ Pdir[direction,:,:].T @ ((om**dto * nabo) * (k * xo * dofs ** (xo-1))) @ Pdir[direction,:,:] @ (Pslave - Pmaster) 
        if dto != 0:
            djdom =  (Pslave - Pmaster).T @ Pdir[direction,:,:].T @ ((dto*om**(dto-1) * nabo) @ (k * dofs ** xo))
            dJdom += djdom
        dJdx += djdx
    return dJdx,dJdom
    
@njit(cache=True)
def GOEResidual_AFT(x,om,Pdir,Pslave,Pmaster,dto,xo,nabo,k,DFT,DTF):
    R = np.zeros((len(x),))
    for direction in range(Pdir.shape[0]):
        dofs_t = ((om**dto*nabo) @ (Pdir[direction,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
        force_t = (k * dofs_t ** xo)
        f_harm = (Pslave - Pmaster).T @ Pdir[direction,:,:].T @ (force_t @ DTF)
        R += f_harm
    return R

@njit(cache=True)
def GOEJacobian_AFT(x,om,Pdir,Pslave,Pmaster,dto,xo,nabo,k,DFT,DTF):
    dJdx = np.zeros((len(x),len(x)))
    dJdom = np.zeros((len(x),))
    for direction in range(Pdir.shape[0]):
        dofs_t = ((om**dto*nabo) @ (Pdir[direction,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
        djdx = (Pslave - Pmaster).T @ Pdir[direction,:,:].T @ ( DFT @ ((k * xo * dofs_t ** (xo-1)).reshape(-1,1) * DTF)).T @ Pdir[direction,:,:] @ (Pslave - Pmaster) 
        if dto != 0:
            omderiv_dofs_t = ((dto*om**(dto-1.0) * nabo) @ (Pdir[direction,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
            djdom =  (Pslave - Pmaster).T @ Pdir[direction,:,:].T @ ((k * omderiv_dofs_t ** xo) @ DTF)
            dJdom += djdom
        dJdx += djdx
    return dJdx,dJdom
    

class GeneralOrderElement(NodeToNodeElement): 
    """
    This element is the general polynomial element, it can apply any required polynomial link in displacement and in derivative order. 
    
    Attributes:
        dto (int): order of the time derivative.
        xo (int): order of the power to apply to the displacement.
        k (float): linear factor to apply.
    """
    factory_keyword : str = "GOElement"
    """str: keyword that is used to call the creation of this class in the system factory."""

    def __post_init__(self,):
        self.dto = self.data["dto"]
        self.xo = self.data["xo"]
        self.k = self.data["k"]
        self.nabo = np.linalg.matrix_power(self.nabla,self.dto)
        if self.xo == 1:
            self.flag_elemtype = self.dto
        else:
            self.flag_elemtype = -1

    def __flag_update__(self) : 
        if self.xo != 1:
            self.flag_nonlinear = True
            self.flag_AFT = True

    def _evalResidual(self, x, om):
        if not self.flag_AFT : 
            R = GOEResidual_noAFT(x,om,\
                             self.Pdir,self.Pslave,self.Pmaster,\
                             self.dto,self.xo,self.nabo,self.k)
        elif self.flag_AFT : 
            R = GOEResidual_AFT(x,om,\
                             self.Pdir,self.Pslave,self.Pmaster,\
                             self.dto,self.xo,self.nabo,self.k,self.D["ft"],self.D["tf"])
        return R
    
    def _evalJacobian(self, x, om):
        if not self.flag_AFT : 
            dJdx,dJdom = GOEJacobian_noAFT(x,om,\
                             self.Pdir,self.Pslave,self.Pmaster,\
                             self.dto,self.xo,self.nabo,self.k)
        elif self.flag_AFT : 
            dJdx,dJdom = GOEJacobian_AFT(x,om,\
                             self.Pdir,self.Pslave,self.Pmaster,\
                             self.dto,self.xo,self.nabo,self.k,self.D["ft"],self.D["tf"])
        return dJdx,dJdom
    
    def adim(self, lc, wc):
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Attributes:
            k (float): modified linear coefficient to apply according to the characteristic parameters.
            flag_adim (bool): True if equations are to be adimensionalized.
        """
        self.k = self.k * (lc**self.xo) * (wc**self.dto)
        self.flag_adim = True

    def evalResidual(self, xg, om):
        """Computes the residual.
        
        Args:
            x (np.ndarray): full displacement vector.
            om (float): angular frequency value.

        Returns:
            np.ndarray: residual vector.
        """
        x = xg[self.indices]
        self.R = self._evalResidual(x, om)
        return self.R
    
    def evalJacobian(self, xg, om):
        """Computes the jacobians.
        
        Args:
            x (np.ndarray): full displacement vector.
            om (float): angular frequency value.

        Returns:
            np.ndarray: Jacobian with respect to displacement.
            np.ndarray: Jacobian with respect to angular frequency.
        """
        x = xg[self.indices]
        self.J,self.dJdom = self._evalJacobian(x, om)
        return self.J,self.dJdom
