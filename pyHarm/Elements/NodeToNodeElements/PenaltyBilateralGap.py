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
from numba import njit
import numpy as np
import copy

@njit(cache=True)
def PenalGapResidual(x, om, Pdir, Pslave, Pmaster, g, k, DFT, DTF):
    R = np.zeros((len(x),))
    nti = DFT.shape[1]
    x_d = np.zeros((Pdir.shape[0],nti))
    for dir1 in range(Pdir.shape[0]):
        x_d[dir1,:] = ((Pdir[dir1,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
    r = np.sqrt(np.sum(x_d**2,axis=0))
    gap_closed = np.where(r>=g)
    for dir1 in range(Pdir.shape[0]):
        f_time = np.zeros((nti,))
        x_d1 = x_d[dir1,:]
        f_time[gap_closed] = k * (r[gap_closed]-g) * x_d1[gap_closed]/(r[gap_closed]+1e-12)
        R+=(Pslave - Pmaster).T @ Pdir[dir1,:,:].T @ (f_time @ DTF)
    return R



@njit(cache=True)
def PenalGapJacobian(x, om, Pdir, Pslave, Pmaster, g, k, DFT, DTF):
    dJdx = np.zeros((len(x),len(x)))
    dJdom = np.zeros((len(x),))
    nti = DFT.shape[1]
    x_d = np.zeros((Pdir.shape[0],nti))
    for dir1 in range(Pdir.shape[0]):
        x_d[dir1,:] = ((Pdir[dir1,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
    r = np.sqrt(np.sum(x_d**2,axis=0))
    gap_closed = np.where(r>=g)
    for dir1 in range(Pdir.shape[0]):
        x_k = x_d[dir1,:]
        for dir2 in range(Pdir.shape[0]):
            dRkdxj = np.zeros((nti,))
            x_j = x_d[dir2,:]
            if dir1 == dir2 :
                kronecker = 1
            else : 
                kronecker = 0 
            dRkdxj[gap_closed] = k*(1-g/(r[gap_closed]+1e-12))*kronecker + k*g* x_k[gap_closed]*x_j[gap_closed]/(r[gap_closed]**3+1e-12)
            dJdx += (Pslave - Pmaster).T@Pdir[dir1,:,:].T@(DFT*dRkdxj@DTF).T@Pdir[dir2,:,:]@(Pslave - Pmaster)
    return dJdx,dJdom
    
class PenaltyBilateralGap(NodeToNodeElement):  
    """
    This element is an approximation of bilateral contact by adding a rigidity when contact is made. 
    
    Attributes:
        g (float): gap value.
        k (float): linear spring value.
        jac (str): if "analytical" uses analytical expression of jacobian, if "jax" uses automatic differentiation, if "DF" uses finite difference.
    """
    factory_keyword : str = "PenaltyBilateralGap"
    """str: keyword that is used to call the creation of this class in the system factory."""
    def __post_init__(self,):
        self.g = self.data["g"]
        self.k = self.data["k"]
        self.nabo = np.linalg.matrix_power(self.nabla,0)

    def __flag_update__(self,):
        self.flag_nonlinear = True
        self.flag_AFT = True

    def _evalJaco_DF(self, xg, om, step):
        R_init = self.evalResidual(xg, om)
        dJdx = np.zeros((len(self.indices), len(self.indices)))
        dJdom = np.zeros((len(self.indices),1))
        for kk,idid in enumerate(self.indices) : 
            x_m = copy.copy(xg)
            x_m[idid] += step
            R_idid = self.evalResidual(x_m, om)
            dJdx[:,kk] = (R_idid - R_init) / step
        R_om = self.evalResidual(xg, om+step)
        dJdom[:,0] = (R_om - R_init) / step
        self.J = copy.copy(dJdx)
        self.dJdom = copy.copy(dJdom)
        return dJdx, dJdom
    
    def _evalJaco_ana(self, xg, om):
        x = xg[self.indices]
        dJdx,dJdom = PenalGapJacobian(x, om, self.Pdir, self.Pslave, self.Pmaster,\
                             self.g, self.k,\
                             self.D["ft"],self.D["tf"])
        self.J = copy.copy(dJdx)
        self.dJdom = copy.copy(dJdom)
        return dJdx, dJdom
    
    def adim(self, lc, wc):
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Attributes:
            k (float): modified linear coefficient to apply according to the characteristic parameters.
            g (float): modified gap value according to the characteristic parameters.
            flag_adim (bool): True if equations are to be adimensionalized.
        """
        self.g = self.g / lc
        self.k = self.k * lc
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
        self.R = PenalGapResidual(x, om, self.Pdir, self.Pslave, self.Pmaster,\
                             self.g, self.k,\
                             self.D["ft"],self.D["tf"])
        return self.R
    
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