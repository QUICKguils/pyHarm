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
import jax
import jax.numpy as jnp

@jax.jit
def PenalGapResidual_jax(x, om, Pdir, Pslave, Pmaster, g, k, DFT, DTF, N0=0.):
    R = jnp.zeros((len(x),))
    nti = jnp.shape(DFT)[1]
    x_d = jnp.zeros((jnp.shape(Pdir)[0],nti))
    for dir1 in range(jnp.shape(Pdir)[0]):
        x_d = x_d.at[dir1,:].set(((Pdir[dir1,:,:] @ (Pslave - Pmaster) @ x)) @ DFT)
    for dir1 in range(jnp.shape(Pdir)[0]):
        f_time = jnp.zeros((nti,))
        x_d1 = x_d[dir1,:]

        def true_fn(x_d1, g, N0, k):
            return x_d1>=g, g
        def false_fn(x_d1, g, N0, k):
            return x_d1<=N0/k, 0.
        contact, g = jax.lax.cond(N0 == 0., true_fn, false_fn, x_d1, g, N0, k)

        def function_to_loop(t, state):
            f_time = state 
            def true_f(k, x_d1, g, i): 
                return k * (x_d1[i]-g)
            def false_f(k, x_d1, g, i):
                return 0.
            f_time = f_time.at[t].set(jax.lax.cond(contact[t], true_f, false_f, k, x_d1, g, t))
            return f_time
    
        initial_state = f_time
        final_state = jax.lax.fori_loop(0, nti, function_to_loop, initial_state)
        f_time = final_state

        R+=(Pslave - Pmaster).T @ Pdir[dir1,:,:].T @ (f_time @ DTF)
    return R, f_time

@njit(cache=True)
def PenalGapResidual(x, om, Pdir, Pslave, Pmaster, g, k, DFT, DTF, N0=0.):
    R = np.zeros((len(x),))
    nti = DFT.shape[1]
    x_d = np.zeros((Pdir.shape[0],nti))
    for dir1 in range(Pdir.shape[0]):
        x_d[dir1,:] = ((Pdir[dir1,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
    for dir1 in range(Pdir.shape[0]):
        f_time = np.zeros((nti,))
        x_d1 = x_d[dir1,:]
        if N0==0.: # gap imposé
            contact = np.where(x_d1>=g)
        else: # effort normal imposé
            contact = np.where(x_d1<=N0/k)
            g = 0
        f_time[contact] = k * (x_d1[contact]-g)
        R+=(Pslave - Pmaster).T @ Pdir[dir1,:,:].T @ (f_time @ DTF)
    return R, f_time


@njit(cache=True)
def PenalGapJacobian(x, om, Pdir, Pslave, Pmaster, g, k, DFT, DTF, N0=0.):
    dJdx = np.zeros((len(x),len(x)))
    dJdom = np.zeros((len(x),))
    nti = DFT.shape[1]
    x_d = np.zeros((Pdir.shape[0],nti))
    for dir1 in range(Pdir.shape[0]):
        x_d[dir1,:] = ((Pdir[dir1,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
    r = np.sqrt(np.sum(x_d**2,axis=0))
    for dir1 in range(Pdir.shape[0]):
        x_k = x_d[dir1,:]
        if N0==0.:
            contact = np.where(x_k>=g)
        else:
            contact = np.where(x_k<=N0/k)
        df_time = np.zeros(len(x_k))
        df_time[contact] = k
        dRdx = (Pslave - Pmaster).T @ Pdir[dir1,:,:].T @ (DFT*df_time @ DTF).T @ Pdir[dir1,:,:] @ (Pslave - Pmaster)
        dJdx += dRdx
    return dJdx,dJdom
    

class PenaltyUnilateralGap(NodeToNodeElement): 
    """
    This element is an approximation of unilateral contact by adding a rigidity when contact is made. 
    
    Attributes:
        g (float): gap value.
        k (foat): linear spring value.
        jac (str): if "analytical" uses analytical expression of jacobian, if "jax" uses automatic differentiation, if "DF" uses finite difference.
    """
    factory_keyword : str = "PenaltyUnilateralGap"
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
        self.R, _ = PenalGapResidual(x, om, self.Pdir, self.Pslave, self.Pmaster,\
                             self.g, self.k,\
                             self.D["ft"],self.D["tf"])
        return self.R
    
    def evalJacobian(self, xg, om):
        """Compute the jacobians.
        
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