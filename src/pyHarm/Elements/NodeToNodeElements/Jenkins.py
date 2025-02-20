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
def JenkinsCorLoop_jax(x_k, k, DFT, limit_friction_force):
    z = jnp.zeros(jnp.shape(x_k))
    z0 = jnp.zeros((jnp.shape(x_k)[0],))
    fpre = jnp.zeros(jnp.shape(x_k))
    nti = jnp.shape(DFT)[1]

    def function_to_loop(t, state):
        z0, z, fpre = state
        def true_fn(x_k, z0, k, limit_friction_force, delta_x, dx_norm):
            v = delta_x / dx_norm
            z0 = x_k[:,t] - limit_friction_force[t]*v/k
            return z0
        def false_fn(x_k, z0, k, limit_friction_force, delta_x, dx_norm):
            return z0
        fpre = fpre.at[:, t].set(k*(x_k[:,t]-z0))
        slip = jnp.sqrt(jnp.sum(fpre[:,t]**2)) >= limit_friction_force[t]
        delta_x = x_k[:,t]-z0
        dx_norm =  jnp.sqrt(jnp.sum(delta_x**2))
        z0 = jax.lax.cond(slip, true_fn, false_fn, x_k, z0, k, limit_friction_force, delta_x, dx_norm)
        z = z.at[:,t].set(z0)
        return (z0, z, fpre)

    for _ in range(3) :
        initial_state = (z0, z, fpre)
        final_state = jax.lax.fori_loop(0, nti, function_to_loop, initial_state)
        z0, z, fpre = final_state

    return z

@jax.jit
def JenkinsResidual_jax(x, om, Pdir, Pslave, Pmaster, mu, N0, k, DFT, DTF, f_time_n=0.):
    limit_friction_force = mu * jnp.abs(f_time_n - N0 * jnp.ones(jnp.shape(DFT)[1]))
    R = jnp.zeros((len(x),))
    nti = jnp.shape(DFT)[1]
    x_k = jnp.zeros((jnp.shape(Pdir)[0],nti))
    for dir_k in range(jnp.shape(Pdir)[0]):
        x_k = x_k.at[dir_k,:].set(((Pdir[dir_k,:,:] @ (Pslave - Pmaster) @ x)) @ DFT)
    z = JenkinsCorLoop_jax(x_k, k, DFT, limit_friction_force)
    f_time = k*(x_k-z) * ((f_time_n - N0 * jnp.ones(jnp.shape(DFT)[1]))<=0)
    for dir1 in range(jnp.shape(Pdir)[0]):
        R+=(Pslave - Pmaster).T @ Pdir[dir1,:,:].T @ (f_time[dir1,:] @ DTF)
    return R

jaxf = jax.jacfwd(JenkinsResidual_jax)
jaxf = jax.jit(jaxf)

@njit(cache=True)
def JenkinsCorLoop(x_k, k, dx_kdx_j, DFT, limit_friction_force):
    z = np.zeros(x_k.shape)
    z0 = np.zeros((x_k.shape[0],))
    dz_kdx_j = np.zeros(dx_kdx_j.shape)
    dz0 = np.zeros((dx_kdx_j.shape[0], dx_kdx_j.shape[1], dx_kdx_j.shape[3]))
    fpre = np.zeros(x_k.shape)
    nti = DFT.shape[1]
    stuck_all = []

    # Here we need to loop 3 times
    for period in range(3):
        for t in range(nti): 
            fpre[:,t] = k*(x_k[:,t]-z0)
            slip = np.sqrt(np.sum(fpre[:,t]**2)) >= limit_friction_force[t]
            delta_x = x_k[:,t]-z0
            dx_norm =  np.sqrt(np.sum(delta_x**2))
            ddelta_xdx = dx_kdx_j[:,:,t,:]-dz0
            if slip:
                v = delta_x / dx_norm # if slip, dx_norm is different than 0
                z0 = x_k[:,t] - limit_friction_force[t]*v/k
                for dir_j in range(x_k.shape[0]):
                    dvdx_j = (ddelta_xdx[:,dir_j,:])/dx_norm -(delta_x.reshape(-1,1) @ (delta_x @ ddelta_xdx[:,dir_j,:]).reshape(1,-1))/dx_norm**3
                    dz0[:,dir_j,:] = dx_kdx_j[:,dir_j,t,:] - limit_friction_force[t]*dvdx_j/k
            if period == 2: # if last period, then store results
                z[:,t] = z0
                dz_kdx_j[:,:,t,:] = dz0
                stuck_all.append(not(slip))
    return z, dz_kdx_j, stuck_all

@njit(cache=True)
def JenkinsResidual(x, om, Pdir, Pslave, Pmaster, mu, N0, k, DFT, DTF):
    limit_friction_force = mu * N0 * np.ones(DFT.shape[1])
    R = np.zeros((len(x),))
    nti = DFT.shape[1]
    nh = DFT.shape[0]
    dx_kdx_j = np.zeros((Pdir.shape[0],Pdir.shape[0],nti,nh))
    x_k = np.zeros((Pdir.shape[0],nti))
    for dir_k in range(Pdir.shape[0]):
        x_k[dir_k,:] = ((Pdir[dir_k,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
    for dir_k in range(Pdir.shape[0]):
        for dir_j in range(Pdir.shape[0]):
            kronecker = 1*(dir_k==dir_j)
            dx_kdx_j[dir_k,dir_j,:,:] = kronecker * DFT.T
    z, _, stuck = JenkinsCorLoop(x_k, k, dx_kdx_j, DFT, limit_friction_force)
    f_time = k*(x_k-z)
    for dir1 in range(Pdir.shape[0]):
        R+=(Pslave - Pmaster).T @ Pdir[dir1,:,:].T @ (f_time[dir1,:] @ DTF)
    return R, stuck


@njit(cache=True)
def JenkinsJacobian(x, om, Pdir, Pslave, Pmaster, mu, N0, k, DFT, DTF):
    limit_friction_force = mu * N0 * np.ones(DFT.shape[1])
    dJdx = np.zeros((len(x),len(x)))
    dJdom = np.zeros((len(x),))
    nti = DFT.shape[1]
    nh = DFT.shape[0]
    dx_kdx_j = np.zeros((Pdir.shape[0],Pdir.shape[0],nti,nh))
    x_k = np.zeros((Pdir.shape[0],nti))
    for dir_k in range(Pdir.shape[0]):
        x_k[dir_k,:] = ((Pdir[dir_k,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
    for dir_k in range(Pdir.shape[0]):
        for dir_j in range(Pdir.shape[0]):
            kronecker = 1*(dir_k==dir_j)
            dx_kdx_j[dir_k,dir_j,:,:] = kronecker * DFT.T
    _, dzdx, _ = JenkinsCorLoop(x_k, k, dx_kdx_j, DFT, limit_friction_force)
    dfdx = k * (dx_kdx_j - dzdx)
    for dir1 in range(Pdir.shape[0]): 
        for dir2 in range(Pdir.shape[0]):
            dJdx += (Pslave - Pmaster).T @ Pdir[dir1,:,:].T @ (DTF.T @ dfdx[dir1,dir2,:,:]) @ Pdir[dir2,:,:] @ (Pslave - Pmaster)
    return dJdx,dJdom
    

class Jenkins(NodeToNodeElement) :
    """
    This element is jenkins element, modeling a friction contact using an approximate Coulomb law considering a linear spring behavior when stuck. 
    
    Attributes:
        mu (float): friction coefficient.
        N0 (float): normal preload on the element.
        k (foat): linear spring value.
        jac (str): if "analytical" uses analytical expression of jacobian, if "jax" uses automatic differentiation, if "DF" uses finite difference.
    """
    factory_keyword : str = "Jenkins" 
    """str: keyword that is used to call the creation of this class in the system factory."""
    def __post_init__(self,):
        self.mu = self.data["mu"]
        self.N0 = self.data["N0"]
        self.k = self.data["k"]
        if "jac" in self.data.keys():
            self.jac = self.data["jac"]
        else:
            self.jac = "analytical"
        self.nabo = np.linalg.matrix_power(self.nabla,0)

    def __flag_update__(self,):
        self.flag_nonlinear = True
        self.flag_AFT = True

    def _evalJaco_ana(self, xg, om, *args):
        x = xg[self.indices]
        dJdx,dJdom = JenkinsJacobian(x, om, self.Pdir, self.Pslave, self.Pmaster, self.mu, self.N0, self.k, self.D["ft"],self.D["tf"])
        self.J = copy.copy(dJdx)
        self.dJdom = copy.copy(dJdom)
        return dJdx, dJdom
    
    def _evalJaco_jax(self, xg, om, *args):
        x = xg[self.indices]
        self.dJdom = np.zeros((len(x),))
        self.dJdx = jaxf(x, om, self.Pdir, self.Pslave, self.Pmaster, self.mu, self.N0, self.k, self.D["ft"],self.D["tf"])        
        return self.dJdx, self.dJdom
    
    def adim(self, lc, wc):
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Attributes:
            k (float): modified linear coefficient to apply according to the characteristic parameters.
            flag_adim (bool): True if equations are to be adimensionalized
        """
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
        self.R, stuck = JenkinsResidual(x, om, self.Pdir, self.Pslave, self.Pmaster, self.mu, self.N0, self.k, self.D["ft"],self.D["tf"])
        self.stuck = stuck
        self.tau = 100 * np.where(np.array(stuck)==True)[0].shape[0] / len(stuck)
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
        if self.jac == "analytical":
            dJdx, dJdom = self._evalJaco_ana(xg, om)
        elif self.jac == "DF":
            dJdx, dJdom = self._evalJaco_DF(xg, om)
        elif self.jac == "jax":
            dJdx, dJdom = self._evalJaco_jax(xg, om)
        self.J = dJdx
        self.dJdom = dJdom
        return self.J,self.dJdom