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
def DLFTFrictionResidualCorLoop_jax(lambda_opt_t,lambda_pre_t,limit_friction_force):
    slip = jnp.sqrt(jnp.sum(lambda_pre_t[:,-1]**2))>=limit_friction_force[-1]
    nti = jnp.shape(lambda_pre_t)[1]
    lambda_cor_t = jnp.zeros(lambda_pre_t.shape)

    def function_to_loop(t, state):
        lambda_cor_t, lambda_pre_t, slip = state
        tp1 = (t + 1) * (t + 1 != nti)
        def true_fn(lambda_cor_tm1, limit_friction_force, lambda_pre_t1):
            return lambda_cor_tm1 + lambda_pre_t1 * (1 - limit_friction_force / jnp.sqrt(jnp.sum(lambda_pre_t1 ** 2)))
        def false_fn(lambda_cor_tm1, limit_friction_force, lambda_pre_t1):
            return lambda_cor_tm1
        lambda_cor_t = lambda_cor_t.at[:,t].set(jax.lax.cond(slip, true_fn, false_fn, lambda_cor_t[:,t-1], limit_friction_force[t], lambda_pre_t[:,t]))
        lambda_pre_t = lambda_pre_t.at[:,tp1].set(lambda_opt_t[:,tp1] - lambda_cor_t[:,t])
        slip = jnp.sqrt(jnp.sum(lambda_pre_t[:,tp1]**2))>=limit_friction_force[tp1]
        return (lambda_cor_t, lambda_pre_t, slip)

    for _ in range(3) :
        initial_state = (lambda_cor_t, lambda_pre_t, slip)
        final_state = jax.lax.fori_loop(0, nti, function_to_loop, initial_state)
        lambda_cor_t, lambda_pre_t, slip = final_state

    slip_all = jnp.sqrt(jnp.sum(lambda_pre_t**2, axis=0))>=limit_friction_force

    return lambda_cor_t, lambda_pre_t, slip_all

@jax.jit
def DLFTFrictionResidual_jax(x, om, Rlin,\
                         nbSub, Pdir, Pslave, Pmaster,\
                         mu, N0, eps,\
                         DFT, DTF, f_time_n):
    limit_friction_force = mu * jnp.abs(f_time_n - N0 * jnp.ones(DFT.shape[1]))
    R = jnp.zeros((len(x),))
    nti = jnp.shape(DFT)[1]
    lambda_opt_t = jnp.zeros((jnp.shape(Pdir)[0],nti))
    for dir_k in range(jnp.shape(Pdir)[0]):
        lambda_opt_t = lambda_opt_t.at[dir_k,:].set((-1/nbSub*(Pdir[dir_k,:,:] @ (Pslave - Pmaster) @ Rlin)\
                                 + eps*(Pdir[dir_k,:,:] @ (Pslave - Pmaster) @ x)) @ DFT)
    lambda_pre_t = lambda_opt_t - jnp.zeros((jnp.shape(Pdir)[0],nti))
    lambda_cor_t,lambda_pre_t,slip_all = DLFTFrictionResidualCorLoop_jax(lambda_opt_t,lambda_pre_t,limit_friction_force)
    lambda_t = lambda_opt_t - lambda_cor_t
    for dir_k in range(jnp.shape(Pdir)[0]):
        R += (Pslave-Pmaster).T @ Pdir[dir_k,:,:].T @ (lambda_t[dir_k,:] @ DTF)
    return R,slip_all

jaxf = jax.jacfwd(DLFTFrictionResidual_jax)
jaxf = jax.jit(jaxf)


@njit(cache=True)
def DLFTFrictionResidualCorLoop(lambda_opt_t,lambda_pre_t,mu,N0):
    STUCK = np.sqrt(np.sum(lambda_pre_t[:,-1]**2))<mu*N0
    EPS = 1E-12
    nti = lambda_pre_t.shape[1]
    lambda_cor_t = np.zeros(lambda_pre_t.shape)
    for _ in range(3) :
        for t in range(nti) :
            tp1 = t+1
            if t+1 == nti:
                tp1 = 0
            if STUCK : 
                lambda_cor_t[:,t] = lambda_cor_t[:,t-1]
            else : 
                lambda_cor_t[:,t] = lambda_cor_t[:,t-1] + \
                                    lambda_pre_t[:,t] * (1 - mu*N0/np.sqrt(np.sum(lambda_pre_t[:,t]**2)))
            lambda_pre_t[:,tp1] = lambda_opt_t[:,tp1] - lambda_cor_t[:,t]
            STUCK = np.sqrt(np.sum(lambda_pre_t[:,tp1]**2))<mu*N0
    return lambda_cor_t,lambda_pre_t


@njit(cache=True)
def DLFTFrictionResidual(x, om, Rlin,\
                         nbSub, Pdir, Pslave, Pmaster,\
                         mu, N0, eps,\
                         DFT, DTF):
    R = np.zeros((len(x),))
    EPS = 1E-12
    nti = DFT.shape[1]
    nh = DFT.shape[0]
    lambda_opt_t = np.zeros((Pdir.shape[0],nti))
    for dir_k in range(Pdir.shape[0]):
        lambda_opt_t[dir_k,:] = (-1/nbSub*(Pdir[dir_k,:,:] @ (Pslave - Pmaster) @ Rlin)\
                                 + eps*(Pdir[dir_k,:,:] @ (Pslave - Pmaster) @ x)) @ DFT
    lambda_pre_t = lambda_opt_t - np.zeros((Pdir.shape[0],nti))
    lambda_cor_t,lambda_pre_t = DLFTFrictionResidualCorLoop(lambda_opt_t,lambda_pre_t,mu,N0)
    lambda_t = lambda_opt_t - lambda_cor_t
    for dir_k in range(Pdir.shape[0]):
        R += (Pslave-Pmaster).T @ Pdir[dir_k,:,:].T @ (lambda_t[dir_k,:] @ DTF)
    return R,lambda_opt_t,lambda_cor_t,lambda_pre_t


@njit(cache=True)
def DLFTFrictionJacobianCorLoop(dlamdx_opt_t_ss,dlamdx_opt_t_sm,dlamdom_opt_t,lambda_pre_t,mu,N0):
    STUCK = np.sqrt(np.sum(lambda_pre_t[:,0]**2)) < mu*N0
    dlamdx_cor_t_ss = np.zeros(dlamdx_opt_t_ss.shape)
    dlamdx_cor_t_sm = np.zeros(dlamdx_opt_t_ss.shape)
    dlamdom_cor_t = np.zeros(dlamdom_opt_t.shape)
    dlamdx_pre_t_ss = np.zeros(dlamdx_opt_t_ss.shape)
    dlamdx_pre_t_sm = np.zeros(dlamdx_opt_t_ss.shape)
    dlamdom_pre_t = np.zeros(dlamdom_opt_t.shape)
    nti = lambda_pre_t.shape[1]
    T = np.arange(0,nti)
    for _ in range(3) :
        for t in range(nti) :
            tp1 = t+1
            if t+1 == nti:
                tp1 = 0
            if STUCK : 
                dlamdx_cor_t_ss[:,:,:,t] = dlamdx_cor_t_ss[:,:,:,t-1]
                dlamdx_cor_t_sm[:,:,:,t] = dlamdx_cor_t_sm[:,:,:,t-1]
                dlamdom_cor_t[:,t] = dlamdom_cor_t[:,t-1]
                
            else :
                for dir_k in range(dlamdx_opt_t_ss.shape[0]):
                    # dlamdx_cor_t_ss
                    sum_directions_ss = (lambda_pre_t[:,t]@dlamdx_pre_t_ss[dir_k,:,:,t]).reshape(1,-1)
                    lambda_pre_nicelyshaped = np.expand_dims(lambda_pre_t[:,t],axis=-1)
                    dlamdx_cor_t_ss[dir_k,:,:,t] = dlamdx_cor_t_ss[dir_k,:,:,t-1] + \
                                              dlamdx_pre_t_ss[dir_k,:,:,t] * (1-mu*N0/np.sqrt(np.sum(lambda_pre_t[:,t]**2))) +\
                                              mu*N0*lambda_pre_nicelyshaped @\
                    sum_directions_ss / np.sqrt(np.sum(lambda_pre_t[:,t]**2))**3
                    
                    # dlamdx_cor_t_sm
                    sum_directions_ms = (lambda_pre_t[:,t]@dlamdx_pre_t_sm[dir_k,:,:,t]).reshape(1,-1)
                    dlamdx_cor_t_sm[dir_k,:,:,t] = dlamdx_cor_t_sm[dir_k,:,:,t-1] + \
                                              dlamdx_pre_t_sm[dir_k,:,:,t] * (1-mu*N0/np.sqrt(np.sum(lambda_pre_t[:,t]**2))) +\
                                              mu*N0*lambda_pre_nicelyshaped @\
                    sum_directions_ms / np.sqrt(np.sum(lambda_pre_t[:,t]**2))**3
                    
                dlamdom_cor_t[:,t] = dlamdom_cor_t[:,t-1]\
                                     + dlamdom_pre_t[:,t]*(1-mu*N0/np.sqrt(np.sum(lambda_pre_t[:,t]**2)))\
                                     + mu*N0*lambda_pre_t[:,t]*(lambda_pre_t[:,t]@dlamdom_pre_t[:,t])/ np.sqrt(np.sum(lambda_pre_t[:,t]**2))**3
                
                
                
            dlamdx_pre_t_ss[:,:,:,tp1] = dlamdx_opt_t_ss[:,:,:,tp1] - dlamdx_cor_t_ss[:,:,:,t]
            dlamdx_pre_t_sm[:,:,:,tp1] = dlamdx_opt_t_sm[:,:,:,tp1] - dlamdx_cor_t_sm[:,:,:,t]
            dlamdom_pre_t[:,tp1] = dlamdom_opt_t[:,tp1] - dlamdom_cor_t[:,t]
            STUCK = np.sqrt(np.sum(lambda_pre_t[:,tp1]**2)) < mu*N0
    return dlamdx_cor_t_ss,dlamdx_cor_t_sm,dlamdom_cor_t

@njit(cache=True)
def DLFTFrictionJacobian(x, om, Rlin, dJdxlin, dJdomlin,\
                         nbSub, Pdir, Pslave, Pmaster,\
                         mu, N0, eps,\
                         DFT, DTF):
    dJdx = np.zeros((len(x),len(x)))
    dJdom = np.zeros((len(x),))
    nti = DFT.shape[1]
    nh = DFT.shape[0]
    dlamdx_opt_t_ss = np.zeros((Pdir.shape[0],Pdir.shape[0],nh,nti))
    dlamdx_opt_t_ms = np.zeros((Pdir.shape[0],Pdir.shape[0],nh,nti))
    dlamdom_opt_t = np.zeros((Pdir.shape[0],nti))
    _,lambda_opt,lambda_cor,lambda_pre = DLFTFrictionResidual(x, om, Rlin,\
                                                              nbSub, Pdir, Pslave, Pmaster,\
                                                              mu, N0, eps,\
                                                              DFT, DTF)
    for dir_k in range(Pdir.shape[0]):
        for dir_j in range(Pdir.shape[0]):
            dlamdx_opt_t_ss[dir_k,dir_j,:,:] = (\
                -  1/nbSub * Pdir[dir_k,:,:] @ (Pslave-Pmaster) @dJdxlin \
                + eps*(Pdir[dir_k,:,:] @ (Pslave - Pmaster) )
            )  @ (Pslave).T @ Pdir[dir_j,:,:].T @ DFT
            dlamdx_opt_t_ms[dir_k,dir_j,:,:] = (\
                -  1/nbSub * Pdir[dir_k,:,:] @ (Pslave-Pmaster) @dJdxlin \
                - eps*(Pdir[dir_k,:,:] @ (Pslave - Pmaster) )
            ) @ (Pmaster).T @ Pdir[dir_j,:,:].T @ DFT
        dlamdom_opt_t[dir_k,:] = (- 1/nbSub * Pdir[dir_k,:,:] @ (Pslave-Pmaster) @ dJdomlin).reshape(-1,) @ DFT
    dlamdx_cor_t_ss, dlamdx_cor_t_ms,dlamdom_cor_t = DLFTFrictionJacobianCorLoop(dlamdx_opt_t_ss, dlamdx_opt_t_ms, dlamdom_opt_t, lambda_pre, mu, N0)
    for dir_k in range(Pdir.shape[0]):
        for dir_j in range(Pdir.shape[0]):
            dlam_h_ss = Pdir[dir_k,:,:].T @ (dlamdx_opt_t_ss[dir_k,dir_j,:,:] - dlamdx_cor_t_ss[dir_k,dir_j,:,:]) @ DTF @ Pdir[dir_j,:,:]
            dlam_h_sm = Pdir[dir_k,:,:].T @ (dlamdx_opt_t_ms[dir_k,dir_j,:,:] - dlamdx_cor_t_ms[dir_k,dir_j,:,:]) @ DTF @ Pdir[dir_j,:,:]
            dJdx += ((Pslave).T @ dlam_h_ss.T @ (Pslave))
            dJdx -= ((Pslave).T @ dlam_h_ss.T @ (Pmaster))
            dJdx -= ((Pmaster).T @ dlam_h_sm.T @ (Pslave))
            dJdx += ((Pmaster).T @ dlam_h_sm.T @ (Pmaster))
        dJdom += (Pslave-Pmaster).T @ Pdir[dir_k,:,:].T @ ((dlamdom_opt_t[dir_k,:] - dlamdom_cor_t[dir_k,:]) @ DTF)
    return dJdx,dJdom

class DLFTFriction(DLFTElement): 
    """
    This element is the DLFT friction element, modeling a friction contact using a Coulomb law. 
    
    Attributes:
        mu (float): friction coefficient.
        N0 (float): normal precharge on the element.
        eps (foat): penalty parameter of the DLFT method.
        jac (str): if "analytical" uses analytical expression of jacobian, if "jax" uses automatic differentiation, if "DF" uses finite difference.
    """
    factory_keyword : str = "DLFTFriction"
    """str: keyword that is used to call the creation of this class in the system factory."""
    def __post_init__(self,):
        self.mu = self.data["mu"]
        self.N0 = self.data["N0"]
        self.eps = self.data["eps"]
        self.nabo = np.linalg.matrix_power(self.nabla,1)
        if "jac" in self.data.keys():
            self.jac = self.data["jac"]
        else:
            self.jac = "analytical"

    def _evalJaco_ana(self, xg, om, Rglin, dJgdxlin, dJgdomlin,*args):
        x = xg[self.indices]
        Rlin = Rglin[self.indices]
        dJdxlin = dJgdxlin[tuple(np.meshgrid(self.indices,self.indices))]
        dJdomlin = dJgdomlin[self.indices]
        dJdx,dJdom = DLFTFrictionJacobian(x, om, Rlin, dJdxlin, dJdomlin,\
                                          float(self.nbSub), self.Pdir, self.Pslave, self.Pmaster,\
                                          self.mu, self.N0, self.eps,\
                                          self.D["ft"],self.D["tf"])
        self.J = copy.copy(dJdx)
        self.dJdom = copy.copy(dJdom)
        return dJdx, dJdom
    
    def _evalJaco_jax(self, xg, om, Rglin=None):
        x = xg[self.indices]
        Rlin = Rglin[self.indices]
        self.dJdom = np.zeros((len(x),))
        self.dJdx = jaxf(x, om, Rlin,float(self.nbSub), self.Pdir, self.Pslave, self.Pmaster,self.mu, self.N0, self.eps, self.D["ft"],self.D["tf"])   
        return self.dJdx, self.dJdom
    
    def adim(self, lc, wc) :
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Attributes:
            eps (float): modified penalty coefficient to apply according to the characteristic parameters.
            flag_adim (bool): characteristic angular frequency value.
        """
        self.eps = self.eps * lc
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
        self.R,_,_,_ = DLFTFrictionResidual(x, om, Rlin,\
                                            float(self.nbSub), self.Pdir, self.Pslave, self.Pmaster,\
                                            self.mu, self.N0, self.eps,\
                                            self.D["ft"],self.D["tf"])
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
            dJdx, dJdom = self._evalJaco_ana(xg, om, Rglin, dJgdxlin, dJgdomlin, 1e-8)
        elif self.jac == "jax":
            dJdx, dJdom = self._evalJaco_jax(xg, om, Rglin)
        
        self.J = dJdx
        self.dJdom = dJdom
        return self.J,self.dJdom