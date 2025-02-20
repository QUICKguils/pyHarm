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

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.KinematicConditions.ABCKinematic import ABCKinematic
import numpy as np
import pandas as pd



class BaseProjection(ABCKinematic) : 
    """Kinematic condition that imposes base projection between two substructures. 
    A linear transformation is imposed through base projection matrix in between a master substructure and the slave substructure.
    Residual contribution computed on the slave substructure is projected back onto the master substructure.

    Attributes:
        factory_keyword (str): keyword that is used to call the creation of this class in the system factory.
        default (dict): dictionary containing the default parameters for the kinematic condition.
        phi (np.ndarray): transformation matrix that goes from master base to slave base.
        phi_inv (np.ndarray): transpose of phi
    """
    factory_keyword:str = "BaseProjection"
    default = {}
    def __post_init__(self,):
        self.data = getCustomOptionDictionary(self.data,self.default)
        self.phi = np.kron(np.eye(2*self.nh+1),self.data["phi"])
        if "phi_inv" not in self.data.keys() : 
            self.phi_inv = self.phi.T
        else : self.phi_inv = np.kron(np.eye(2*self.nh+1),self.data["phi_inv"])

    def generateIndices(self,expl_dofs:pd.DataFrame) :
        condition_in = expl_dofs['sub'].isin(self.subs)
        condition_out = ~expl_dofs['sub'].isin(self.subs)
        self.indices = expl_dofs[condition_in].index
        self.indices_out = expl_dofs[condition_out].index
        ed = expl_dofs.loc[self.indices].reset_index(drop=True)
        i_master = ed[
            (
                (ed['sub']==self.subs[1])
            )
        ].index
        i_slave = ed[
            (
                (ed['sub']==self.subs[0])
            )
        ].index
        self.Pmaster,self.Pslave = np.eye(len(self.indices))[i_master,:],np.eye(len(self.indices))[i_slave,:]

    def complete_x(self, x) :
        """Returns a vector x_add of same size of x that completes the vector of displacement x = x + x_add such that the kinematic condition is verified.
        
        Args:
            x (np.ndarray): displacement vector.
            om (float): angular frequency.

        Returns:
            (np.ndarray): vector of displacement to add to the displacement vector in order to impose the kinematic condition.
        """
        xadd = np.zeros(x.shape)
        xadd[self.indices] += self.Pslave.T @ self.phi @self.Pmaster @ x[self.indices]
        return xadd
        
    def complete_R(self, R, x):
        """Computes the transfer of residual of the kinematicaly constrained dofs with respect to the displacement.
        
        Args:
            R (np.ndarray): residual vector.
            x (np.ndarray): displacement vector.

        Returns:
            (np.ndarray): vector of residual contributions to add to the residual vetor in order to impose the kinematic condition.
        """
        R_add = np.zeros(R.shape)
        R_add[self.indices] = self.Pmaster.T @ self.phi_inv @ self.Pslave @ R[self.indices]
        return R_add
    
    def complete_J(self, Jx, Jom, x):
        """Computes the transfer of jacobian of the kinematicaly constrained dofs with respect to the displacement.
        
        Args:
            Jx (np.ndarray): jacobian matrix with respect to displacement.
            Jom (np.ndarray): jacobian matrix with respect to angular frequency.
            x (np.ndarray): displacement vector.

        Returns:
            (tuple[np.ndarray,np.ndarray]): tuple containing the jacobians contributions to add to the jacobians in order to impose the kinematic condition.
        """
        Jx_add = np.zeros(Jx.shape)
        Jom_add = np.zeros(Jom.shape)
        master_indexes = (self.Pmaster @ self.indices).astype(int)
        slave_indexes = (self.Pslave @ self.indices).astype(int)
        other_indexes = (self.indices_out).astype(int)
        
        msh_master = tuple(np.meshgrid(master_indexes,master_indexes,indexing='ij'))
        msh_slave = tuple(np.meshgrid(slave_indexes,slave_indexes,indexing='ij'))
        
        msh_slave_master = tuple(np.meshgrid(slave_indexes,master_indexes,indexing='ij'))
        msh_master_slave = tuple(np.meshgrid(master_indexes,slave_indexes,indexing='ij'))
        
        msh_slave_other = tuple(np.meshgrid(slave_indexes,other_indexes,indexing='ij'))
        msh_master_other = tuple(np.meshgrid(master_indexes,other_indexes,indexing='ij'))
        msh_other_slave = tuple(np.meshgrid(other_indexes,slave_indexes,indexing='ij'))
        msh_other_master = tuple(np.meshgrid(other_indexes,master_indexes,indexing='ij'))
        
        Jom_add[master_indexes] += self.phi_inv @ Jom_add[slave_indexes]
        Jx_add[msh_master] += self.phi_inv @ Jx[msh_slave] @ self.phi
        Jx_add[msh_master_other] += self.phi_inv @ Jx[msh_slave_other]
        Jx_add[msh_other_master] += Jx[msh_other_slave] @ self.phi
        Jx_add[msh_master] += Jx[msh_master_slave] @ self.phi
        Jx_add[msh_master] += self.phi_inv @ Jx[msh_slave_master] 

        return Jx_add, Jom_add

    def adim(self, lc, wc):
        pass