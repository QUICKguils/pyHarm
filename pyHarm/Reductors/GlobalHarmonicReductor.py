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
from pyHarm.Reductors.ABCReductor import ABCReductor
import numpy as np


class GlobalHarmonicReductor(ABCReductor) :  
    """
    The global harmonic reductor is a reductor aiming at reducing the number of harmonics considered in the problem while solving.

    Hence in linear situation, it is expected that the problem is gonna be reduced solely to the harmonics being involved in the forcing. 
    The reduction method is based on Gastaldi et al. proposal in : 
        - A method to solve the efficiency-accuracy trade-off of multi-harmonic balance calculation of structures with friction contacts.
    This Reductor is global in the sense that it keeps all the dofs of a certain harmonic number even if only one dof needs this harmonic number.

    Attributes : 
        max_nh (int): maximum number of harmonics.
        data (dict): dictionary of input + missing parameters from default dictionary.
        tot_dofs (int): total number of dofs.
        disp_cut_off (float): displacement cut off parameter used to put compute the criterion 
        err_admissible (float): admissible error commited on the representation of the input displacement using limited amount of harmonics
        h_always_kept (np.ndarray): harmonic numbers that shall be kept in any condition
        verbose (bool): parameter to display info at each update of the number of harmonics 
        tol_update (float): tolerance over the jacobian that tells if an update of the number of harmonics might be necessary
    """
    factory_keyword : str = "globalHarmonic"
    """str: keyword that is used to call the creation of this class in the factory."""
    
    default = {"nh_start":"max_nh",
               "disp_cut_off":1e-9,
               "err_admissible":5e-2,
               "h_always_kept":np.array([0,1]),
               "tol_update":1e-9,
               "verbose":False}
    """dict: dictionary containing default parameters."""
    
    def __post_init__(self,*args):
        self.max_nh = np.max(self.expl_dofs["harm"])
        self.data = getCustomOptionDictionary(self.data,self.default)
        self.tot_dofs = len(self.expl_dofs)
        self.disp_cut_off = self.data["disp_cut_off"]
        self.err_admissible = self.data["err_admissible"]
        self._get_combinaisons_sub_node_dof()
        self.h_always_kept = self.data["h_always_kept"]
        self.verbose = self.data["verbose"]
        self.tol_update = self.data["tol_update"]
        if not isinstance(self.data["nh_start"],np.ndarray) : 
            H_to_add = np.arange(0,self.max_nh+1)
        else : 
            H_to_add = self.data["nh_start"]
        self.harmonic_to_keep = np.unique(np.concatenate([self.h_always_kept,\
                                                     H_to_add]
                                                     )
                                    )
        self.build_phi(self.harmonic_to_keep)
        self.J_ch = np.zeros((self.tot_dofs,self.tot_dofs))
        
    
    def update_reductor(self, xpred, J_f, expl_dofs, *args) :
        """
        Updates the number of harmonics to be solved.

        The update is made according to Gastaldi et al. paper named : 
        - A method to solve the efficiency-accuracy trade-off of multi-harmonic balance calculation of structures with friction contacts (2017)
        A tolerance on the change of the Jacobian has been added in order to avoid costly update if the Jacobian remains identical to previous update.

        Args:
            xpred (np.ndarray): point predicted as the new starting point for the next iteration of the analysis process.
            J_f (np.ndarray): full jacobian with respect to displacement and angular frequency (contains the correction equation residual).

        Attributes:
            phi (np.ndarray): Updated transformation matrix generated depending of the number of harmonics to keep.

        Returns:
            np.ndarray: same displacement vector given in input after passing through the reductor
            np.ndarray: same jacobian matrix given in input after passing through the reductor
            pd.DataFrame: same explicit dof DataFrame after passing through the reductor

        """
        self.expl_dofs = expl_dofs
        self.tot_dofs = len(self.expl_dofs)
        FE = J_f[:-1,:-1] @ xpred[:-1]
        J_uc = self._get_uncoupled_jacobian(J_f)
        update_nh = self._update_nh_kept(J_f,J_uc)
        if update_nh : 
            criterion = self._get_criterion(FE,xpred,J_uc)
            self.harmonic_to_keep = self._harmonic_selection(criterion)
            self.build_phi(self.harmonic_to_keep)
        self._display_infos(update_nh,self.harmonic_to_keep)
        self.output_expl_dofs = self._get_output_expl_dofs()
        return self.reduce_vector(xpred), self.reduce_matrix(J_f), self.output_expl_dofs
    
    def expand(self,q:np.ndarray) -> np.ndarray:
        """
        Applies the inverse transformation matrix to the reduced size displacement.

        Args:
            q (np.ndarray): vector of transformed displacement.

        Returns:
            np.ndarray: vector of original displacement.
        """
        x = self.phi @ q
        return x
    
    def reduce_vector(self,x:np.ndarray) -> np.ndarray:
        """
        Transforms the displacement vector.

        Args:
            x (np.ndarray): vector of displacement.

        Returns:
            np.ndarray: vector of reduced displacement.
        """
        q = self.phi.T @ x
        return q
    
    def reduce_matrix(self,dJdxom:np.ndarray,*args) -> np.ndarray:
        """
        From original matrix, performs the transformation to get the reduced matrix.

        Args:
            dJdxom (np.ndarray): full size jacobian matrix with respect to displacement and angular frequency.

        Returns:
            np.ndarray: reduced jacobian matrix with respect to displacement and angular frequency.
        """
        return self.phi.T @ dJdxom @ self.phi
    
    def build_phi(self,nh_kept):
        """
        Creates the masking phi matrix based on the harmonic numbers that are to be kept.

        Args:
            nh_kept (np.ndarray): Harmonic numbers to keep in the reduced vector.

        Attributes:
            phi (np.nd.array): Updated transformation matrix.
        """
        self.nh_kept = nh_kept
        kept_index = np.where(self.expl_dofs["harm"].isin(nh_kept))[0]
        phi = np.zeros((self.tot_dofs,len(kept_index)))
        columns = np.arange(0,len(kept_index))
        phi[kept_index,columns] = 1
        self.phi = np.block([[phi,np.zeros((phi.shape[0],1))],
                             [np.zeros((1,phi.shape[1])),np.ones((1,1))]])
        pass
    
    def harmonic_index(self,H):
        """
        Provides the index of dofs corresponding to the required harmonic number.

        Args:
            H (int): Harmonic number.

        Returns:
            np.nd.array: array of dof numbers associated to the required harmonic number.
        """
        return np.array((self.expl_dofs["harm"]==H).index)
    
    def _get_criterion(self,FE,xpred,J_uc):
        """
        Calculates the criterion based on proposal and the provided admissible error.

        Args:
            FE (np.ndarray): Linearized residual obtained by multiplying the jacobian with the displacement xpred.
            xpred (np.ndarray): predicted starting point for the next analysis step.
            J_uc (np.ndarray): cross harmonic matrix.

        Returns:
            np.ndarray[bool]: array of booleans that indicates if the criterion is reached.
        """
        delta_q = np.linalg.solve(J_uc[:-1,:-1], FE) - xpred[:-1]
        delta_q[np.abs(delta_q)<=self.disp_cut_off]=0
        norm_vec = self._max_disp_norm_per_dof(xpred)
        criterion =(np.abs(delta_q / norm_vec)>= self.err_admissible)
        return criterion

    def _harmonic_selection(self,criterion):
        """"
        Based on the criterion values and the harmonics that are set to be kept all the time, 
        creates the array with all the harmonics to keep in the calculation.

        Args:
            criterion (np.ndarray[bool]): array of booleans that indicates if the criterion is reached.
            
        Returns:
            np.ndarray: array of harmonic numbers to keep.
        """
        H_to_add = np.unique(self.expl_dofs[criterion]["harm"]).astype(int)
        harmonic_to_keep = np.unique(np.concatenate([self.h_always_kept,\
                                                        H_to_add]
                                                        )
                                    )
        return harmonic_to_keep

    def _display_infos(self,update_nh,harmonic_to_keep) : 
        """
        Displays the harmonics that are kept if verbose is set to True.

        Args:
            update_nh (bool): True if updating the reducer has been necessary.
            harmonic_to_keep (np.ndarray): array of harmonic numbers to keep.
        """
        if self.verbose : 
            vocabulary = {True:"updated to ", False:"kept    to "}
            verbe = vocabulary[update_nh]
            htk = self.harmonic_to_keep
            print(f"Harmonic {verbe}{htk}")
            
    def _update_nh_kept(self,J_f,J_uc):
        """
        Low computational sub-criterion to know if the number of harmonics to keep has to be updated based on changes of the Jacobian.

        Args:
            J_f (np.ndarray): Full size jacobian with respect to displacement and angular frequency.
            J_uc (np.ndarrray): Cross harmonic part of the jacobian.

        Returns:
            bool: True if updating the reducer has been necessary. 
        """
        J_ch = J_f[:-1,:-1] - J_uc[:-1,:-1]
        if J_ch.shape != self.J_ch.shape : 
            return True
        tol = np.linalg.norm(self.J_ch - J_ch) 
        self.J_ch = J_ch
        update_nh = (tol >= self.tol_update)
        if update_nh : 
            return True
        else : return False
        
    def _max_disp_norm_per_dof(self,xpred) : 
        """
        Obtains normalisation value for dof based on maximal value over the harmonic number of the dof.

        Args:
            xpred (np.ndarray): predicted starting point for the next analysis step.

        Returns:
            np.ndarray: infinite norm of the harmonics for each dof.
        """
        norm_vec = np.zeros(self.tot_dofs)
        df = self.expl_dofs
        for sub,node,dof in self.combinaisons : 
            index = np.array((df["sub"] == sub) & (df["node_num"] == node) & (df["dof_num"] == dof))
            norm_vec[index] = np.max(np.abs(xpred[:-1][index]))
        return norm_vec
    
    def _get_uncoupled_jacobian(self,J_f) :
        """
        Returns the cross Harmonic terms in the Jacobian for criterion computation.

        Args:
            J_f (np.ndarray): full size jacobian.

        Returns:
            np.ndarray: Cross harmonic components
        """
        J_uc = np.zeros(J_f.shape)
        for h in range(0,self.max_nh+1) : 
            mshg = self._get_mshg_block(h)
            J_uc[mshg] = J_f[mshg]
        return J_uc
    
    def _get_combinaisons_sub_node_dof(self):
        """
        Returns a combination of (substructure, node, dof_number) based on the explicit dof DataFrame.

        Attributes:
            combinaisons (tuple[str,int,int]): combination of each sub, node, dof possible onto the system.
        """
        index = self._get_index_block(0)
        df = self.expl_dofs.iloc[index,:]
        self.combinaisons = tuple([(sub,node,dof) for sub,node,dof\
                                  in zip(df['sub'], df['node_num'], df['dof_num'])])
        pass 
    
    def _get_index_block(self,h):
        """
        Returns an array containing the indexes of dofs for a given harmonic number.

        Args:
            h (int): harmonic number.

        Returns:
            np.ndarray: indexes of dofs for a given harmonic number.
        """
        index = np.array(np.where((self.expl_dofs["harm"]==h)==True))[0]
        return index
    
    def _get_mshg_block(self,h):
        """
        Generates the meshgrid to get a block of a matrix.

        Args:
            h (int): harmonic number.

        Returns:
            np.ndarray: meshgrid to get the corresponding block of a matrix.
        """
        index = self._get_index_block(h)
        mshg_uc = tuple(np.meshgrid(index,index,indexing="ij"))
        return mshg_uc
    
    def _get_output_expl_dofs(self,):
        """
        Obtains the modified explicit dof DataFrame after passing through the reducer.

        Returns:
            np.ndarray: Modified explicit dof DataFrame after passing through the reducer.
        """
        output_expl_dofs = self.expl_dofs.loc[np.where(self.phi[:-1,:-1] != 0)[0]].reset_index(drop=True)
        return output_expl_dofs