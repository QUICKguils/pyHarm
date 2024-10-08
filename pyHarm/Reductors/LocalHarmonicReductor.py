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

from pyHarm.Reductors.FactoryReductors import GlobalHarmonicReductor
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
import pandas as pd
import numpy as np


class LocalHarmonicReductor(GlobalHarmonicReductor) : 
    """
    The local harmonic reductor is a reductor (inheriting from the global Harmonic reductor) aiming at reducing the number of harmonics considered in the problem while solving. 

    Hence in linear situation, it is expected that the problem is gonna be reduced solely to the harmonics being involved in the forcing. 
    The reduction method is based on Gastaldi et al. proposal in : 
        - A method to solve the efficiency-accuracy trade-off of multi-harmonic balance calculation of structures with friction contacts.
    This Reductor is local in the sense that it keeps only the higher harmonic numbers for the needed dofs
    
    Attributes : 
        factory_keyword (str): name to use as input when willing to create an instance of this object.
        default (dict): dictionary of default reducer options.
        max_nh (int): maximum number of harmonics.
        data (dict): dictionary of input + missing parameters from default dictionary.
        tot_dofs (int): total number of dofs.
        disp_cut_off (float): displacement cut off parameter used to put compute the criterion 
        err_admissible (float): admissible error commited on the representation of the input displacement using limited amount of harmonics
        h_always_kept (np.ndarray): harmonic numbers that shall be kept in any condition
        verbose (bool): parameter to display info at each update of the number of harmonics 
        tol_update (float): tolerance over the jacobian that tells if an update of the number of harmonics might be necessary
    """
    
    factory_keyword : str = "localHarmonic"
    """str: keyword that is used to call the creation of this class in the factory."""
    
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
        self.harmonic_to_keep = np.array(self.expl_dofs.index)
        self.build_phi(self.harmonic_to_keep)
        self.J_ch = np.zeros((self.tot_dofs,self.tot_dofs))
        pass
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
            phi (np.ndarray): Updated transformation matrix generated depending on the number of harmonics to keep.

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
        
    def build_phi(self,nh_kept,local=False):
        """
        Creates the masking phi matrix based on the harmonic numbers that are to be kept.

        Args:
            nh_kept (np.ndarray): Harmonic numbers to keep in the reduced vector.

        Attributes:
            phi (np.nd.array): Updated transformation matrix.
        """
        phi = np.zeros((self.tot_dofs,len(nh_kept)))
        columns = np.arange(0,len(nh_kept))
        phi[nh_kept,columns] = 1
        self.phi = np.block([[phi,np.zeros((phi.shape[0],1))],
                             [np.zeros((1,phi.shape[1])),np.ones((1,1))]])
        pass
    
    def _harmonic_selection(self,criterion):
        """"
        Based on the criterion values and the harmonics that are set to be kept all the time, 
        creates the array with all the harmonics to keep in the calculation for each dof.

        Args:
            criterion (np.ndarray[bool]): array of booleans that indicates if the criterion is verified.
            
        Returns:
            np.ndarray: array of harmonic numbers to keep.
        """
        atomatic_kept = np.array(self.expl_dofs[self.expl_dofs["harm"].isin(self.h_always_kept)].index)
        df_crit = self.expl_dofs[criterion]
        df_matching = self.expl_dofs[self.expl_dofs[["sub","harm","node_num","dof_num"]].isin(
            df_crit[["sub","harm","node_num","dof_num"]].to_dict(orient='list')).all(axis=1)]
        matching_indices = df_matching.index.tolist()
        criterion[matching_indices] = True 
        criterion_kept = np.array(self.expl_dofs[criterion].index)
        harmonic_to_keep = np.unique(np.concatenate([atomatic_kept,\
                                                     criterion_kept]
                                                        )
                                    )
        return harmonic_to_keep
    
    def _get_output_expl_dofs(self,):
        """
        Obtains the modified explicit dof DataFrame after passing through the reducer.

        Returns:
            np.ndarray: Modified explicit dof DataFrame after passing through the reducer.
        """
        output_expl_dofs = self.expl_dofs.loc[np.where(self.phi[:-1,:-1] != 0)[0]].reset_index(drop=True)
        return output_expl_dofs