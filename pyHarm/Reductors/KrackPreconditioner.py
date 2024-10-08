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
import copy


class KrackPreconditioner(ABCReductor) : 
    """
    This Reductor is a preconditioner that does not reduce the system size but preconditions the system to make it easier to solve for the solvers.

    This reductor is based on the preconditioning proposal made in Krack et al. book named :
        - Harmonic Balance for Nonlinear Vibration Problems
    It uses displacement vector and tries to normalise the values to 1. as long as the values remains under a certain threshold. 

    Attributes : 
        data (dict): input dictionary + default parameters when not provided as input
        cut_off (float): value of the displacement where the scaling to 1e0 displacement is no more made to avoid numerical errors.
    """
    factory_keyword : str = "KrackPreconditioner"
    """str: keyword that is used to call the creation of this class in the factory."""

    default = {"cut_off":1e-7}
    """dict: dictionary containing default parameters."""
    
    def __post_init__(self,*args):
        self.data = getCustomOptionDictionary(self.data,self.default)
        self.cut_off = self.data["cut_off"]
        pass
        
    def update_reductor(self,xpred,J_f,*args) :
        """
        Computes a preconditionning scaling matrix based on the QR decomposition of the jacobian.

        Args:
            xpred (np.ndarray): point predicted as the new starting point for the next iteration of the analysis process.
            J_f (np.ndarray): full jacobian with respect to displacement and angular frequency (contains the correction equation residual).

        Returns:
            np.ndarray: modified displacement vector given as input after passing through the reductor
            np.ndarray: modified jacobian matrix given as input after passing through the reductor
            pd.DataFrame: modified explicit dof DataFrame after passing through the reductor

        """
        scaling = copy.deepcopy(xpred)
        scaling[np.abs(scaling)<=self.cut_off] = 1.
        scaling[-1] = 1.
        self.phi_reduce = np.diag(np.abs(1/scaling))
        self.phi_expand = np.diag(np.abs(scaling))
        self.output_expl_dofs = self._get_output_expl_dofs()
        return self.reduce_vector_x(xpred), self.reduce_matrix(J_f), self.output_expl_dofs
    
    def expand(self,q:np.ndarray) -> np.ndarray:
        """
        Unscales the displacement vector.

        Args:
            q (np.ndarray): vector of scaled displacement.

        Returns:
            np.ndarray: vector of unscaled displacement.
        """
        x = self.phi_expand @ q
        return x
    
    def reduce_vector_x(self,x:np.ndarray) -> np.ndarray:
        """
        Scales the displacement vector.

        Args:
            x (np.ndarray): vector of displacement.

        Returns:
            np.ndarray: scaled vector of displacement.
        """
        q = self.phi_reduce @ x
        return q
    
    def reduce_vector(self,R:np.ndarray) -> np.ndarray:
        """
        Applies the scaling matrix to the residual vector.

        Args:
            R (np.ndarray): residual vector.

        Returns:
            np.ndarray: preconditioned residual vector.
        """
        R_red = R
        return R_red
    
    def reduce_matrix(self,dJdxom:np.ndarray,*args) -> np.ndarray:
        """
        From original matrix, performs the transformation to get the preconditioned matrix.

        Args:
            dJdxom (np.ndarray): full size jacobian matrix with respect to displacement and angular frequency.

        Returns:
            np.ndarray: preconditioned jacobian matrix with respect to displacement and angular frequency.
        """
        return dJdxom @ self.phi_expand
    
    def _get_output_expl_dofs(self,):
        """
        Returns the explicit dof list after transformation by the reducer.
        
        Returns:
            pd.DataFrame: reduced explicit dof DataFrame after passing through the reducer.

        """
        return self.expl_dofs
