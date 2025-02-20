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

from pyHarm.Reductors.ABCReductor import ABCReductor
import numpy as np
import pandas as pd

class AllgowerPreconditioner(ABCReductor) : 
    """
    This Reductor is a preconditioner that does not reduce the system size but preconditions the system to make it easier to solve for the solvers.

    This reductor is based on the preconditioning proposal made in Allgower et al. book named Numerical Continuation Methods: An Introduction.
    It uses QR decomposition of the Jacobian matrix and normalises the Jacobian using the diagonal terms in the R matrix.

    Attributes : 
        qr (Callable): QR decomposition function from scipy.linalg
    """
    factory_keyword : str = "AllgowerPreconditioner"
    """str: keyword that is used to call the creation of this class in the factory."""

    default = {}
    """dict: dictionary containing default parameters."""
    
    def __post_init__(self,*args):
        from scipy.linalg import qr
        self.qr = qr
        pass
        
    def update_reductor(self,xpred:np.ndarray,J_f:np.ndarray,*args) :
        """
        Computes a preconditioning scaling matrix based on the QR decomposition of the jacobian.

        Args:
            xpred (np.ndarray): point predicted as the new starting point for the next iteration of the analysis process.
            J_f (np.ndarray): full jacobian with respect to displacement and angular frequency (contains the correction equation residual).

        Returns:
            np.ndarray: modified displacement vector given as input after passing through the reductor
            np.ndarray: modified jacobian matrix given as input after passing through the reductor
            pd.DataFrame: modified explicit dof DataFrame after passing through the reductor

        """
        Q,R = self.qr(J_f)
        self.phi_reduce = np.diag(np.abs(1/np.diag(R)))
        self.output_expl_dofs = self._get_output_expl_dofs()
        return self.reduce_vector_x(xpred), self.reduce_matrix(J_f), self.output_expl_dofs
    
    def expand(self,q:np.ndarray) -> np.ndarray:
        """
        Does nothing for this class, as the preconditioner does not change size of the system.

        Args:
            q (np.ndarray): vector of displacement.

        Returns:
            np.ndarray: same vector than the input.
        """
        x = q
        return x
    
    def reduce_vector_x(self,x:np.ndarray) -> np.ndarray:
        """
        Does nothing for this class, as the preconditioner does not change size of the system.

        Args:
            x (np.ndarray): vector of displacement.

        Returns:
            np.ndarray: same vector than the input.
        """
        q = x
        return q
    
    def reduce_vector(self,R:np.ndarray) -> np.ndarray:
        """
        From original residual vector, performs the transformation to get the preconditioned residual vector.

        Args:
            R (np.ndarray): residual vector.

        Returns:
            np.ndarray: preconditioned residual vector.
        """
        R_red = self.phi_reduce @ R
        return R_red
    
    def reduce_matrix(self,dJdxom:np.ndarray,*args) -> np.ndarray:
        """
        From original matrix, performs the transformation to get the preconditioned matrix.

        Args:
            dJdxom (np.ndarray): full size jacobian matrix with respect to displacement and angular frequency.

        Returns:
            np.ndarray: preconditioned jacobian matrix with respect to displacement and angular frequency.
        """
        return self.phi_reduce @ dJdxom
    
    def _get_output_expl_dofs(self,) -> pd.DataFrame:
        """
        Returns the explicit dof list after transforamtion buy the reducer.
        
        Returns:
            pd.DataFrame: reduced explicit dof DataFrame after passing through the reducer.

        """
        return self.expl_dofs
    