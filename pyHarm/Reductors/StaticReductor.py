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

class StaticReductor(ABCReductor) : 
    """
    This reductor applies a provided static transformation matrix in order to reduce the system.

    Attributes : 
        phi (np.ndarray): transformation matrix
    """
    factory_keyword : str = "static"
    """str: keyword that is used to call the creation of this class in the factory."""

    def __post_init__(self,*args) : 
        phi_x = self.data["phi"]
        self.phi = np.block([[phi_x, np.zeros((self.phi_x.shape[0],1))],
                             [np.zeros((1,self.phi_x.shape[1])), np.ones((1,1))]])
    
    def update_reductor(self, xpred:np.ndarray, J_f:np.ndarray, *args) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Does nothing as the transformation is taken as static.

        Returns:
            np.ndarray: same displacement vector given as input after passing through the reductor
            np.ndarray: same jacobian matrix given as input after passing through the reductor
            pd.DataFrame: same explicit dof DataFrame after passing through the reductor

        """
        return self.reduce_vector_x(xpred), self.reduce_matrix(J_f), self.output_expl_dofs()
    
    def expand(self,q:np.ndarray) -> np.ndarray:
        """
        Inverse transforms the displacement vector.

        Args:
            q (np.ndarray): vector of transformed displacement.

        Returns:
            np.ndarray: vector of original displacement.
        """
        x = self.phi @ q
        return x
    
    def reduce_vector_x(self,x:np.ndarray) -> np.ndarray:
        """
        Transforms the displacement vector.

        Args:
            x (np.ndarray): vector of displacement.

        Returns:
            np.ndarray: vector of transformed displacement.
        """
        q = self.phi.T @ x
        return q
    
    def reduce_vector(self,R:np.ndarray) -> np.ndarray:
        """
        Transforms the residual vector.

        Args:
            R (np.ndarray): vector of residuals.

        Returns:
            np.ndarray: vector of transformed residuals.
        """
        q = self.phi.T @ R 
        return q
    
    def reduce_matrix(self,dJdxom:np.ndarray,*args) -> np.ndarray:
        """
        From original matrix, performs the transformation to get the transformed matrix.

        Args:
            dJdxom (np.ndarray): full size jacobian matrix with respect to displacement and angular frequency.

        Returns:
            np.ndarray: transformed jacobian matrix with respect to displacement and angular frequency.
        """
        return self.phi.T @ dJdxom @ self.phi
    
    def _get_output_expl_dofs(self,):
        """
        This is not well implemented => this has to be redone !!.
        
        Returns:
            pd.DataFrame: reduced explicit dof DataFrame after passing through the reducer.

        """
        return self.expl_dofs