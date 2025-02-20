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

class NoReductor(ABCReductor) : 
    """
    This Reductor does nothing.
    """
    factory_keyword : str = "noreductor"
    """str: keyword that is used to call the creation of this class in the factory."""
    
    def update_reductor(self, xpred, J_f, *args) :
        """
        Nothing is done here.

        Returns:
            np.ndarray: modified displacement vector given as input after passing through the reductor
            np.ndarray: modified jacobian matrix given as input after passing through the reductor
            pd.DataFrame: modified explicit dof DataFrame after passing through the reductor

        """
        return xpred, J_f, self.expl_dofs
    
    def expand(self,q:np.ndarray) -> np.ndarray:
        """
        Does nothing.

        Args:
            q (np.ndarray): vector of displacement.

        Returns:
            np.ndarray: Same vector as input.
        """
        return q
    
    def reduce_vector(self,x:np.ndarray) -> np.ndarray:
        """
        Does nothing.

        Args:
            R (np.ndarray): residual vector.

        Returns:
            np.ndarray: same residual vector.
        """
        return x
    
    def reduce_matrix(self,dJdx:np.ndarray,*args) -> np.ndarray:
        """
        Does nothing.

        Args:
            dJdxom (np.ndarray): full size jacobian matrix with respect to displacement and angular frequency.

        Returns:
            np.ndarray: same matrix.
        """
        return dJdx
    
    def _get_output_expl_dofs(self,):
        """
        Does nothing.
        
        Returns:
            pd.DataFrame: reduced explicit dof DataFrame given as input.

        """
        return self.expl_dofs 
