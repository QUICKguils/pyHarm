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

from pyHarm.Reductors.FactoryReductors import generateReductor
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Reductors.ABCReductor import ABCReductor
import copy
import numpy as np

class ChainReductor(ABCReductor) :   
    """
    This is a ABCReductor that is a list of ABCReductor allowing to chain the reducing operations. 
    
    Attributes : 
        reductors (list[ABCReductor]): list of ABCReductor.
    """
    factory_keyword : str = "ChainReductor"
    """str: keyword that is used to call the creation of this class in the factory."""

    default = {"type":"noreductor"}
    """dict: dictionary containing default parameters."""

    def __post_init__(self,*args):
        self.reductors = []
        if len(self.data) == 0 :
            self.data = self.default
        expl_dofs = self.expl_dofs
        for data_red in self.data : 
            red = generateReductor(data_red,expl_dofs)
            expl_dofs = red._get_output_expl_dofs()
            self.reductors.append(red)
        pass
        
    def update_reductor(self, xpred, J_f, expl_dofs, system, *args) :
        """
        Loops over the reducers in the list and update them.

        Args:
            xpred (np.ndarray): point predicted as the new starting point for the next iteration of the analysis process.
            J_f (np.ndarray): full jacobian with respect to displacement and angular frequency (contains the correction equation residual).
            expl_dofs (pd.DataFrame): explicit dof DataFrame created by the ABCSystem studied.
            system (ABCSystem): System studied.

        Returns:
            np.ndarray: same displacement vector given as input after passing through the reductor
            np.ndarray: same jacobian matrix given as input after passing through the reductor
            pd.DataFrame: same explicit dof DataFrame after passing through the reductor

        """
        xpred_red = copy.copy(xpred)
        J_f_red = copy.copy(J_f)
        output_expl_dofs = copy.copy(expl_dofs)
        for red in self.reductors :
            xpred_red, J_f_red, output_expl_dofs = red.update_reductor(xpred_red,J_f_red,output_expl_dofs,system,*args)
        return xpred_red, J_f_red,output_expl_dofs
    
    def expand(self,q:np.ndarray) -> np.ndarray:
        """
        Expands the reduced dof vector by looping over the reducers contained in the list.

        Args:
            q (np.ndarray): vector of reduced displacement.

        Returns:
            np.ndarray: vector of original displacement with the linear dof being exact solution.
        """
        x = copy.copy(q)
        for red in self.reductors[-1::-1] :
            x = red.expand(x)
        return x
    
    def reduce_vectorx(self,x:np.ndarray) -> np.ndarray:
        """
        Reduces the displacement vector by applying the chain of reductor methods.

        Args:
            x (np.ndarray): vector of displacement.

        Returns:
            np.ndarray: vector of reduced displacement.
        """
        q = copy.copy(x)
        for red in self.reductors :
            q = red.reduce_vectorx(x)
        return q
    
    def reduce_vector(self,R:np.ndarray) -> np.ndarray:
        """
        Reduces the residual vector by applying the chain of reductor methods.

        Args:
            R (np.ndarray): vector of residual.

        Returns:
            np.ndarray: vector of reduced residual.
        """
        R_red = copy.copy(R)
        for red in self.reductors :
            R_red = red.reduce_vector(R_red)
        return R_red
    
    def reduce_matrix(self,dJdxom:np.ndarray,*args) -> np.ndarray:
        """
        From original jacobian, performs the reduction by applying the chain of reduction.

        Args:
            dJdxom (np.ndarray): full size jacobian matrix with respect to displacement and angular frequency.

        Returns:
            np.ndarray: reduced jacobian matrix with respect to displacement and angular frequency.
        """
        dJdxom_red = copy.copy(dJdxom)
        for red in self.reductors :
            dJdxom_red = red.reduce_matrix(dJdxom_red,*args)
        return dJdxom_red