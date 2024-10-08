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
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary


class NLdofsReductor(ABCReductor):  
    """
    The nonlinear dofs reductor is a reductor aiming at reducing the system to the nonlinear dofs only while using a linear solver to obtain the solution for the linear dofs. 

    The reduction method is inspired by the PhD of Colaitis but adapted to the closure equation included in pyHarm before the reduction layer. 
        - Expansion solve the linear problem associated in order to return the solutions for the linear dofs based on the values of the nonlinear dofs
        - Jacobian matrix is completed with the contribution of linear dofs for the closure equation and the rest of the Jacobian by solving two linear problems
    
    Attributes : 
        factory_keyword (str): name to use as input when willing to create an instance of this object.
        solve (Callable): solve from scipy.linalg module.
        lu_factor (Callable): lu_factor from scipy.linalg module.
        lu_solve (Callable): lu_solve from scipy.linalg module.
    """
    factory_keyword : str = "NLdofs"
    """str: keyword that is used to call the creation of this class in the factory."""

    default={}
    """dict: dictionary containing default parameters."""

    def __post_init__(self):
        from scipy.linalg import lu_factor, lu_solve, solve
        self.solve = solve
        self.lu_factor = lu_factor
        self.lu_solve = lu_solve
        self.init_dofs = False
        self.data = getCustomOptionDictionary(self.data,self.default)
        pass
        
    def update_reductor(self, xpred, J_f, expl_dofs, system,*args):
        """
        Updates the reducer for the new point.

        Args:
            xpred (np.ndarray): point predicted as the new starting point for the next iteration of the analysis process.
            J_f (np.ndarray): full jacobian with respect to displacement and angular frequency (contains the correction equation residual).
            expl_dofs (pd.DataFrame): explicit dof DataFrame created by the ABCSystem studied.
            system (ABCSystem): System studied.

        Attributes:
            J_add (np.ndarray): Updated jacobian to add onto the non-linear dofs.

        Returns:
            np.ndarray: same displacement vector given as input after passing through the reductor
            np.ndarray: same jacobian matrix given as input after passing through the reductor
            pd.DataFrame: same explicit dof DataFrame after passing through the reductor

        """
        self._init_dofs(expl_dofs,system)
        self.J_add = self._update_J_add(J_f)
        return self.reduce_vector(xpred), self.reduce_matrix(J_f), self.output_expl_dofs
    
    def expand(self,q:np.ndarray) -> np.ndarray:
        """
        Makes a linear solve onto the linear part of the residual and jacobian in order to give the necessary displacement on the linear dofs.

        Args:
            q (np.ndarray): vector of reduced displacement.

        Returns:
            np.ndarray: vector of original displacement with the linear dof being exact solution.
        """
        x = np.zeros(len(self.expl_dofs)+1)
        x[self.nl_dofs] = q[:-1]
        x[-1] = q[-1]
         
        xnull = np.zeros(len(self.expl_dofs)+1)
        xnull[-1] = q[-1]
        xf = self.system._expand_q(xnull) # expand to the full system size
        xf += self.system._complete_x(self.system.LC, xf) # add the Kinematic conditions
        Fe_l = self.system._residual(self.system.LE_extforcing, xf) 
        Fe_l += self.system._residual(self.system.LE_linear, xf) # residual is needed since load with disp
        Jx,Jom=self.system._jacobian(self.system.LE_linear, xf)
        Fe_l += self.system._complete_R(self.system.LC, Fe_l, xf) # complete system
        Fe_l = (self.system.kick_kc_dofs @ Fe_l)[self.lin_dofs] # and cut it
        
        # same goes for jacobian
        jxadd, jomadd = self.system._complete_J(self.system.LC, Jx, Jom, xf)
        Jx += jxadd
        Jom += jomadd
        Jx = self.system.kick_kc_dofs @ Jx @ self.system.kick_kc_dofs.T
        Jom = self.system.kick_kc_dofs @ Jom
        
        J_ll, J_ln, J_nl = self._extract_blocks(Jx,self.lin_dofs,self.nl_dofs)
        x[self.lin_dofs] = self.solve(J_ll,(- Fe_l - J_ln@q[:-1]))
        return x
    
    def _extract_blocks(self,J,i,j):
        """
        Extracts blocks of the jacobian matrix.

        Args:
            J (np.ndarray): vector of displacement.
            i (np.ndarray): vector of index.
            j (np.ndarray): vector of index.

        Returns:
            np.ndarray: jacobian block restrained to the meshgrid of indexes i,i.
            np.ndarray: jacobian block restrained to the meshgrid of indexes i,j.
            np.ndarray: jacobian block restrained to the meshgrid of indexes j,i.
        """
        Jii, Jij, Jji = J[self._get_mshg(i,i)],J[self._get_mshg(i,j)],J[self._get_mshg(j,i)]
        return Jii, Jij, Jji
    
    def reduce_vector(self,x:np.ndarray) -> np.ndarray:
        """
        Transforms the displacement vector.

        Args:
            x (np.ndarray): vector of displacement.

        Returns:
            np.ndarray: vector of reduced displacement.
        """
        q = self.phi_reduce @ x
        return q
    
    def reduce_matrix(self,dJdxom:np.ndarray,*args) -> np.ndarray:
        """
        From original jacobian, performs the transformation to get the reduced matrix while adding the linear influence.

        Args:
            dJdxom (np.ndarray): full size jacobian matrix with respect to displacement and angular frequency.

        Returns:
            np.ndarray: reduced jacobian matrix with respect to displacement and angular frequency.
        """
        self.J_add = self._update_J_add(dJdxom)
        return self.phi_reduce @ dJdxom @ self.phi_reduce.T - self.J_add
    
    
        
    def _update_J_add(self,dJdxom):
        """
        Updates the jacobian block to add corresponding to the linear influence.

        Args:
            dJdxom (np.ndarray): full size jacobian matrix with respect to displacement and angular frequency.

        Returns:
            np.ndarray: jacobian to add corresponding to the linear influence.
        """
        J_ll, J_ln, J_nl = self._extract_blocks(dJdxom,self.lin_dofs,self.nl_dofs)
        Jom_l = dJdxom[self._get_mshg(self.lin_dofs,self.om_dofs)]
        Cl = dJdxom[self._get_mshg(self.om_dofs,self.lin_dofs)]
        J_ll_T_lu,J_ll_T_piv = self.lu_factor(J_ll.T)
        JnlJllm1 = self._get_AijAiim1LU((J_ll_T_lu,J_ll_T_piv),J_nl)
        ClJllm1 = self._get_AijAiim1LU((J_ll_T_lu,J_ll_T_piv),Cl)
        J_add = self._add_omega(JnlJllm1@J_ln,
                               JnlJllm1@Jom_l,
                               ClJllm1@J_ln,
                               ClJllm1@Jom_l)
        
        return J_add
        
    def _get_AijAiim1(self,Aii,Aij):
        """
        Obtains the linear solution to Aii.T,Aij.T and transposes the results.

        Args:
            Aii (np.ndarray): square matrix block of the jacobian.
            Aij (np.ndarray): rectangle matrix block of the jacobian.

        Returns:
            np.ndarray: linear solution to Aii.T,Aij.T and transpose the results.
        """
        C = np.linalg.solve(Aii.T,Aij.T)
        return np.transpose(C)
        
    def _get_AijAiim1LU(self,AiiTLU,Aij):
        """
        Obtains the linear solution to Aii.T,Aij.T and transposes the results but uses the LU decomposition of Aii.

        Args:
            AiiTLU (np.ndarray): LU matrix decomposition of a square transpose block of the jacobian.
            Aij (np.ndarray): rectangle matrix block of the jacobian.

        Returns:
            np.ndarray: linear solution to AiiTLU,Aij.T and transpose the results.
        """
        C = self.lu_solve(AiiTLU,Aij.T)
        return np.transpose(C)
        
    def _get_mshg(self,lines,columns):
        """
        Obtains the meshgrid for getting the block of matrix.

        Args:
            lines (np.ndarray): indexes of lines.
            columns (np.ndarray): indexes of columns.

        Returns:
            tuple[np.ndarray,np.ndarray]: meshgrid of the block.
        """
        return tuple(np.meshgrid(lines,columns,indexing="ij"))
    
    def _add_omega(self,J,Jom,Cn,Cw):
        """
        Constructs a matrix by blocks.

        Args:
            J (np.ndarray): jacobian with respect to displacement.
            Jom (np.ndarray): jacobian with respect to angular frequency.
            Cn (np.ndarray): correction jacobian with respect to displacement.
            Cw (np.array): correction jacobian with respect to angular frequency.

        Returns:
            np.ndarray: matrix built with the provided blocks.
        """
        J_add = np.block([[J, Jom],
                          [Cn,Cw]])
        return J_add
    
    def _add_omega_vec(self,R):
        """
        Extends a residual vector by adding a 0 at the end.

        Args:
            R (np.ndarray): residual vector.
            Cw (np.array): correction jacobian with respect to angular frequency.

        Returns:
            np.ndarray: residual vector augmented with a component of 0 at the end.
        """
        R_add = np.block([R, np.zeros(1,)])
        return R_add
    
    def _init_dofs(self,expl_dofs,system):
        """
        Constructs the phi matrix that suppresses the linear dofs when applied.

        Args:
            expl_dofs (pd.DataFrame): explicit dof DataFrame created by the ABCSystem studied.
            system (ABCSystem): System studied.

        Attributes:
            init_dofs (bool): initialisation of the indexes of linear and non-linear dof done.
            phi_reduce (np.ndarray): transformation matrix that cuts the linear dofs.
        """
        if not self.init_dofs : 
            self.system = system
            self.expl_dofs = expl_dofs
            self.lin_dofs = self.expl_dofs[self.expl_dofs["NL"] == 0].index
            self.nl_dofs = self.expl_dofs[self.expl_dofs["NL"] == 1].index
            self.om_dofs = np.array([-1])
            self.phi_reduce = np.zeros((len(self.nl_dofs)+1,len(self.expl_dofs)+1))
            self.phi_reduce[np.arange(len(self.nl_dofs)),np.array(self.nl_dofs)] = 1
            self.phi_reduce[-1,-1] = 1
            self.output_expl_dofs = self._get_output_expl_dofs()
            self.init_dofs = True
        else : 
            pass
        
    def _get_output_expl_dofs(self,):
        """
        Obtains the modified explicit dof DataFrame after passing through the reducer.

        Returns:
            np.ndarray: Modified explicit dof DataFrame after passing through the reducer.
        """
        output_expl_dofs = self.expl_dofs[self.expl_dofs["NL"]==1].reset_index(drop=True)
        return output_expl_dofs