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

from pyHarm.NonLinearSolver.ABCNonLinearSolver import ABCNLSolver
import copy
import numpy as np
from pyHarm.Solver import FirstSolution,SystemSolution
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary


class Solver_MoorePenrose(ABCNLSolver):
    """This nonlinear solver is an implementation of iterative MoorePenrose solving procedure.
    
    Attributes:
        factory_keyword (str): keyword that is used to call the creation of this class in the system factory.
        solver_options (dict): dictionary containing other options for creation of the solver class.
        residual (Callable): function that returns the residual vector of the system to be solved.
        jacobian (Callable): function that returns the jacobian matrix of the system to be solved.
        solver_options_root (dict): dictionary containing options for the root function.
        extcall (Callable): root function of scipy.optimize.
    """
    
    factory_keyword : str = "MoorePenrose"
    """str: keyword that is used to call the creation of this class in the system factory."""

    default = {"tol_residual":1e-8,"tol_delta_x":1e-8,"max_iter":30}
    """dict: dictionary containing the default solver_options"""
    
    def __post_init__(self):
        from scipy.linalg import solve as solve
        self.extcall_moore = solve
        solver_options = self.solver_options
        self.solver_options = getCustomOptionDictionary(solver_options,self.default)
        
    def Solve(self,sol:SystemSolution,SolList:list) -> SystemSolution:
        """Runs the solver.

        Args:
            sol (SystemSolution): SystemSolution that contains the starting point.
            SolList (SystemSolution): list of previously solved solutions.
        
        Returns:
            sol (SystemSolution): SystemSolution solved and completed with the output information.
        """
        self.x = sol.x_start
        self.xprec = sol.x_start
        # Get residual and jacobian at starting point : 
        self.FXk = self.residual(sol.x_start, sol)[:-1]
        self.AXk = self.jacobian(sol.x_start, sol)[:-1,:]
        self.Vk = self._get_first_Vk(sol,self.AXk)
        self.iter = 0
        self.status = 1
        while np.linalg.norm(self.FXk)>=self.solver_options["tol_residual"] or\
             np.linalg.norm(self.x - self.xprec)>=self.solver_options["tol_delta_x"]:
            self.deltak = self.linSysdeltak()
            self.Tk = self.linSysTk()
            self.xprec = self.x
            self.x += -self.deltak
            self.Vk = (self.Vk - self.Tk.reshape(-1,1))/np.linalg.norm((self.Vk - self.Tk.reshape(-1,1)))
            self.FXk = self.residual(self.x, sol)[:-1]
            self.AXk = self.jacobian(self.x, sol)[:-1,:]
            self.iter+=1
            if self.iter>=self.solver_options["max_iter"]:
                self.status=5
                self._complete_solution(sol,SolList)
                return sol
        self._complete_solution(sol,SolList)
        return sol

    def _complete_solution(self, sol, SolList):
        """Completes the SystemSolution class with solver information.

        Args:
            S (SystemSolution): output of root function.
            sol (SystemSolution): SystemSolution that ran into the solver.
            SolList (SystemSolution): list of previously solved solutions.
        
        """
        sol.x_red = copy.deepcopy(self.x)
        sol.R_solver = copy.deepcopy(self.FXk)
        sol.flag_intosolver = True
        if ((self.status == 1) or (isinstance(sol,FirstSolution))) :
            sol.flag_accepted = True
        sol.status_solver = self.status
    
    def _get_first_Vk(self,sol,AXk) : 
        import scipy.linalg as spl
        J_x_T_qr = spl.qr(np.transpose(AXk[:-1,:]))
        Vk = (np.sign(J_x_T_qr[0][-1,-1]) * J_x_T_qr[0][:,-1]).reshape(-1,1)
        return Vk
    
    def linSysdeltak(self):
        matA = np.concatenate([self.AXk,self.Vk.T],axis=0)
        matB = np.concatenate([self.FXk,np.array([0])],axis=0)
        return self.extcall_moore(matA,matB)

    def linSysTk(self):
        matA = np.concatenate([self.AXk,self.Vk.T],axis=0) # np.dot(self.AXk,self.Vk)
        matB = np.concatenate([np.dot(self.AXk,self.Vk),np.array([[0]])],axis=0)
        return self.extcall_moore(matA,matB)