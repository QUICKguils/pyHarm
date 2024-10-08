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


class Solver_NewtonRaphson(ABCNLSolver):
    """This nonlinear solver is an implementation of iterative Newton Raphson solving procedure.
    
    Attributes:
        factory_keyword (str): keyword that is used to call the creation of this class in the system factory.
        solver_options (dict): dictionary containing other options for creation of the solver class.
        residual (Callable): function that returns the residual vector of the system to be solved.
        jacobian (Callable): function that returns the jacobian matrix of the system to be solved.
        solver_options_root (dict): dictionary containing options for the root function.
        extcall (Callable): root function of scipy.optimize.
    """
    
    factory_keyword : str = "NewtonRaphson"
    """str: keyword that is used to call the creation of this class in the system factory."""

    default = {"tol_residual":1e-8          ,
               "tol_delta_x" :1e-8          ,
               "max_iter"    :50            } # Maximum iterations accepted before confirming divergence
    """dict: dictionary containing the default solver_options"""


    def __post_init__(self):
        from scipy.linalg import solve as solve
        self.linearsolve = solve
        self.solver_options = getCustomOptionDictionary(self.solver_options,self.default)
               
    def Solve(self,sol:SystemSolution,SolList:list) -> SystemSolution:
        """
        Runs the solver.

        Args:
            sol (SystemSolution): SystemSolution that contains the starting point.
            SolList (SystemSolution): list of previously solved solutions.
        
        Returns:
            sol (SystemSolution): SystemSolution solved and completed with the output information.
        """
        self.x = sol.x_start
        self.xprec = sol.x_start
        self.FXk = self.residual(sol.x_start, sol)
        self.AXk = self.jacobian(sol.x_start, sol)
        self.iter = 0
        self.status = 1
        while ((np.linalg.norm(self.FXk)>=self.solver_options["tol_residual"]) or\
             (np.linalg.norm(self.x - self.xprec)>=self.solver_options["tol_delta_x"])):
            self.deltak = self.linSysdeltak()
            self.xprec = copy.deepcopy(self.x)
            self.x -= self.deltak
            self.FXk = self.residual(self.x, sol)
            self.AXk = self.jacobian(self.x, sol)
            self.iter+=1
            if self.iter>=self.solver_options["max_iter"]:
                self.status=5
                self.CompleteSystemSolution(sol,SolList)
                return sol
        self.CompleteSystemSolution(sol,SolList)
        return sol


    def CompleteSystemSolution(self, sol, SolList):
        """
        Function that allows to retrieve information of interest

        Args:
            sol (SystemSolution): SystemSolution that contains the starting point.
            SolList (SystemSolution): list of previously solved solutions.
        """
        sol.x_red = copy.deepcopy(self.x)
        sol.R_solver = copy.deepcopy(self.FXk)
        sol.iter_numb = self.iter
        sol.flag_R = True 
        sol.flag_J = True
        sol.flag_J_f = True
        sol.J_f = self.jacobian(sol.x,sol)
        sol.flag_intosolver = True
        if self.status == 1:
            sol.flag_accepted = True
        sol.status_solver = self.status


    def linSysdeltak(self):
        """
        Calculation of the 'deltak' correction to apply to the current iteration in order to converge towards the solution.

        Returns:
            self.extcall_newton(matA,matB): Correction 'deltak'
        """
        return self.linearsolve(self.AXk,self.FXk)