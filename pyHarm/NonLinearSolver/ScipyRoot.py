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

class Solver_ScipyRoot(ABCNLSolver):
    """This nonlinear solver is a wrapping of scipy.root nonlinear solver adapted to the interfaces of pyHarm.
    
    Attributes:
        factory_keyword (str): keyword that is used to call the creation of this class in the system factory.
        solver_options (dict): dictionary containing other options for creation of the solver class.
        residual (Callable): function that returns the residual vector of the system to be solved.
        jacobian (Callable): function that returns the jacobian matrix of the system to be solved.
        solver_options_root (dict): dictionary containing options for the root function.
        extcall (Callable): root function of scipy.optimize.
        end_status_accepted (list): list of accepted ended status of the non-linear solver.
    """
    factory_keyword : str = "scipyroot"
    """str: keyword that is used to call the creation of this class in the system factory."""

    name = "scipyroot solver"

    default = {"root":{"method":"hybr","options":{"diag":None}},
               "end_status_accepted":[1],\
                "residual_tolerance":1e-4}
    """dict: dictionary containing the default solver_options"""
    
    def __post_init__(self):
        from scipy.optimize import root
        solver_options = self.solver_options
        self.extcall = root
        self.solver_options = getCustomOptionDictionary(solver_options,self.default)
        self.solver_options_root = getCustomOptionDictionary(self.solver_options["root"],self.default["root"])
        self.end_status_accepted = self.solver_options["end_status_accepted"]
        self.residual_tolerance = self.solver_options["residual_tolerance"]

    def Solve(self,sol:SystemSolution,SolList:list[SystemSolution]) -> SystemSolution:
        """Runs the solver.

        Args:
            sol (SystemSolution): SystemSolution that contains the starting point.
            SolList (SystemSolution): list of previously solved solutions.
        
        Returns:
            sol (SystemSolution): SystemSolution solved and completed with the output information.
        """
        self.precond_kwargs = {"jacobian":self.jacobian}
        S = self.extcall(self.residual, sol.x_start, args=(sol), jac=self.jacobian,**self.solver_options_root)
        self._complete_solution(S,sol,SolList)
        return sol

    def solution_accepted(self, S, sol):
        """Updates the flag_accepted attribute of the SystemSolution.

        Args:
            S (SystemSolution): output of root function.
            sol (SystemSolution): SystemSolution that ran into the solver..
        
        Attributes:
            flag_accepted (bool): SystemSolution in output is considered accepted if True.
        """
        if  isinstance(sol,FirstSolution) : 
            sol.flag_accepted = True
        elif (S.status == 1):
            sol.flag_accepted = True
        elif (S.status in self.end_status_accepted):
            sol.flag_accepted = self.check_residual_tol(sol)

    def check_residual_tol(self, sol): 
        """Checks if residual at the output is below the residual tolerance.

        This check is done only if the solver gave an ending status different than 1 (solved with no problem by root function)

        Args:
            S (SystemSolution): output of root function.
            sol (SystemSolution): SystemSolution that ran into the solver.
            SolList (SystemSolution): list of previously solved solutions.
        
        """
        if np.linalg.norm(sol.R_solver) <= self.residual_tolerance : 
            return True
        else : 
            return False

    def _complete_solution(self,S,sol:SystemSolution,SolList:list[SystemSolution]):
        """Completes the SystemSolution class with solver informations.

        Args:
            S (SystemSolution): output of root function.
            sol (SystemSolution): SystemSolution that ran into the solver.
            SolList (SystemSolution): list of previous solved solutions.
        
        """
        sol.x_red = copy.deepcopy(S.x)
        sol.R_solver = copy.deepcopy(S.fun)
        sol.flag_intosolver = True
        self.solution_accepted(S, sol)
        sol.status_solver = S.status
        sol.message_solver = S.message