import copy

from scipy import linalg

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.NonLinearSolver.ABCNonLinearSolver import ABCNLSolver
from pyHarm.Solver import SystemSolution


class SolverMyNewtonRaphson(ABCNLSolver):
    """This nonlinear solver is an implementation of iterative Newton Raphson solving procedure.

    Attributes:
        factory_keyword (str): keyword that is used to call the creation of this class in the system factory.
        solver_options (dict): dictionary containing other options for creation of the solver class.
        residual (Callable): function that returns the residual vector of the system to be solved.
        jacobian (Callable): function that returns the jacobian matrix of the system to be solved.
    """

    factory_keyword: str = "MyNewtonRaphson"
    """str: Concrete class name used by the factory to instantiate it."""

    default = {
        "tol_residual": 1e-8,
        "tol_delta_x": 1e-8,
        "max_iter": 5,
        "pseudo": False,  # True -> don't update Jacobian at each corr step
        "pert": 0.0,  # perturbation introduced in the residual equation
    }
    """dict: dictionary containing the default solver_options"""

    def __post_init__(self):
        self.solver_options = getCustomOptionDictionary(self.solver_options, self.default)

    def solve(self, sol: SystemSolution):
        """Run the solver.

        Args:
            sol (SystemSolution): SystemSolution that contains the starting point.

        Writes:
            sol (SystemSolution): SystemSolution solved and completed with the output information.
        """
        self.flag_solved = False
        self.pert = self.solver_options["pert"]
        self.x = sol.x_start
        x_prec = sol.x_start
        self.H = self.residual(sol.x_start, sol)[:-1]
        self.J = self.jacobian(sol.x_start, sol)[:-1, :]

        for self.iter in range(1, self.solver_options["max_iter"] + 1):
            x_prec = copy.deepcopy(self.x)
            corr = linalg.pinv(self.J) @ (self.H - self.pert)
            self.x -= corr
            self.H = self.residual(self.x, sol)[:-1]
            if not self.solver_options["pseudo"]:
                self.J = self.jacobian(self.x, sol)[:-1, :]

            if (
                linalg.norm(self.H - self.pert) < self.solver_options["tol_residual"]
                and linalg.norm(self.x - x_prec) < self.solver_options["tol_delta_x"]
            ):
                self.flag_solved = True
                break

        self.complete_solution(sol)

    def complete_solution(self, sol):
        """Function that allows to retrieve information of interest.

        Args:
            sol (SystemSolution): SystemSolution that contains the starting point.
        """
        sol.x_red = copy.deepcopy(self.x)
        sol.R_solver = copy.deepcopy(self.H)
        sol.J_f = self.jacobian(sol.x, sol)
        sol.niter = self.iter
        sol.flag_R = True
        sol.flag_J = True
        sol.flag_J_f = True
        sol.flag_intosolver = True
        sol.flag_accepted = self.flag_solved
