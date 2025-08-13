import numpy as np
from scipy import linalg

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator, BifurcationType
from pyHarm.NonLinearSolver.SolverNewtonRaphson import SolverNewtonRaphson
from pyHarm.Solver import FirstSolution, SystemSolution


class BifurcatorPerturbation(ABCBifurcator):
    """Bifurcation handling through small perturbations."""

    factory_keyword: str = "perturbation"
    """str: Concrete class name used by the factory to instantiate it."""

    default_options = {
        "verbose": True,
        "blind_spot": 40,
        "back_steps": 3,
        "pert_steps": 5,
        "perturbation": 1E-4,
    }
    """dict:
      Set of default parameters for this concrete Bifurcator if not given in the input argument.

    It contains a 'blind_spot' keyword that sets a blind angle to avoid the predictor to search in
    the inwards direction.
    It contains a 'pert_step' keyword that
    """

    def __post_init__(self):
        self.opts = getCustomOptionDictionary(self.opts, self.default_options)

    def detect(self, sol: SystemSolution):
        """Only raise flag_bifurcation for branching, no need to invert sign_ds"""

        J = sol.get_jacobian("full")
        sol.det_Jf = linalg.det(J)
        sol.det_Jx = linalg.det(J[:-1, :-1])

        if isinstance(sol, FirstSolution):
            return

        prev_sol = sol.precedent_solution

        # TODO: separate detection and simple jumps
        detected = False
        if np.sign(sol.det_Jx) != np.sign(prev_sol.det_Jx):
            if np.sign(sol.det_Jf) == np.sign(prev_sol.det_Jf) or np.sign(sol.det_Jf) != np.sign(
                sol.det_Jx
            ):
                sol.bifurcation_type = BifurcationType.FOLD
                sol.flag_bifurcation = True
            else:
                sol.bifurcation_type = BifurcationType.BRANCHING
                sol.flag_bifurcation = True
                detected = True
        elif np.dot(prev_sol.dir, sol.dir) < -np.cos(np.deg2rad(self.opts["blind_spot"] / 2)):
            sol.bifurcation_type = BifurcationType.UNKNOWN
            sol.flag_bifurcation = True
            detected = True

        if self.opts["verbose"] and sol.flag_bifurcation:
            print(f"Warning: a {sol.bifurcation_type.value} was detected")

        return detected

    def localize(self, sol_list: list[SystemSolution], solver: SolverNewtonRaphson):
        """Here, localize atcually localized previous valid solution from which to restart from."""
        # Go a few steps back
        go_back = (sol for sol in reversed(sol_list) if sol.flag_accepted)
        for _ in range(self.opts["back_steps"]):
            next(go_back)
        back_sol = next(go_back)
        bback_sol = next(go_back)
        sol_list.append(bback_sol)
        sol_list.append(back_sol)
        # WARN: check is sol.precedent_solution is coherent with that

        # Set Newton solver so that it now solves the perturbed equations.
        solver.solver_options["pert"] = self.opts["perturbation"]

        # Set tracking
        self.step_counter = self.opts["pert_steps"]
        self.in_operation = True

    def track(self, solver: SolverNewtonRaphson):
        """Here, the track method basically acts as a counter."""
        if self.step_counter == 0:
            # Reset Newton solver so that it solves the unperturbed equations.
            solver.solver_options["pert"] = 0.0
            self.in_operation = False
        self.step_counter -= 1
