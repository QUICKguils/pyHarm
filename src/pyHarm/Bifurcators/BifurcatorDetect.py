import numpy as np
from scipy import linalg

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator, BifurcationType
from pyHarm.Solver import FirstSolution, SystemSolution


class BifurcatorDetect(ABCBifurcator):
    """Simply detect the presence of bifurcation, but do not treat them."""

    factory_keyword: str = "detect"
    """str: Concrete class name used by the factory to instantiate it."""

    default_options = {
        "verbose": True,
        "blind_spot": 40,
    }
    """dict: dictionary containing the default options for this concrete Bifurcator.

    It contains a 'verbose' keyword that can be set to True (default)
    if information about detection of bifurcations is to be displayed during solving.
    """

    def __post_init__(self):
        self.opts = getCustomOptionDictionary(self.opts, self.default_options)

    def detect(self, sol: SystemSolution) -> bool:
        """yee"""

        J = sol.get_jacobian("full")
        sol.det_Jf = linalg.det(J)
        sol.det_Jx = linalg.det(J[:-1, :-1])

        if isinstance(sol, FirstSolution):
            return

        prev_sol = sol.precedent_solution

        # TODO: separate detection and simple jumps
        if np.sign(sol.det_Jx) != np.sign(prev_sol.det_Jx):
            print("=== HERE ===")
            sol.flag_bifurcation = True
            if np.sign(sol.det_Jf) == np.sign(prev_sol.det_Jf) or np.sign(sol.det_Jf) != np.sign(sol.det_Jx):
                sol.bifurcation_type = BifurcationType.FOLD
            else:
                sol.bifurcation_type = BifurcationType.BRANCHING
                sol.sign_ds *= -1
            if self.opts["verbose"]:
                print(f"Warning: a {sol.bifurcation_type.value} was detected")
        elif np.dot(prev_sol.dir, sol.dir) < -np.cos(np.deg2rad(self.opts["blind_spot"] / 2)):
            sol.flag_bifurcation = True
            sol.sign_ds *= -1
            if self.opts["verbose"]:
                print("A potential higher order bifurcation is detected")

        return sol.flag_bifurcation

    def localize(self):
        pass

    def track(self):
        pass
