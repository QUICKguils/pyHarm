import numpy as np
from scipy import linalg

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator, BifurcationType
from pyHarm.Solver import FirstSolution, SystemSolution


class BifurcatorPerturbation(ABCBifurcator):
    """Bifurcation handling through small perturbations."""

    factory_keyword: str = "perturbation"
    """str: Concrete class name used by the factory to instantiate it."""

    default_options = {
        "blind_spot": 40,
    }
    """dict: dictionary containing the default options for this concrete Bifurcator."""

    def __post_init__(self):
        self.bifurcator_options = getCustomOptionDictionary(self.bifurcator_options, self.default_options)

    # XXX: heavy to compute dets. Consider bordering techniques
    # FIX: still written for the Predictor class
    def detect(self, last_sol: SystemSolution):
        """
        Makes a bifurcation detection analysis computing determinant of jacobian matrix
        and analyzing change of sign.

        Args:
            last_sol (SystemSolution): Previously accepted point in direct link with the actual solved point.

        Attributes:
            sign_ds (float): Attribute is modified if a fold bifurcation is detected.
            bifurcation_type (BifurcationType): Attribute is assigned if a bifurcation is detected.
        """
        J = last_sol.get_jacobian("full")
        last_sol.det_Jf = linalg.det(J)
        last_sol.det_Jx = linalg.det(J[:-1, :-1])

        if isinstance(last_sol, FirstSolution):
            return

        prev_sol = last_sol.precedent_solution

        if np.sign(last_sol.det_Jx) != np.sign(prev_sol.det_Jx):
            last_sol.flag_bifurcation = True
            if np.sign(last_sol.det_Jf) == np.sign(prev_sol.det_Jf) or np.sign(last_sol.det_Jf) != np.sign(last_sol.det_Jx):
                last_sol.bifurcation_type = BifurcationType.FOLD
            else:
                last_sol.bifurcation_type = BifurcationType.BRANCHING
                self.sign_ds *= -1
            if self.flag_print:
                print(f"Warning: a {last_sol.bifurcation_type.value} was detected")
        elif (
            not isinstance(last_sol, FirstSolution)
            and np.dot(prev_sol.dir, last_sol.dir) < -np.cos(np.deg2rad(self.predictor_options["blind_spot"]/2))
        ):
            last_sol.flag_bifurcation = True
            self.sign_ds *= -1
            if self.flag_print:
                print("A potential higher order bifurcation is detected")

    def localize(self):
        pass

    def track(self):
        pert = 1E-5 * np.ones(self.H.shape)
        pass
