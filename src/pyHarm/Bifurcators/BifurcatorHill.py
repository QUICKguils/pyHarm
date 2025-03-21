from enum import Enum

import numpy as np
import scipy.linalg as spl

from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
from pyHarm.Solver import FirstSolution, SystemSolution


class BifurcationType(Enum):
    FOLD = "fold bifurcation"
    BRANCHING = "branching point bifurcation"
    NEIMARK_SACKER = "Neimark-Sacker bifurcation"


class BifurcatorHill(ABCBifurcator):
    """Branching predictions through the Hill's method."""
    # TODO: better docstring

    bifurcator_name = "Hill's method"

    factory_keyword: str = "hill"
    """str: Concrete class name used by the factory to instantiate it."""

    def detect(self, lstpt: SystemSolution):
        """
        Makes a bifurcation detection analysis computing determinant of
        jacobian matrix and analysing change of sign.

        Args:
            lstpt (SystemSolution): previously accepted point in direct link
              with the actual solved point.

        Attributes:
            sign_ds (float): Attribute is modified if a fold bifurcation is
              detected.
            bifurcation_type (BifurcationType): Attribute is assigned if a
              bifurcation is detected.
        """
        Jaco = lstpt.getJacobian("full")
        # XXX: heavy to compute dets. Consider bordering techniques
        det_J_f = spl.det(Jaco)
        det_J_x = spl.det(Jaco[:-1, :-1])
        lstpt.det_J_f = det_J_f
        lstpt.det_J_x = det_J_x
        if isinstance(lstpt, FirstSolution):
            det_J_x_prec = det_J_x
            det_J_f_prec = det_J_f
        else:
            det_J_x_prec = lstpt.precedent_solution.det_J_x
            det_J_f_prec = lstpt.precedent_solution.det_J_f

        if np.sign(det_J_x) != np.sign(det_J_x_prec):
            lstpt.flag_bifurcation = True
            if np.sign(det_J_f) == np.sign(det_J_f_prec) or np.sign(det_J_f) != np.sign(det_J_x):
                lstpt.bifurcation_type = BifurcationType.FOLD
                self.sign_ds *= -1
            else:
                lstpt.bifurcation_type = BifurcationType.BRANCHING

            if self.flag_print:
                print(f"Warning: a {lstpt.bifurcation_type.value} was detected")
                if lstpt.bifurcation_type is BifurcationType.FOLD:
                    print("--> path direction is reversed")

    def localize(self):
        pass

    def track(self):
        pass
