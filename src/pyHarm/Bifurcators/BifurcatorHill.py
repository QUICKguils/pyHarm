from enum import Enum

import numpy as np
from scipy import linalg

from pyHarm.DynamicOperator import nabla
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
        J = lstpt.get_jacobian("full")
        # XXX: heavy to compute dets. Consider bordering techniques
        det_J_f = linalg.det(J)
        det_J_x = linalg.det(J[:-1, :-1])
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

    def _compute_Hill_matrix(self, last_point: SystemSolution):
        J = last_point.get_jacobian("full")
        h_z = J[:-1, :-1]
        delta_1 = np.kron(Om * nabla(), 2 * M) + np.kron(np.eye(2 * nh + 1), C)
        delta_2 = np.kron(np.eye(2 * nh + 1), M)
        delta_2_inv = linalg.inv(delta_2)

        return np.array(
            [[delta_2_inv * delta_1, delta_2_inv * h_z], [np.eye(2 * nh + 1), np.zeros(2 * nh + 1)]]
        )
