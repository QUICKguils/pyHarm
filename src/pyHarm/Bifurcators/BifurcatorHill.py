from enum import Enum

import numpy as np
from scipy import linalg

from pyHarm.DynamicOperator import nabla
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
from pyHarm.Solver import SystemSolution
from pyHarm.Systems import ABCSystem


# WARN: only works if one and only one substructure is defined
# TODO: made a check for that
class BifurcatorHill(ABCBifurcator):
    """Branching predictions through the Hill's method."""

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
        pass

    def localize(self):
        pass

    def track(self):
        pass

    def _compute_Hill_matrix(self, last_solution: SystemSolution, system: ABCSystem):
        J = last_solution.get_jacobian("full")
        Om = last_solution.x[-1]

        M = system.LE[0].data["matrix"]["M"]
        C = system.LE[0].data["matrix"]["C"]
        nh = system.LE[0].nh


        h_z = J[:-1, :-1]
        delta_1 = np.kron(Om * nabla(nh), 2 * M) + np.kron(np.eye(2 * nh + 1), C)
        delta_2 = np.kron(np.eye(2 * nh + 1), M)
        delta_2_inv = linalg.inv(delta_2)

        return np.array(
            [[delta_2_inv * delta_1, delta_2_inv * h_z], [np.eye(2 * nh + 1), np.zeros(2 * nh + 1)]]
        )
