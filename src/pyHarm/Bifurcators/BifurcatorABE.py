import numpy as np
from scipy import linalg

from pyHarm.DynamicOperator import nabla
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
from pyHarm.Solver import SystemSolution
from pyHarm.Systems import ABCSystem


class BifurcatorABE(ABCBifurcator):
    """Branching predictions through the Hill's method."""

    # TODO: better docstring

    bifurcator_name = "Algebraic bifurcation equation"

    factory_keyword: str = "abe"
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
