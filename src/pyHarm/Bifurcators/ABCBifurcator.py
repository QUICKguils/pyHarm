import abc
from enum import Enum

import numpy as np
from scipy import linalg

from pyHarm.Solver import FirstSolution, SystemSolution


class BifurcationType(Enum):
    FOLD = "fold bifurcation"
    BRANCHING = "branching point bifurcation"
    UNKNOWN = "potential higer-order bifurcation"


class ABCBifurcator(abc.ABC):
    """Abstract class for the bifurcation detectors.

    An `ABCBifurcator` handles bifurcation detection, localization and tracking.

    Attributes:
        bifurcator_options (dict):
          dictionary containing the kwargs and completed using the default
          options if the keywords are missing.
        flag_print (bool):
          information are printed during the analysis if True.
    """

    def __init__(self, bifurcator_options):
        self.opts = bifurcator_options
        self.in_operation = False
        self.__post_init__()

    @abc.abstractmethod
    def __post_init__(self):
        pass

    @property
    @abc.abstractmethod
    def factory_keyword(self) -> str:
        """str: Concrete class name used by the factory to instantiate it."""
        pass

    @abc.abstractmethod
    def detect(self, sol: SystemSolution) -> bool:
        pass

    @abc.abstractmethod
    def localize(
        self, SolList: list[SystemSolution], ds: float, k_imposed=None
    ) -> tuple[np.ndarray, SystemSolution, float]:
        pass

    @abc.abstractmethod
    def track(self):
        pass


def default_detect(self, sol: SystemSolution) -> bool:
    """yee"""

    J = sol.get_jacobian("full")
    sol.det_Jf = linalg.det(J)
    sol.det_Jx = linalg.det(J[:-1, :-1])

    if isinstance(sol, FirstSolution):
        return

    prev_sol = sol.precedent_solution

    # TODO: separate detection and simple jumps
    if np.sign(sol.det_Jx) != np.sign(prev_sol.det_Jx):
        sol.flag_bifurcation = True
        if np.sign(sol.det_Jf) == np.sign(prev_sol.det_Jf) or np.sign(sol.det_Jf) != np.sign(
            sol.det_Jx
        ):
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
