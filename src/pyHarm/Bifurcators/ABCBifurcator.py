import abc
from enum import Enum

from pyHarm.NonLinearSolver.SolverNewtonRaphson import SolverNewtonRaphson
from pyHarm.Solver import SystemSolution
from pyHarm.Systems.System import System


# TODO: at this point sol.flag_bifurcation should be unuseful. Clean that
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
    def localize(self, SolList: list[SystemSolution], solver: SolverNewtonRaphson, system: System):
        pass

    @abc.abstractmethod
    def track(self, solver: SolverNewtonRaphson):
        pass
