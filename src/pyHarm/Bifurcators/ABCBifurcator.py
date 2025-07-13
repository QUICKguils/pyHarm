import abc
from enum import Enum

import numpy as np

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Solver import SystemSolution


class BifurcationType(Enum):
    FOLD = "fold bifurcation"
    BRANCHING = "branching point bifurcation"
    NEIMARK_SACKER = "Neimark-Sacker bifurcation"


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

    default_options = {
        "verbose": True
    }
    """dict: set of default parameters for the system class if not given in the input argument.

    It contains a 'verbose' keyword that can be set to True (default)
    if information about detection of bifurcations is to be displayed during solving.
    """

    def __init__(self, bifurcator_options):
        self.bifurcator_options = getCustomOptionDictionary(bifurcator_options, self.default_options)
        self.flag_print = self.bifurcator_options["verbose"]
        self.__post_init__()

    @property
    @abc.abstractmethod
    def factory_keyword(self) -> str:
        """str: Concrete class name used by the factory to instantiate it."""
        pass

    @abc.abstractmethod
    def __post_init__(self):
        pass

    @abc.abstractmethod
    def detect(
        self, SolList: list[SystemSolution], ds: float, k_imposed=None
    ) -> tuple[np.ndarray, SystemSolution, float]:
        pass

    @abc.abstractmethod
    def localize(
        self, SolList: list[SystemSolution], ds: float, k_imposed=None
    ) -> tuple[np.ndarray, SystemSolution, float]:
        pass

    @abc.abstractmethod
    def track(self):
        pass
