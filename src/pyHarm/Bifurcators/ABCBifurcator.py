import abc

import numpy as np

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Solver import SystemSolution


class ABCBifurcator(abc.ABC):
    """Abstract class for the bifurcation detectors.

    An `ABCBifurcator` handles bifurcation detection, localization and tracking.

    Attributes:
        predictor_options (dict):
          dictionary containing the kwargs and completed using the default
          options if the keywords are missing.
        flag_print (bool):
          information are printed during the analysis if True.
    """

    @property
    @abc.abstractmethod
    def factory_keyword(self) -> str:
        """str: Concrete class name used by the factory to instantiate it."""
        pass

    default_options = {
        "verbose": True,
    }
    """dict: set of default parameters for the system class if not given in the input argument.

    It contains a 'verbose' keyword that can be set to True (default) if
    information about detection of bifurcations is to be displayed during
    solving.
    """

    def __init__(self, **kwargs):
        self.bifurcator_options = getCustomOptionDictionary(kwargs, self.default_options)
        self.flag_print = self.bifurcator_options["verbose"]

    @abc.abstractmethod
    def detect(
        self, sollist: list[SystemSolution], ds: float, k_imposed=None
    ) -> tuple[np.ndarray, SystemSolution, float]:
        pass

    @abc.abstractmethod
    def localize(
        self, sollist: list[SystemSolution], ds: float, k_imposed=None
    ) -> tuple[np.ndarray, SystemSolution, float]:
        pass

    @abc.abstractmethod
    def track(self):
        pass
