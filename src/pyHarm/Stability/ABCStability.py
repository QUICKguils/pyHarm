import abc
from enum import Enum

import numpy as np

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Solver import SystemSolution


class ABCStability(abc.ABC):
    """Abstract class for the stability analyzers.

    Attributes:
        stability_options (dict):
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

    def __init__(self, stability_options):
        self.stability_options = getCustomOptionDictionary(stability_options, self.default_options)
        self.in_operation = False
        self.flag_print = self.stability_options["verbose"]
        self.__post_init__()

    @property
    @abc.abstractmethod
    def factory_keyword(self) -> str:
        """str: Concrete class name used by the factory to instantiate it."""
        pass

    @abc.abstractmethod
    def __post_init__(self):
        pass
