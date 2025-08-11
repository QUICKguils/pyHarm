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
        "pert_steps": 4,
    }
    """dict:
      Set of default parameters for this concrete Bifurcator if not given in the input argument.

    It contains a 'blind_spot' keyword that sets a blind angle to avoid the predictor to search in
    the inwards direction.
    It contains a 'pert_step' keyword that
    """

    def __post_init__(self):
        self.opts = getCustomOptionDictionary(self.opts, self.default_options)

    def detect(self, sol: SystemSolution):
        pass

    def localize(self):
        pass

    def track(self):
        pass
