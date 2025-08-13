from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator, default_detect
from pyHarm.Solver import SystemSolution


class BifurcatorJump(ABCBifurcator):
    """Simply jumps over detected bifurcations, but do not treat them."""

    factory_keyword: str = "jump"
    """str: Concrete class name used by the factory to instantiate it."""

    default_options = {
        "verbose": True,
        "blind_spot": 40,
    }
    """dict: dictionary containing the default options for this concrete Bifurcator.

    It contains a 'verbose' keyword that can be set to True (default)
    if information about detection of bifurcations is to be displayed during solving.
    """

    def __post_init__(self):
        self.opts = getCustomOptionDictionary(self.opts, self.default_options)

    def detect(self, sol: SystemSolution) -> bool:
        return default_detect(self, sol)

    def localize(self):
        pass

    def track(self):
        pass
