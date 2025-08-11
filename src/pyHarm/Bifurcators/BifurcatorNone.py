from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
from pyHarm.Solver import SystemSolution


class BifurcatorNone(ABCBifurcator):
    """Dummy Bifurcator that does not want to do his job."""

    factory_keyword: str = "none"
    """str: Concrete class name used by the factory to instantiate it."""

    def __post_init__(self):
        pass

    def detect(self, last_sol: SystemSolution) -> bool:
        False

    def localize(self):
        pass

    def track(self):
        pass
