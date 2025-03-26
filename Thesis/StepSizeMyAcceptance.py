import numpy as np

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Solver import SystemSolution
from pyHarm.StepSizeRules.ABCStepSizeRule import ABCStepSizeRule


class StepSizeMyAcceptance(ABCStepSizeRule):
    """
    Step size is divided if last solution is not accepted or multiplied if a
    number of consecutive accepted solutions is reached.

    Attributes:
        default_options (dict): contains default step size options concerning
          consecutive_accept if not provided during creation.
        consecutive_accept (int): number of consecutive accept before
          increasing step size.
    """

    name = "accepted step size adaptation"
    factory_keyword: str = "myacceptance"
    default_options = {"consecutive_accept": 5}

    def __init__(self, bounds: list[float, float], **kwargs):
        super().__init__(bounds)
        self.stepsize_options = getCustomOptionDictionary(
            kwargs.get("stepsize_options", dict()), self.default_options
        )
        self.consecutive_accept = self.stepsize_options["consecutive_accept"]

    def getStepSize(self, ds: float, sollist: list[SystemSolution], **kwargs) -> float:
        """Returns the step size to be used for the prediction step of the analysis.

        Args:
            ds (float): Current step size.
            sollist (list[SystemSolution]): list of SystemSolution returned during the analysis.

        Returns:
            float: updated step size.
        """
        if (not sollist[-1].flag_accepted) and (ds > self.ds_min):
            ds /= 5
        try:
            acc = np.array([sol.flag_accepted for sol in sollist[-self.consecutive_accept : :]])
            if np.sum(acc) == self.consecutive_accept and ds < self.ds_max:
                ds *= 1.3
        except Exception:
            pass
        if ds < self.ds_min or ds > self.ds_max:
            ds = self.ProjectInBounds(ds)  # shouldn't be necessary
        return ds
