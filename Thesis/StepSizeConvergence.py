import numpy as np

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Solver import SystemSolution
from pyHarm.StepSizeRules.ABCStepSizeRule import ABCStepSizeRule


class StepSizeConvergence(ABCStepSizeRule):
    """Step size is scaled based on how mush the nonlinear solver struggled to converge.

    ds = M_opt/M * ds_prev
    where M is the number of iterations performed by the solver before converging.


    Attributes:
        default_options (dict): contains default step size options concerning consecutive_accept if not provided during creation.
        optimal_nstep (int): optimal number of interations that should be performed by the solver before converging.
    """

    # FIX: only works with newton solver, since it needs the number of iterations

    name = "solver convergence step size adaptation"
    factory_keyword: str = "convergence"
    default_options = {"optimal_nstep": 6}  # TODO: tweak that, maybe fix relative to max_iter

    def __init__(self, bounds: list[float, float], **kwargs):
        super().__init__(bounds)
        self.stepsize_options = getCustomOptionDictionary(
            kwargs.get("stepsize_options", dict()), self.default_options
        )
        self.optimal_nstep = self.stepsize_options["optimal_nstep"]

    def getStepSize(self, ds: float, sollist: list[SystemSolution], **kwargs) -> float:
        """Returns the step size to be used for the prediction step of the analysis.

        Args:
            ds (float): Current step size.
            sollist (list[SystemSolution]): list of SystemSolution returned during the analysis.

        Returns:
            ds (float): Updated step size.
        """
        # Refine also if solution is not accepted
        if (not sollist[-1].flag_accepted) and (ds > self.ds_min):
            ds /= 4
        else:
            ds *= self.optimal_nstep/sollist[-1].iter
        return ds
