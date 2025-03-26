import numpy as np

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Solver import FirstSolution, SystemSolution
from pyHarm.StepSizeRules.ABCStepSizeRule import ABCStepSizeRule


class StepSizeHybrid(ABCStepSizeRule):
    """Step size is scaled based on how mush the nonlinear solver struggled to converge.

    ds = M_opt/M * ds_prev
    where M is the number of iterations performed by the solver before converging.


    Attributes:
        default_options (dict): contains default step size options concerning consecutive_accept if not provided during creation.
        consecutive_accept (int): number of consecutive accept before increasing step size.
        optimal_nstep (int): optimal number of interations that should be performed by the solver before converging.
    """

    name = "hybrid convergence-acceptance stepsizer"
    factory_keyword: str = "hybrid"
    default_options = {
        "consecutive_accept": 5,
        "optimal_nstep": 3,  # TODO: tweak that properly, maybe fix relative to max_iter
    }

    def __init__(self, bounds: list[float, float], **kwargs):
        super().__init__(bounds)
        self.stepsize_options = getCustomOptionDictionary(
            kwargs.get("stepsize_options", dict()), self.default_options
        )
        self.consecutive_accept = self.stepsize_options["consecutive_accept"]
        self.optimal_nstep = self.stepsize_options["optimal_nstep"]

    def getStepSize(self, ds: float, sollist: list[SystemSolution], **kwargs) -> float:
        """Returns the step size to be used for the prediction step of the analysis.

        Args:
            ds (float): Current step size.
            sollist (list[SystemSolution]): list of SystemSolution returned during the analysis.

        Returns:
            ds (float): Updated step size.
        """
        sol = sollist[-1]

        if (not sol.flag_accepted) and (ds > self.ds_min):
            ds /= 6
        try:
            acc = np.array([sol.flag_accepted for sol in sollist[-self.consecutive_accept : :]])
            if np.sum(acc) == self.consecutive_accept and ds < self.ds_max:
                ds *= 1.5
        except Exception:
            pass
        if ds < self.ds_min or ds > self.ds_max:
            ds = self.ProjectInBounds(ds)  # shouldn't be necessary

        # FIX: see why niter is 0 for some solutions
        # must have a link with what the code does after rejcting a solution
        elif not isinstance(sol, FirstSolution) and sol.niter > 0:
            ds *= self.optimal_nstep / sol.niter
        return ds
