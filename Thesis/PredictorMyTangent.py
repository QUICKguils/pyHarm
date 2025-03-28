import numpy as np
from scipy import linalg

from pyHarm.Predictors.ABCPredictor import ABCPredictor
from pyHarm.Solver import FirstSolution, SystemSolution


class PredictorMyTangent(ABCPredictor):
    """Implement a tangent-type predictor."""

    predictor_name = "Tangent Predictor"
    factory_keyword: str = "mytangent"
    """str: Concrete class name used by the factory to instantiate it."""

    def predict(
        self, SolList: list[SystemSolution], ds: float, k_imposed=None
    ) -> tuple[np.ndarray, SystemSolution]:
        """Predicts the next starting point using the tangent.

        Args:
            SolList (list[SystemSolution]): list of SystemSolution already solved during the analysis.
            ds (float): step size for the prediction.
            k_imposed (None | int): if not None, uses the k_imposed as the index of the last
              solution pointer.

        Returns:
            np.ndarray: next predicted starting point.
            SystemSolution: last accepted point in the list of solutions.
        """
        last_sol = self.get_last_point(SolList, k_imposed)
        prev_sol = last_sol.precedent_solution
        last_sol.get_jacobian("full")  # this makes lstpt.J_f available
        solx_len = last_sol.J_f[:-1, :-1].shape[0]

        if self.predictor_options["bifurcation_detect"]:
            self.bifurcation_detect(last_sol)

        # # QR decomposition of transpose of Jacobian, without correction equation
        # Q, R = linalg.qr(np.transpose(lstpt.J_f))
        # #
        # # Evaluate the direction of zero gradient for the redsidual
        # dir = np.sign(R.diagonal()).prod() * Q[:, -1]

        if isinstance(last_sol, FirstSolution):
            prev_dir = np.vstack((np.zeros((solx_len, 1)), last_sol.sign_ds))
        else:
            prev_dir = prev_sol.dir
        A = np.vstack((last_sol.J_f[:-1, :], prev_dir.T))
        b = np.vstack((np.zeros((solx_len, 1)), 1))
        dir = linalg.solve(A, b).ravel()
        dir = self.normalize(dir)

        # Euler prediction: step along the tangent to the solution branch
        xpred = last_sol.x + dir * ds

        # # Debug prints
        # print(f"{ds=}, {self.sign_ds=}")
        # print(f"{dir=}")
        # print(f"{xpred=}")

        # write in the previous computed solution
        last_sol.dir = dir
        last_sol.x_pred = xpred

        return xpred, last_sol
