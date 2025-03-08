import numpy as np
import scipy.linalg as spl

from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
from pyHarm.Solver import SystemSolution


class BifurcatorHill(ABCBifurcator):
    """Branching predictions through the Hill's method.
    """
    # TODO: better docstring

    branching_name = "Hill's method"
    factory_keyword: str = "hill"
    """str: keyword that is used to call the creation of this class in the system factory."""

    # XXX: taken from PredictorTangent.py
    def detect_bifurcation(
        self, sollist: list[SystemSolution], ds: float, k_imposed=None
    ) -> tuple[np.ndarray, SystemSolution, float]:
        """Predicts the next starting point using the tangent.

        Args:
            sollist (list[SystemSolution]): list of SystemSolution already solved during the analysis.
            ds (float): step size for the prediction.
            k_imposed (None | int): if not None, uses the k_imposed as the index of the last solution pointer.

        Returns:
            np.ndarray: next predicted starting point.
            SystemSolution: last accepted point in the list of solutions.
            float: sign of the prediction used (-1 | 1)
        """
        ### Get pointer to solution, Jacobian in full mode, and bifurcation detection
        lstpt = self.getPointerToSolution(sollist, k_imposed)
        lstpt.getJacobian("full")  # get J_f
        if self.predictor_options["bifurcation_detect"]:
            self.bifurcation_detect(lstpt)
        ### Get the tangent
        # get QR decomposition of transpose of Jacobian without correction equation
        lstpt.J_x_T_qr = spl.qr(np.transpose(lstpt.J_f[:-1, :]))
        lstpt.flag_J_x_T_qr = True
        # det_Q,det_R = spl.det(lstpt.J_x_T_qr[0]),spl.det(lstpt.J_x_T_qr[1][:-1,:])
        dir = (
            np.sign(lstpt.J_x_T_qr[0][-1, -1]) * lstpt.J_x_T_qr[0][:, -1]
        )  # no normalisation needed already normalized to norm=1
        dir = self.norm_dir(dir) * np.sign(dir[-1])
        xpred = lstpt.x + dir * ds * self.sign_ds
        ## write some stuff in the solution
        lstpt.dir = dir
        lstpt.x_pred = xpred
        # lstpt.sign_ds = self.sign_ds
        return xpred, lstpt, self.sign_ds

    # XXX: taken from ABCPredictors.py
    def bifurcation_detect(self, lstpt: SystemSolution):
        """Makes a bifurcation detection analysis computing determinant of jacobian matrix and analysing change of sign.

        Args:
            lstpt (SystemSolution): previously accepted point in direct link with the actual solved point.

        Attributes:
            sign_ds (float): Attribute is modified if a turning point is detected.
        """
        Jaco = lstpt.get_jacobian("full")
        det_J_f = spl.det(Jaco)
        det_J_x = spl.det(Jaco[:-1, :-1])
        lstpt.det_J_f = det_J_f
        lstpt.det_J_x = det_J_x
        if isinstance(lstpt, FirstSolution):
            det_J_x_prec = det_J_x
            det_J_f_prec = det_J_f
        else:
            det_J_x_prec = lstpt.precedent_solution.det_J_x
            det_J_f_prec = lstpt.precedent_solution.det_J_f

        if np.sign(det_J_x) != np.sign(det_J_x_prec):
            lstpt.flag_bifurcation = True
            if np.sign(det_J_f) == np.sign(det_J_f_prec) or np.sign(det_J_f) != np.sign(det_J_x):
                self.sign_ds *= -1
                if self.flag_print:
                    print(
                        "Warning: a limit point (fold bifurcation) was detected"
                        " --> path direction is reversed"
                    )
            else:
                if self.flag_print:
                    print("Warning: a branching point was detected")

    def switch_branch(self):
        pass
