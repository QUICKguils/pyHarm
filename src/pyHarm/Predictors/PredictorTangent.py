# Copyright 2024 SAFRAN SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy import linalg

from pyHarm.Predictors.ABCPredictor import ABCPredictor
from pyHarm.Solver import FirstSolution, SystemSolution


class PredictorTangent(ABCPredictor):
    """Define the tangent type of predictor.

    Using the Jacobian at solution point, a tangent to R(x)=0 solution is drawn and used as a
    prediction direction.

    The tangent is computed using a QR decomposition of the Jacobian at the solution point.
    """

    predictor_name = "Tangent Predictor"
    factory_keyword: str = "tangent"
    """str: Concrete class name used by the factory to instantiate it."""

    def predict(
        self, sollist: list[SystemSolution], ds: float, k_imposed=None
    ) -> tuple[np.ndarray, SystemSolution, float]:
        """Predicts the next starting point using the tangent.

        Args:
            sollist (list[SystemSolution]): list of SystemSolution already solved during the
              analysis.
            ds (float): step size for the prediction.
            k_imposed (None | int): if not None, uses the k_imposed as the index of the last
              solution pointer.

        Returns:
            np.ndarray: next predicted starting point.
            SystemSolution: last accepted point in the list of solutions.
            float: sign of the prediction used (-1 | 1).
            float: direction of the prediction used (-1 | 1).
        """
        lstpt = self.getPointerToSolution(sollist, k_imposed)
        lstpt.getJacobian("full")  # this makes lstpt.J_f available

        if self.predictor_options["bifurcation_detect"]:
            self.bifurcation_detect(lstpt)

        # QR decomposition of transpose of Jacobian, without correction equation
        lstpt.J_x_T_qr = linalg.qr(np.transpose(lstpt.J_f[:-1, :]))

        # Evaluate the direction of zero gradient for the redsidual
        dir = np.sign(lstpt.J_x_T_qr[0][-1, -1]) * lstpt.J_x_T_qr[0][:, -1]
        dir = self.norm_dir(dir) * np.sign(dir[-1])  # keep an omega-positive direction

        # # NOTE: this paragraph is experimental
        # # QR decomposition of transpose of Jacobian, without correction equation
        # Q, R = linalg.qr(np.transpose(lstpt.J_f))
        # #
        # # Evaluate the direction of zero gradient for the redsidual
        # dir = np.sign(R.diagonal()).prod() * Q[:, -1]

        # # NOTE: this paragraph is experimental
        # solx_len = lstpt.J_f[:-1, :-1].shape[0]
        # if isinstance(lstpt, FirstSolution):
        #     lstpt.dir = np.vstack((np.zeros((solx_len, 1)), 1))
        # A = np.vstack((lstpt.J_f[:-1, :], lstpt.dir.T))
        # b = np.vstack((np.zeros((solx_len, 1)), 1))
        # dir = linalg.solve(A, b).reshape(-1)

        # Euler prediction: step along the tangent to the solution branch
        xpred = lstpt.x + dir * ds * self.sign_ds

        # # Debug prints
        # print(f"{ds=}, {self.sign_ds=}")
        # print(f"{dir=}")
        # print(f"{xpred=}")

        # 
        lstpt.dir = dir
        lstpt.x_pred = xpred

        return xpred, lstpt, self.sign_ds
