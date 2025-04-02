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
from pyHarm.Solver import SystemSolution


class PredictorTangent(ABCPredictor):
    """Implement a tangent-type predictor."""

    predictor_name = "Tangent Predictor"
    factory_keyword: str = "tangent"
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

        # See Allgower, p.29 : Jacobian tangent from its QR decomposition
        last_sol.get_jacobian("full")  # this makes last_sol.J_f available
        Q, R = linalg.qr(np.transpose(last_sol.J_f[:-1,:]))
        sign = - np.sign(np.prod(np.diag(R)))  # sign = sign(det(R)*det(Q)), with det(Q) = 1
        last_sol.dir = sign * Q[:, -1]

        if self.predictor_options["bifurcation_detect"]:
            self.bifurcation_detect(last_sol)

        # Euler prediction: step along the tangent to the solution branch
        last_sol.x_pred = last_sol.x + self.sign_ds * ds * last_sol.dir

        return last_sol
