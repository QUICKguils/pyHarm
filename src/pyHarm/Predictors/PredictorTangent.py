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

    # WARN: change: k_impose should come from get_last_solution now
    def compute_tangent(self, sol: SystemSolution):
        """Compute the tangent of the solution curve at the given solution point.

        Args:
            sol (SystemSolution):
                Previous computed solution. ds should already been have computed.

        Writes:
            sol.dir:
                Direction of the tangent.
        """
        # See Allgower, p.29 : Jacobian tangent from its QR decomposition
        sol.get_jacobian("full")  # this makes last_sol.J_f available
        Q, R = linalg.qr(np.transpose(sol.J_f[:-1, :]))
        sign = -np.sign(np.prod(np.diag(R)))  # sign = sign(det(R)*det(Q)), with det(Q) = 1
        sol.dir = sign * Q[:, -1]

    def predict(self, sol: SystemSolution):
        """Predict the next solution, through an Euler prediction.

        Args:
            sol (SystemSolution):
                Previous computed solution.
                ds, sign_ds and dir should already been have computed.

        Writes:
            sol.x_pred:
                Prediction of the next solution.
        """

        # Euler prediction: step along the tangent to the solution branch
        sol.x_pred = sol.x + sol.sign_ds * sol.ds * sol.dir
