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

from pyHarm.Predictors.ABCPredictor import ABCPredictor
from pyHarm.Solver import SystemSolution, get_last_solution


class PredictorPreviousSolution(ABCPredictor):
    """Defines the previous solution type of predictor.

    Uses the previous solution for the displacement and puts the step purely on
    the angular frequency.
    """

    factory_keyword: str = "previous"
    """str: Concrete class name used by the factory to instantiate it."""

    predictor_name = "Previous Solution Predictor"

    def predict(self, solList: list[SystemSolution], ds: float, k_imposed=None) -> SystemSolution:
        """Predicts the next starting point by taking the previous solution and making a step in angular frequency.

        Args:
            SolList (list[SystemSolution]): list of SystemSolution already solved during the analysis.
            ds (float): step size for the prediction.
            k_imposed (None | int): if not None, uses the k_imposed as the index of the last solution pointer.

        Returns:
            SystemSolution: last accepted solution of SolList, with computed prediction written in it.
        """
        # Get pointer to solution, Jacobian in full mode, and bifurcation detection
        last_sol = get_last_solution(solList, k_imposed)
        last_sol.get_jacobian("full")
        self.bifurcation_detect(last_sol)
        dir = np.zeros(last_sol.x.shape)
        dir[-1] = 1
        xpred = last_sol.x + dir * ds * self.sign_ds

        # write some stuff in the solution
        last_sol.dir = dir
        last_sol.x_pred = xpred

        return last_sol
