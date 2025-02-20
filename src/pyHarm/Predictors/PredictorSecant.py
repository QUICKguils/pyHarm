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

from pyHarm.Predictors.PredictorTangent import PredictorTangent
from pyHarm.Solver import FirstSolution, SystemSolution
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
import numpy as np
import scipy.linalg as spl

class PredictorSecant(PredictorTangent):
    """Define the Secant predictor. From the last two solution points, generates the adequate direction. When only one solution point is available, makes use of the tangent predictor.
    """
    predictor_name = "Secant Predictor"
    """str: keyword that is used to call the creation of this class in the system factory."""
    factory_keyword : str = "secant"

    def predict_usingtan(self, sollist:list, ds:float, k_imposed=None) -> tuple[np.ndarray,SystemSolution,float]:
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
        return super().predict(sollist,ds,k_imposed=None)

    def predict(self, sollist:list, ds:float, k_imposed=None) -> tuple[np.ndarray,SystemSolution,float]:
        """Predicts the next starting point using secant prediction.

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
        lstpt = self.getPointerToSolution(sollist,k_imposed) # get pointer
        lstpt.getJacobian("full") # get J_f
        self.bifurcation_detect(lstpt) # get pointer
        if isinstance(lstpt,FirstSolution) : 
            xpred,lstpt,self.sign_ds = self.predict_usingtan(sollist,ds,k_imposed=None)
        else : 
            dir = (lstpt.x - lstpt.precedent_solution.x)/np.linalg.norm(lstpt.x - lstpt.precedent_solution.x)
            dir = self.norm_dir(dir) * np.sign(dir[-1])
            xpred = lstpt.x + dir * ds * self.sign_ds
            ## write some stuff in the solution
            lstpt.dir = dir
            lstpt.x_pred = xpred
        return xpred,lstpt,self.sign_ds