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

from pyHarm.Predictors.ABCPredictor import ABCPredictor
from pyHarm.Solver import FirstSolution, SystemSolution
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
import numpy as np
import scipy.linalg as spl

class PredictorTangent(ABCPredictor):
    """Define the tangent type of predictor. Using the Jacobian at solution point, a tangent to R(x)=0 solution is drawn and used as a prediction direction.

    The tangent is computed using a QR decomposition of the Jacobian at the solution point.
    """
    predictor_name = "Tangent Predictor"
    factory_keyword : str = "tangent"
    """str: keyword that is used to call the creation of this class in the system factory."""

    def predict(self, sollist:list[SystemSolution], ds:float, k_imposed=None) -> tuple[np.ndarray,SystemSolution,float]:
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
        lstpt = self.getPointerToSolution(sollist,k_imposed) # get pointer
        lstpt.getJacobian("full") # get J_f
        self.bifurcation_detect(lstpt) # get pointer
        ### Get the tangent
        # get QR decomposition of transpose of Jacobian without correction equation
        lstpt.J_x_T_qr = spl.qr(np.transpose(lstpt.J_f[:-1,:]))
        lstpt.flag_J_x_T_qr = True
        # det_Q,det_R = spl.det(lstpt.J_x_T_qr[0]),spl.det(lstpt.J_x_T_qr[1][:-1,:])
        dir = np.sign(lstpt.J_x_T_qr[0][-1,-1]) * lstpt.J_x_T_qr[0][:,-1] # no normalisation needed already normalized to norm=1
        dir = self.norm_dir(dir) * np.sign(dir[-1])
        xpred = lstpt.x + dir * ds * self.sign_ds
        ## write some stuff in the solution
        lstpt.dir = dir
        lstpt.x_pred = xpred
        # lstpt.sign_ds = self.sign_ds
        return xpred,lstpt,self.sign_ds