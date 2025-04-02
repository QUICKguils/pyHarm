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

import abc
from enum import Enum

import numpy as np
import scipy.linalg as spl

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Solver import FirstSolution, SystemSolution


class BifurcationType(Enum):
    FOLD = "fold bifurcation"
    BRANCHING = "branching point bifurcation"
    NEIMARK_SACKER = "Neimark-Sacker bifurcation"


class ABCPredictor(abc.ABC):
    """Abstract class for the predictor.

    Any added predictor shall be constructed from this class.

    Args:
        dir (np.ndarray):
            Normalized tangent vector at the last solved point.

    Attributes:
        flag_print (bool):
          information are printed during the analysis if True.
        predictor_options (dict):
          dictionary containing the kwargs and completed using the default options
          if the keywords are missing.
    """

    @property
    @abc.abstractmethod
    def factory_keyword(self) -> str:
        """str: Concrete class name used by the factory to instantiate it."""
        pass

    default_options = {
        "norm": "norm1",
        "bifurcation_detect": True,
        "blind_spot": 30,
        "verbose": True,
    }
    """
    dict:
      Set of default parameters for the system class if not given in the input argument.

    It contains a normalisation parameter using the keyword 'norm' that can be set to either 'norm1'
    (default) if the direction is normed to 1 or 'om' if the direction is normed to 1 only for the
    angular frequency.
    It contains a bifurcation detection using the 'bifurcation_detect' keyword that can be set to
    True (default) if detection is needed.
    It contains a 'blind_angle' keyword that sets a blind angle to avoid the predictor to search in
    the inwards direction.
    It contains a 'verbose' keyword that can be set to True (default) if information about detection
    of bifurcations is to be displayed during solving.
    """

    def __init__(self, sign_ds, **kwargs):
        self.sign_ds = sign_ds
        self.predictor_options = getCustomOptionDictionary(kwargs, self.default_options)
        self.flag_print = self.predictor_options["verbose"]

    @abc.abstractmethod
    def predict(
        self, SolList: list[SystemSolution], ds: float
    ) -> tuple[np.ndarray, SystemSolution]:
        """Predict the next starting point.

        Args:
            SolList (list[SystemSolution]): list of SystemSolution already solved during the analysis.
            ds (float): Step size for the prediction.

        Returns:
            np.ndarray: Next predicted starting point.
            SystemSolution: last accepted point in the list of solutions.
        """
        pass

    def bifurcation_detect(self, last_sol: SystemSolution):
        """
        Makes a bifurcation detection analysis computing determinant of jacobian matrix
        and analyzing change of sign.

        Args:
            lstpt (SystemSolution): Previously accepted point in direct link
              with the actual solved point.

        Attributes:
            sign_ds (float): Attribute is modified if a fold bifurcation is detected.
            bifurcation_type (BifurcationType): Attribute is assigned if a bifurcation is detected.
        """
        J = last_sol.get_jacobian("full")
        prev_sol = last_sol.precedent_solution
        # XXX: heavy to compute dets. Consider bordering techniques
        det_J = spl.det(J)
        det_J_x = spl.det(J[:-1, :-1])
        last_sol.det_J_f = det_J
        last_sol.det_J_x = det_J_x
        if isinstance(last_sol, FirstSolution):
            det_J_x_prev = det_J_x
            det_J_f_prev = det_J
        else:
            det_J_x_prev = prev_sol.det_J_x
            det_J_f_prev = prev_sol.det_J_f

        if np.sign(det_J_x) != np.sign(det_J_x_prev):
            last_sol.flag_bifurcation = True
            if np.sign(det_J) == np.sign(det_J_f_prev) or np.sign(det_J) != np.sign(det_J_x):
                last_sol.bifurcation_type = BifurcationType.FOLD
            else:
                last_sol.bifurcation_type = BifurcationType.BRANCHING
                self.sign_ds *= -1
            if self.flag_print:
                print(f"Warning: a {last_sol.bifurcation_type.value} was detected")
        elif (
            not isinstance(last_sol, FirstSolution)
            and np.dot(prev_sol.dir, last_sol.dir) < -np.cos(np.deg2rad(self.predictor_options["blind_spot"]/2))
        ):
            last_sol.flag_bifurcation = True
            self.sign_ds *= -1
            if self.flag_print:
                print("A potential higher order bifurcation is detected")

    @staticmethod
    def get_last_point(sollist: list[SystemSolution], k_imposed=None) -> SystemSolution:
        """Gets the last accepted solution in direct link with the studied point.

        Args:
            sollist (list[SystemSolution]): list of SystemSolution already solved during the analysis.
            k_imposed (None|int): if not None then the provided index is used as the last accepted point.

        Returns:
            SystemSolution: last accepted point.
        """
        if k_imposed is None:
            lstpt = sollist[-1]
            k = 0
            while not lstpt.flag_accepted:
                lstpt = sollist[-1 - k]
                k += 1
        else:
            lstpt = sollist[k_imposed]
        return lstpt

    def normalize(self, dir: float) -> float:
        """Normalises the direction according to the choice of norm given in the class attributes.

        Args:
            dir (np.ndarray): prediction direction.

        Returns:
            np.ndarray: normalized prediction direction.
        """
        if self.predictor_options["norm"] == "norm1":
            dir = dir / np.linalg.norm(dir)
        elif self.predictor_options["norm"] == "om":
            dir = dir / dir[-1]
        else:
            print('Wrong normalisation option, please choose between "norm1" and "om"')
        return dir
