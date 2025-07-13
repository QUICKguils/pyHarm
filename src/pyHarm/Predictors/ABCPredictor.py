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

import numpy as np

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Solver import SystemSolution


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

    default_options = {
        "bifurcation_detect": False,  # XXX: remove that when Bifurcator implemented
        "norm": "norm1",
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

    @property
    @abc.abstractmethod
    def factory_keyword(self) -> str:
        """str: Concrete class name used by the factory to instantiate it."""
        pass

    @abc.abstractmethod
    def predict(self, SolList: list[SystemSolution], ds: float) -> SystemSolution:
        """Predict the next starting point.

        Args:
            SolList (list[SystemSolution]): list of SystemSolution already solved during the analysis.
            ds (float): Step size for the prediction.

        Returns:
            SystemSolution: last accepted solution of SolList, with computed prediction written in it.
        """
        pass

    def normalize(self, dir: float) -> float | None:
        """Normalises the direction according to the choice of norm given in the class attributes.

        Args:
            dir (np.ndarray): prediction direction.

        Returns:
            np.ndarray: normalized prediction direction.
        """
        if self.predictor_options["norm"] == "norm1":
            return dir / np.linalg.norm(dir)
        if self.predictor_options["norm"] == "om":
            return dir / dir[-1]
        print('Wrong normalisation option, please choose between "norm1" and "om"')
        return None
