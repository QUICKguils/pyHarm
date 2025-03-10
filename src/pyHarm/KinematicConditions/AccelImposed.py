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

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.KinematicConditions.GODisplacement import GODisplacement


class AccelImposed(GODisplacement):
    """Kinematic condition that imposes an acceleration on specific dof.

    Attributes:
        amp (float): amplitude to impose.
    """

    factory_keyword: str = "AccelImposed"
    """str: Concrete class name used by the factory to instantiate it."""

    default = {"phi": 0.0, "ho": 1}
    """dict: dictionary containing the default parameters of the kinematic condition"""

    def __post_init__(self):
        self.data = getCustomOptionDictionary(self.data, self.default)
        self.amp = self.data["amp"]
        self.ho = self.data["ho"]
        self.dto = 2.0
        if "phi" not in self.data.keys():
            self.phi = 0.0
        else:
            self.phi = float(self.data["phi"])
        self.loadvec = self._loadingdofs()
