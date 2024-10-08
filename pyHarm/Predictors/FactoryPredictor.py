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

"""
This module contains the factory of the predictor objects. 

Attributes:
    Predictor_dico (dict): Dictionary that contains ABCPredictor objects as values and their factory_keyword attribute as their key.
"""
from pyHarm.Predictors.PredictorTangent import PredictorTangent
from pyHarm.Predictors.PredictorSecant import PredictorSecant
from pyHarm.Predictors.PredictorPreviousSolution import PredictorPreviousSolution
from pyHarm.Predictors.ABCPredictor import ABCPredictor

Predictor_dico = {PredictorTangent.factory_keyword:                             PredictorTangent,
                  PredictorSecant.factory_keyword:                              PredictorSecant,
                  PredictorPreviousSolution.factory_keyword:                            PredictorPreviousSolution}
"""dict: Dictionary that contains ABCPredictor objects as values and their factory_keyword attribute as their key."""


def generatePredictor(name_predictor, sign_ds, predictor_options) -> ABCPredictor:
    """
    Factory function that creates a ABCPredictor object.

    Args:
        name_predictor (str): Type of the ABCPredictor object that is to be instantiated.
        sign_ds (float): either -1 or 1, gives the initial direction of prediction.
        predictor_options (dict): dictionary containing supplementary options for the predictor to be instantiated

    Returns:
        ABCPredictor: Instance of the required ABCPredictor class.
    """
    E = Predictor_dico[name_predictor](sign_ds,**predictor_options)
    return E