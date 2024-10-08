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
from pyHarm.Systems.ABCSystem import ABCSystem

class ABCAnalysis(abc.ABC): 
    """This is the abstract class ruling the solver class. The system is responsible of solving the system starting at a given starting point.
    """

    @property
    @abc.abstractmethod
    def factory_keyword(self):
        """
        Class name for factory call during instantiation
        """
        ...

    def __init__(self, inputData:dict, System:ABCSystem, ndofs:int,**kwargs):
        pass

    @abc.abstractmethod
    def initialise(self,**kwargs):
        """
        Initialise step of the analysis.
        """
        pass
    
    @abc.abstractmethod
    def Solve(self,**kwargs):
        """
        Solving step of the analysis.
        """
        pass
    
    @abc.abstractmethod
    def makeStep(self,**kwargs):
        """
        Make a whole step of the analysis.
        """
        pass
    
