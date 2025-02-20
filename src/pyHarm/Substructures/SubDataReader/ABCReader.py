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

""""
This module contains the abstract class defining the main interfaces and behaviors expected from a SubDataReader objects.
The main purpose of such object is to modify and complete the input substructure data provided in the input file in order to adapt the syntax for Substructure object creation.
"""

from abc import ABC, abstractmethod

class ABCReader(ABC):
    """This is the abstract class ruling the reader class for substructure. The reader is responsible for reading, parsing and completing an input dictionary from a external file to comply with pyHarm.
    """

    @property
    @abstractmethod
    def factory_keyword(self):
        """
        Factory keyword to be used when instantiating a reader
        """
        ...

    @abstractmethod
    def data_complete(self,data:dict) -> dict:
        """
        Reads and completes the input dictionary from the file.
        """
        ...
