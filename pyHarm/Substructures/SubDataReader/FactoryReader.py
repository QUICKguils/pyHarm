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
This module is the factory of the Substructure Readers.

Attributes:
    SubstructureReaderDictionary (dict[str, ABCReader]): Dictionary of available Substructure reader subclasses as values and their factory_keyword as key.
"""
from pyHarm.Substructures.SubDataReader.ABCReader import ABCReader
from pyHarm.Substructures.SubDataReader.GenericReader import GenericReader

SubstructureReaderDictionary = \
{
    
    GenericReader.factory_keyword:            GenericReader, # For now the dictionary is useless, but could be usefull if more type of Substructures are needed

}
"""dict[str, ABCReader]: Dictionary of available Substructure readers subclasses as values and their factory_keyword as key."""


def generate_subreader(data:dict) -> ABCReader:
    """
    Function responsible for the instantiation of ABCReader objects.

    Args:
        data (dict): input dictionary

    Returns:
        ABCReader: Instance of a subclass of ABCReader class.
    """
    type_of_reader = data['reader']
    if type_of_reader not in SubstructureReaderDictionary.keys():
        raise ValueError('The required substructure reader does not exist in the factory')
    return SubstructureReaderDictionary[type_of_reader]()