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
This module is responsible for the definition of the Abstract based class ABCSubstructure defining the main interfaces charateristics of a substructure in pyHarm.
The substructure shall be responsible for creating dofs through the generation of an explicit dof list describing which dofs are being created in the system.
"""

from abc import ABC, abstractmethod
from pyHarm.Substructures.SubDataReader.FactoryReader import generate_subreader
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
import pandas as pd
class ABCSubstructure(ABC):
    """
    This class defines a substructure. Its main responsability is to create dofs and put them in an explicit pandas DataFrame. 

    Args:
        nh (int): number of harmonics.
        nti (int): number of time steps.
        name (str): name given to the kinematic condition.
        data (dict): dictionary containing all the definition information of the substructure.

    Attributes:
        nh (int): number of harmonics.
        name (int): name of the substructure.
        nnodes (int): number of nodes.
        nmodes (int): number of modes.
        ndofs (int): number of dofs per node.   
        edf (pd.DataFrame): DataFrame describing the created dofs.   
        connectors (dict): dictionary containing the added Elements to the system.   
        kinematics (dict): dictionary containing the added Kinematic Conditions to the system.   
    """
    flag_substructure = True

    default_data = {'type':'linear','reader':'generic'}

    def __init__(self, nh:int, name:str, data:dict):
        self.nh = nh
        self.name = name

        data = getCustomOptionDictionary(data,self.default_data)
        self.substructure_reader = generate_subreader(data)
        data = self.substructure_reader.data_complete(data)

        self.nnodes = data['nnodes']
        self.nmodes = data['nmodes']
        self.ndofs  = data['ndofs']
        self.dofs_matching = data['matching']
        self.total_dofs = self.nmodes  +  self.nnodes * self.ndofs
        self.edf = self._get_explicit_df()

        self.connectors = self._add_connectors(data)
        self.kinematics = self._add_kinematics(data)

    def __repr__(self) -> str:
        return "{}[{}]".format(self.name, self.__class__.__name__)

    @property
    @abstractmethod
    def factory_keyword(self) -> str:
        """"
        Property defining the factory_keyword to be used for instantiation of daughter class.

        Returns:
            str: factory_keyword
        """
        ...

    @abstractmethod
    def _add_connectors(self) -> dict :
        """"
        Method that adds connectors depending on the type of substructure.
        """
        ...

    @abstractmethod
    def _add_kinematics(self) -> dict :
        """"
        Method that adds kinematic conditions depending on the type of substructure.
        """
        ...

    def _gen_sub_col(self):
        col = [self.name]*self.total_dofs*(2*self.nh+1)
        return col
    
    def _gen_harm_col(self):
        col = list()
        for h in range(0,(2*self.nh+1)):
            col += [(h+1)//2]*self.total_dofs
        return col
    
    def _gen_cs_col(self):
        col = ['c']*self.total_dofs
        for h in range(1,(2*self.nh+1)):
            c_o_s = 'c'
            if (h+1)%2 == 1:
                c_o_s='s'
            col += [c_o_s]*self.total_dofs
        return col
    
    def _gen_nodes_col(self):
        col = [[i]*self.ndofs for i in range(self.nnodes)]
        col += [[self.nnodes+i] for i in range(self.nmodes)]
        col = sum(col,[])
        col = col*(2*self.nh+1)
        return col
    
    def _gen_dofs_col(self):
        col = self.dofs_matching*self.nnodes
        col += [-1]*self.nmodes
        col = col*(2*self.nh+1)
        return col
    
    def _gen_PoM_col(self):
        col = ['p']*self.ndofs*self.nnodes
        col += ['m']*self.nmodes
        col = col*(2*self.nh+1)
        return col
    
    def _get_explicit_df(self):
        dict_df = {
            'sub':self._gen_sub_col(),
            'harm':self._gen_harm_col(),
            'cs':self._gen_cs_col(),
            'node_num':self._gen_nodes_col(),
            'dof_num':self._gen_dofs_col(),
            'PoM':self._gen_PoM_col(),
        }
        edf = pd.DataFrame(dict_df)
        return edf
    
