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
This module contains the basic NodeToNodeElement class being a abstract class derived from the ABCElement class 
It mostly implements a part of the required abstract methods : the methods that generates the indices from the input datas.
"""

import pandas as pd
from pyHarm.Elements.ABCElement import ABCElement
import numpy as np
import copy


def ConstructorPslavemaster(nsub, subs, nodes, sub_expl_dofs):
    """
    Function that creates the selection matrices for the slave dofs and the master dofs.

    Args:
        nsub (int): number of substructures that are connected with the kinematic condition.
        subs (list[str]): list containing the name of the substructures connected with the kinematic condition.
        nodes (list[list[int]]): list of nodes list that are connected with the kinematic condition.
        sub_expl_dofs (pd.DataFrame): explicit dofs DataFrame.

    Returns:
        np.ndarray: selection matrix for the slave dofs.
        np.ndarray: selection matrix for the master dofs.
    """
    sub_expl_dofs = sub_expl_dofs.reset_index(drop=True)
    led = len(sub_expl_dofs)
    if nsub == 1 :
        Pslave = np.zeros((led,led))
        Pslave[np.arange(led),sub_expl_dofs[sub_expl_dofs["sub"]==subs[0]].index] = 1
        Pmaster = np.zeros((led,led))
    elif nsub == 2 :
        Pslave = np.zeros((led//2,led))
        Pslave[np.arange(led//2),sub_expl_dofs[((sub_expl_dofs["sub"]==subs[0]) & 
                                                (sub_expl_dofs["node_num"]==nodes[0][0]))].index] = 1
        Pmaster = np.zeros((led//2,led))
        Pmaster[np.arange(led//2),sub_expl_dofs[((sub_expl_dofs["sub"]==subs[1]) & 
                                                (sub_expl_dofs["node_num"]==nodes[1][0]))].index] = 1
    return Pslave,Pmaster  

class NodeToNodeElement(ABCElement) :
    """
    Abstract ABCElement subclass that implements some of the methods in order to help building a node to node connector.
    
    Args:
        nh (int): number of harmonics.
        nti (int): number of time steps.
        name (str): name given to the kinematic condition.
        data (dict): dictionary containing all the definition information of the kinematic condition.
        CS (CoordinateSystem): local or global coordinate system the kinematic condition is defined on.

    Attributes:
        indices (np.ndarray): index of the dofs that the kinematic conditions needs.
        Pdir (np.ndarray): a slice of first dimension is a transformation matrix to a direction in local coordinate system.
        Pslave (np.ndarray): selection array that selects the slave dofs of the kinematic condition.
        Pmaster (np.ndarray): selection array that selects the master dofs of the kinematic condition.
        subs (list[str]): list containing the name of the substructures tht are involved.
        nbSub (int): number of substructure involved.
        nodes (list[list]): list of list of nodes the kinematic conditions act on.
        nbdofi (int): number of nodes involved per substructure.
    """ 
        

    def __init_data__(self, name, data, CS):
        """"
        Method that interprets and deals with the input dictionary by creating some of the essential attributes.

        Attributes:
            subs (list[str]): list containing the name of the substructures tht are involved.
            nbSub (int): number of substructure involved.
            nodes (list[list]): list of list of nodes the kinematic conditions act on.
            nbdofi (int): number of nodes involved per substructure.
        
        """
        self.indices = []
        self.name = name
        self.CS = CS
        self.subs = list(data["connect"].keys())
        self.nbSub = len(self.subs)
        if "INTERNAL" in self.subs : 
            self.subs[1] = self.subs[0]
        self.nodes = list(data["connect"].values())
        self.nbdofi = len(data["connect"][list(data["connect"].keys())[0]])
        self.data = data

    def __str__(self):
        subnames = self.subs
        sub1 = subnames[0]
        sub1ddls = self.dofssub[0]
        if self.nbSub == 1 :
            sub2 = "ground"
            sub2ddls = ""
        else : 
            sub2 = subnames[1]
            sub2ddls = self.dofssub[1]
        return "Element of type {} that links :\n - {} dofs {} \n to\n - {} dofs {}".format(self.__class__.__name__, sub1, sub1ddls,\
                                                                sub2, sub2ddls,)


    def generateIndices(self, expl_dofs: pd.DataFrame):
        """From the explicit dof DataFrame, generates the index of dofs concerned by the connector.
        
        Args:
            expl_dofs (pd.DataFrame): explicit dof DataFrame from the studied system.

        Attributes:
            indices (np.ndarray): index of the dofs that the connector needs.
            Pdir (np.ndarray): a slice of first dimension is a transformation matrix to a direction in local coordinate system.
            Pslave (np.ndarray): selection array that selects the slave dofs of the connector.
            Pmaster (np.ndarray): selection array that selects the master dofs of the connector.
        """
        # discriminate subs from the explicit dof list :
        cs = pd.Series([False]*len(expl_dofs))
        dof_available = np.unique(np.array(expl_dofs["dof_num"]))
        for indice_sub,sub in enumerate(self.subs) :
            sub_discrim = (expl_dofs["sub"] == sub)
            modal_dof_discrim = (expl_dofs["dof_num"]!=-1)
            cn = pd.Series([False]*len(expl_dofs))
            for node in self.nodes[indice_sub] :
                cn += (expl_dofs["sub"] == sub) * (expl_dofs["node_num"] == node)
            cs += cn
            dof_available = np.intersect1d(dof_available,np.unique(np.array(expl_dofs[(sub_discrim * modal_dof_discrim)]["dof_num"])))
        sub_expl_list = expl_dofs[cs]
        cdd = pd.Series([False]*len(sub_expl_list),index=sub_expl_list.index)
        for dof in dof_available : 
            cdd += (sub_expl_list["dof_num"] == dof)
        self.indices = np.sort(sub_expl_list[cdd].index)
        self.component = dof_available
        self._generateMatrices(sub_expl_list[cdd])

    def _generateMatrices(self,sub_expl_dofs): 
        self.Pslave,self.Pmaster = ConstructorPslavemaster(self.nbSub, self.subs, self.nodes, sub_expl_dofs)
        self.Pdir = self.CS.getTM(self.nh,self.component)[np.array(self.data["dirs"]),:,:]

    def _evalJaco_DF(self, xg, om, step):
        """Computes the jacobian using finite difference method.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.
            step (float): step size for the finite difference method.

        Returns:
            dJdx (np.ndarray): jacobian with respect to displacement vector.
            dJdom (np.ndarray): jacobian with respect to angular frequency.
        """
        R_init = self.evalResidual(xg, om)
        dJdx = np.zeros((len(self.indices), len(self.indices)))
        dJdom = np.zeros((len(self.indices),1))
        for kk,idid in enumerate(self.indices) : 
            x_m = copy.copy(xg)
            x_m[idid] += step
            R_idid = self.evalResidual(x_m, om)
            dJdx[:,kk] = (R_idid - R_init) / step
        R_om = self.evalResidual(xg, om+step)
        dJdom[:,0] = (R_om - R_init) / step
        self.J = dJdx
        self.dJdom = dJdom
        return dJdx, dJdom
    