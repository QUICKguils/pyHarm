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

from pyHarm.DynamicOperator import compute_DFT, nabla
from pyHarm.CoordinateSystem import CoordinateSystem
import numpy as np
import pandas as pd
import abc
import copy

def ConstructorPslavemaster(nsub, subs, nodes, sub_expl_dofs):
    """
    Function that creates the selection matrices for the slave dofs and the master dofs.

    Args:
        nsub (int): number of substructures that are connected with the kinematic condition.
        subs (list[str]): list containing the name of the sbstructures connected with the kinematic condition.
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

class ABCKinematic(abc.ABC):
    """This is the abstract class ruling the kinematic conditions class. The kinematic conditions are responsible to impose kinematic on dofs of the system and transfer the residuals.
    
    Args:
        nh (int): number of harmonics.
        nti (int): number of time steps.
        name (str): name given to the kinematic condition.
        data (dict): dictionary containing all the definition information of the kinematic condition.
        CS (CoordinateSystem): local or global coordinate system the kinematic condition is defined on.

    Attributes:
        nh (int): number of harmonics.
        nti (int): number of time steps.
        D (dict[np.ndarray,np.ndarray]): Dynamic operators containing inverse discrete Fourier transform and discrete Fourier transform.
        nabla (np.ndarray): Derivation operator.
        indices (np.ndarray): index of the dofs that the kinematic conditions needs.
        Pdir (np.ndarray): a slice of first dimension is a transformation matrix to a direction in local coordinate system.
        Pslave (np.ndarray): selection array that selects the slave dofs of the kinematic condition.
        Pmaster (np.ndarray): selection array that selects the master dofs of the kinematic condition.
        subs (list[str]): list containing the name of the substructures that are involved.
        nbSub (int): number of substructure involved.
        nodes (list[list]): list of list of nodes the kinematic conditions act on.
        nbdofi (int): number of nodes involved per substructure.
    """
    
    @property
    @abc.abstractmethod
    def factory_keyword(self):
        ...
        
    def __init__(self, nh:int, nti:int, name:str, data:dict, CS:CoordinateSystem):
        # flags #
        self.__init_flags__()
        # real parameters #
        self.__init_harmonic_operators__(nh,nti)
        self.__init_data__(name,data,CS)
        self.__post_init__()
        self.__flag_update__()

    def __init_flags__(self,):
        pass

    def __init_harmonic_operators__(self,nh,nti):
        self.nh = nh
        self.nti = nti
        self.D = compute_DFT(nti, nh) # not necessary for non AFT elements
        self.nabla = nabla(nh)

    def __init_data__(self, name, data, CS):
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
        if not self.flag_substructure :
            subnames = self.subs
            sub1 = subnames[0]
            sub1ddls = self.dofssub[0]
            if self.nbSub == 1 :
                sub2 = "ground"
                sub2ddls = ""
            else : 
                sub2 = subnames[1]
                sub2ddls = self.dofssub[1]
            return "Kinematic Condition of type {} that links :\n - {} dofs {} \n to\n - {} dofs {}".format(self.__class__.__name__, sub1, sub1ddls,\
                                                                sub2, sub2ddls,)
        else :
            return "Kinematic Condition of type {}".format(self.__class__.__name__)
        
    def __repr__(self):
        return "{}[{}]".format(self.name, self.__class__.__name__)
    
    def __post_init__(self,*args):
        pass

    def __flag_update__(self,*args):
        pass

    def generateIndices(self,expl_dofs:pd.DataFrame) :
        """From the explicit dof DataFrame, generates the index of dofs concerned by the kinematic condition.
        
        Args:
            expl_dofs (pd.DataFrame): explicit dof DataFrame from the studied system.

        Attributes:
            indices (np.ndarray): index of the dofs that the kinematic conditions needs.
            Pdir (np.ndarray): a slice of first dimension is a transformation matrix to a direction in local coordinate system.
            Pslave (np.ndarray): selection array that selects the slave dofs of the kinematic condition.
            Pmaster (np.ndarray): selection array that selects the master dofs of the kinematic condition.
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
    
    @abc.abstractmethod
    def adim(self,lc,wc):
        """Using adim parameters, modifies the kinematic conditions accordingly.
        
        Args:
            lc (float): characteristic length.
            wc (float): characteristic angular frequency.
        """
        pass
    
    @abc.abstractmethod
    def complete_x(self, x, om):
        """Returns a vector x_add of same size of x that completes the vector of displacement x = x + x_add.
        
        Args:
            x (np.ndarray): displacement vector.
            om (float): angular frequency.
        """
        pass
    
    @abc.abstractmethod
    def complete_R(self, R, x):
        """Returns a vector R_add of same size of R that completes the vector of residual R = R + R_add
        
        Args:
            R (np.ndarray): residual vector.
            x (np.ndarray): displacement vector.
        """
        pass

    @abc.abstractmethod
    def complete_J(self, Jx, Jom, x):
        """Returns a vector Jx_add and Jom_add of same size of Jx and Jom that completes the Jacobian
        
        Args:
            Jx (np.ndarray): jacobian matrix with respect to displacement.
            Jom (np.ndarray): jacobian matrix with respect to angular frequency.
            x (np.ndarray): displacement vector.
        """
        pass