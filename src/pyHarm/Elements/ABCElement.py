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
"""
This file defines the main Abstract class for the Elements. 
Any ELement shall somehow inherit from this class.
"""

def ConstructorPslavemaster(nh,ndofi,nsub):
    ne = np.eye(2*nh+1)
    elem_select = np.eye(ndofi)
    elem_null = np.zeros((ndofi,ndofi))
    if nsub == 1 :
        Pslave = np.kron(ne,elem_select)
        Pmaster = np.kron(ne,elem_null)
    elif nsub == 2 :
        Pslave = np.kron(ne,np.concatenate([elem_select,elem_null],axis=1))
        Pmaster = np.kron(ne,np.concatenate([elem_null,elem_select],axis=1))
    return Pslave,Pmaster    


class ABCElement(abc.ABC):
    """This is the abstract class ruling the element class. 
    
    An element consists in an elementary contribution to the residual equations.
    
    Args:
        nh (int): number of harmonics.
        nti (int): number of time steps.
        name (str): name given to the kinematic condition.
        data (dict): dictionary containing all the definition information of the kinematic condition.
        CS (CoordinateSystem): local or global coordinate system the kinematic condition is defined on.

    Attributes:
        flag_nonlinear (bool): if True, the element is nonlinear.
        flag_AFT (bool): if True, the element requires an alternating frequency/time domain procedure for computing residuals.
        flag_extforcing (bool): if True, the element is an external forcing.
        flag_DLFT (bool): if True, the element uses the dynamic Lagrangian method for computing the residuals.
        flag_adim (bool): if True, the element is adimentioned.
        nh (int): number of harmonics.
        nti (int): number of time steps.
        D (dict[np.ndarray,np.ndarray]): Dynamic operators containing inverse discrete Fourier transform and discrete Fourier transform.
        nabla (np.ndarray): Derivation operator.
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
        self.flag_nonlinear = False
        self.flag_AFT = False
        self.flag_extforcing = False
        self.flag_DLFT = False
        self.flag_adim = False
        self.flag_elemtype = 0

    def __init_harmonic_operators__(self,nh,nti):
        self.nh = nh
        self.nti = nti
        self.D = compute_DFT(nti, nh) # not necessary for non AFT elements
        self.nabla = nabla(nh)
        
    def __repr__(self):
        return "{}[{}]".format(self.name, self.__class__.__name__)
    
    def __post_init__(self,*args):
        pass

    def __flag_update__(self,*args):
        pass

    @abc.abstractmethod
    def __init_data__(self, name, data, CS):
        ...

    @abc.abstractmethod
    def __str__(self):
        ...

    @abc.abstractmethod
    def generateIndices(self,expl_dofs:pd.DataFrame) :
        ...
    
    @abc.abstractmethod
    def adim(self,lc,wc):
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Args:
            lc (float): characteristic length value.
            wc (float): characteristic angular frequency value.
        """
        pass
    
    @abc.abstractmethod
    def evalResidual(self, x, om):
        """Computes the residual.
        
        Args:
            x (np.ndarray): full displacement vector.
            om (float): angular frequency value.
        """
        pass

    @abc.abstractmethod
    def evalJacobian(self, x, om):
        """Computes the jacobians.
        
        Args:
            x (np.ndarray): full displacement vector.
            om (float): angular frequency value.
        """
        pass