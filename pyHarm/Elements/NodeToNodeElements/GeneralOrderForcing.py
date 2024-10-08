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

from numba import njit
import numpy as np
from pyHarm.Elements.NodeToNodeElements.NodeToNodeElement import NodeToNodeElement

@njit(cache=True)
def GOFResidual(x,om,loadvec,Pdir,Pslave,dto,nabo,amp):
    R = np.zeros((len(x),))
    for direction in range(Pdir.shape[0]):
        force = (om**dto * nabo) @ (amp*loadvec)
        f_harm = (Pslave).T @ Pdir[direction,:,:].T @ force
        R -= f_harm
    return R
@njit(cache=True)
def GOFJacobian(x,om,loadvec,Pdir,Pslave,dto,nabo,amp):
    dJdx = np.zeros((len(x),len(x)))
    dJdom = np.zeros((len(x),))
    for direction in range(Pdir.shape[0]):
        djdom =  (Pslave).T @ Pdir[direction,:,:].T @ (dto*om**(dto-1) * nabo) @ (amp*loadvec)
        dJdom -= djdom
    return dJdx,dJdom
    

class GeneralOrderForcing(NodeToNodeElement): 
    """
    This element is the general polynomial external forcing. 
    
    A forcing is an element that cannot be applied in between substructures (only considers the first entry in "connect" keyword).

    Attributes:
        dto (int): order of the time derivative.
        ho (int): order of the harmonic where the forcing is applied.
        phi (float): phase lag to apply between the cosine and sine term (0 = pure cosinus forcing).
        amp (float): amplitude of the forcing.
    """
    factory_keyword : str = "GOForcing"
    """str: keyword that is used to call the creation of this class in the system factory."""
    def __post_init__(self,):
        self.dto = self.data["dto"]
        self.ho = self.data["ho"]
        if "phi" not in self.data.keys() : 
            self.phi = 0.
        else : 
            self.phi = float(self.data["phi"])
        self.loadvec = self._loadingdofs()
        self.amp = self.data["amp"]
        self.nabo = np.linalg.matrix_power(self.nabla,self.dto)
        self.flag_elemtype = -1 # it does not contribute to any of the system matrices

    def __flag_update__(self) : 
        self.flag_extforcing = True

    def _loadingdofs(self,):
        loadvec = np.zeros((2*self.nh+1,))
        if self.ho == 0 :
            loadvec[0] = 1
        else : 
            loadvec[2*(self.ho-1)+1:2*(self.ho-1)+3] = np.array([np.cos(self.phi),np.sin(self.phi)])
        return loadvec 
    
    def _evalResidual(self, x, om):
        return GOFResidual(x,om,self.loadvec,\
                             self.Pdir,self.Pslave,\
                             self.dto,self.nabo,self.amp)
    
    def _evalJacobian(self, x, om):
        return GOFJacobian(x,om,self.loadvec,\
                             self.Pdir,self.Pslave,\
                             self.dto,self.nabo,self.amp)
    
    def adim(self,lc,wc):
        """Modifies the element properties according to the characteristic length and angular frequency.
        
        Attributes:
            amp (float): modified amplitude according to the characteristic parameters.
            flag_adim (bool): True if equations are to be adimensionalized
        """
        self.amp = self.amp * (wc**self.dto)
        self.flag_adim = True

    def evalResidual(self, xg, om):
        """Computes the residual.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.

        Returns:
            np.ndarray: residual vector.
        """
        x = xg[self.indices]
        self.R = self._evalResidual(x, om)
        return self.R
    
    def evalJacobian(self, xg, om):
        """Computes the jacobians.
        
        Args:
            xg (np.ndarray): full displacement vector.
            om (float): angular frequency value.

        Returns:
            np.ndarray: Jacobian with respect to displacement.
            np.ndarray: Jacobian with respect to angular frequency.
        """
        x = xg[self.indices]
        self.J,self.dJdom = self._evalJacobian(x, om)
        return self.J,self.dJdom