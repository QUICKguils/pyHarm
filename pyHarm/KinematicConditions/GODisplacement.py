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
from pyHarm.KinematicConditions.ABCKinematic import ABCKinematic
import numpy as np


class GODisplacement(ABCKinematic) : 
    """Kinematic condition that imposes a displacement or any time derivative of displacement to a specific dof.
    
    Attributes:
        amp (float): amplitude to impose.
        ho (int): harmonic loaded.
        dto (int): order of the time derivative.
        phi (float): phase lag to impose on the harmonic (0 = pure cosinus loading).
    """
    factory_keyword:str = "GOdisplacement"
    """str: keyword that is used to call the creation of this class in the system factory."""

    default = {"phi":0., "ho":1, "dto":0}
    """dict: dictionary containing the default parameters of the kinematic condition"""

    def __post_init__(self,):
        self.data = getCustomOptionDictionary(self.data,self.default)
        self.amp = self.data["amp"]
        self.ho = self.data["ho"]
        self.dto = self.data["dto"]
        if "phi" not in self.data.keys() : 
            self.phi = 0.
        else : 
            self.phi = float(self.data["phi"])
        self.loadvec = self._loadingdofs()

    def _loadingdofs(self,):
        loadvec = np.zeros((2*self.nh+1,))
        if self.ho == 0 :
            loadvec[0] = 1
        else : 
            loadvec[2*(self.ho-1)+1:2*(self.ho-1)+3] = np.array([np.cos(self.phi),np.sin(self.phi)])
        return loadvec 
    
    def complete_x(self, x) :
        """Returns a vector x_add of same size of x that completes the vector of displacement x = x + x_add such that the kinematic condition is verified.
        
        Args:
            x (np.ndarray): displacement vector.
            om (float): angular frequency.

        Returns:
            (np.ndarray): vector of displacement to add to the displacement vetor in order to impose the kinematic condition.
        """
        om = x[-1]
        xadd = np.zeros(x.shape)
        for direction in range(self.Pdir.shape[0]):
            xadd[self.indices] += (self.Pslave).T @ self.Pdir[direction,:,:].T @ (self.amp/(om**(self.dto))*self.loadvec)
        return xadd
        
    def dxbdxom(self, xg) : 
        """Computes the derivative of the kinematicaly constrained dofs with respect to the displacement.
        
        Args:
            xg (np.ndarray): full size displacement vector.

        Returns:
            (np.ndarray): vector of displacement contributions to add to the displacement vetor in order to impose the kinematic condition.
        """
        om = xg[-1]
        x = xg[self.indices]
        dxbdom = np.zeros(x.shape[0])
        for direction in range(self.Pdir.shape[0]):
            dxbdom += (self.Pslave).T @ self.Pdir[direction,:,:].T @ (-self.dto*self.amp/(om**(self.dto+1))*self.loadvec)
        return dxbdom.reshape(-1,1)
        
    def complete_R(self, R, x):
        """Computes the transfer of residual of the kinematicaly constrained dofs with respect to the displacement.
        
        Args:
            R (np.ndarray): residual vector.
            x (np.ndarray): displacement vector.

        Returns:
            (np.ndarray): vector of residual contributions to add to the residual vetor in order to impose the kinematic condition.
        """
        R_add = np.zeros(R.shape)
        return R_add
    
    def complete_J(self, Jx, Jom, x):
        """Computes the transfer of jacobian of the kinematicaly constrained dofs with respect to the displacement.
        
        Args:
            Jx (np.ndarray): jacobian matrix with respect to displacement.
            Jom (np.ndarray): jacobian matrix with respect to angular frequency.
            x (np.ndarray): displacement vector.

        Returns:
            (tuple[np.ndarray,np.ndarray]): tuple containing the jacobians contributions to add to the jacobians in order to impose the kinematic condition.
        """
        Jx_add = np.zeros(Jx.shape)
        Jom_add = np.zeros(Jom.shape)
        dxbdom = self.dxbdxom(x)
        Jom_add += Jx[:,self.indices] @ dxbdom
        return Jx_add, Jom_add
    
    def adim(self, lc, wc):
        self.amp = self.amp / (lc* wc**(self.dto))