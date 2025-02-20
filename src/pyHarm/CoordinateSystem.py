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

import numpy as np

def generateCoordinateSystem(dirs:list[list[float]]):
    """Generates a CoordinateSystem object based on the provided directions.
    
    Args:
        dirs (list[list[float]]): A list of lists containing the directions relative to the global
            coordinate system.
    
    Returns:
        CoordinateSystem: A CoordinateSystem object.
    """
    return CoordinateSystem(dirs)

  
class CoordinateSystem:
    """Class that represents a coordinate system. It allows for generating local coordinate systems
    to be attached to elements or substructures and transfering their residuals and Jacobians
    to the global coordinate system.
    
    Args:
        dirs (list[list[float]]): A list of lists containing the directions relative to the global
            coordinate system.
    """
    def __init__(self, dirs:list[list[float]]) : 
        self.dirs = np.array(dirs) / np.linalg.norm(np.array(dirs),axis=1).reshape(-1,1)
        self.n_dirs = self.dirs.shape[0]
        self.n_component = self.dirs.shape[1]
        self.checkOrthonormal()

    def checkOrthonormal(self,):
        """Checks if the provided coordinate system is orthonormal.
        
        Raises:
            ValueError: If the coordinate system is not orthonormal.
        """
        if (np.round(self.dirs @ self.dirs.T,6) == np.eye(self.dirs.shape[0])).all() : 
            pass
        else : 
            raise ValueError(f"The provided coordinate system is not orthonormal as P@P.T is not identity\n P@P.T={self.dirs @ self.dirs.T}")
        pass
        
    def getTM(self, nh:int, component:list[int]) -> np.ndarray:
        """Generates a transform matrix of size (ncompo, ncompo, n_dirs).
        
        Args:
            nh (int): Number of harmonics.
            component (list[int]): The components of the transform matrix.
            
        Returns:
            np.ndarray: The generated transform matrix.
        """
        acomponent = np.array(component)
        n_compo = len(acomponent)
        Pdir = np.zeros((self.n_dirs,(2*nh+1),(2*nh+1)*n_compo))
        for k,direction in enumerate(self.dirs) : 
            Pdir[k,:,:] = np.kron(np.eye((2*nh+1)),direction[acomponent])
        return Pdir

class GlobalCoordinateSystem(CoordinateSystem):
    """Subclass of CoordinateSystem that allows defining global coordinate systems.
    The initialization is modified for ease of instantiation.
    
    Args:
        ndirs (int): The number of directions in the global coordinate system.
    """
    def __init__(self, ndirs:int):
        self.dirs = np.eye(ndirs)
        self.n_dirs = self.dirs.shape[0]
        self.n_component = self.dirs.shape[1] 
        self.checkOrthonormal()