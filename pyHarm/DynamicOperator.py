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
from scipy.linalg import block_diag

def compute_DFT(nti:int, nh:int) -> dict[str:np.ndarray]:
        """
        Builds the Discrete Fourier Transform (DFT) operator adapted to the desired number of time samples and harmonics.
        
        Args:
            nti (int): Number of time steps.
            nh (int): Number of harmonics.
            
        Returns:
            dict[np.ndarray]: Dictionary containing DFT and DTF operators.
        """
        D = dict()
        D['ft'] = np.zeros((2*nh+1, nti))
        D['ft'][0, :] = 1.
        D['ft'][1::2, :] = np.cos(np.reshape(np.arange(1, nh+1), (-1, 1)) * np.arange(0, nti)*2.*np.pi/nti)
        D['ft'][2::2, :] = np.sin(np.reshape(np.arange(1, nh+1), (-1, 1)) * np.arange(0, nti)*2.*np.pi/nti)
        D['tf'] = np.zeros((nti, 2*nh+1))
        D['tf'][:, 0] = 1./nti
        D['tf'][:, 1::2] = np.cos(np.reshape(np.arange(0, nti), (-1, 1)) * np.arange(1, nh+1)*2.*np.pi/nti) * 2./nti
        D['tf'][:, 2::2] = np.sin(np.reshape(np.arange(0, nti), (-1, 1)) * np.arange(1, nh+1)*2.*np.pi/nti) * 2./nti
        return D

def nabla(nh:int) -> np.ndarray:
    """
    Builds the Derivation operator.
    
    Args:
        nh (int): Number of harmonics.
        
    Returns:
        np.ndarray: Derivation operator in the frequency domain.
    """
    nabla = 0
    nh = nh
    for i in range(nh):
        nabla = block_diag(nabla, (i+1)*np.array([[0, 1.0], [-1.0, 0]]))
    return nabla