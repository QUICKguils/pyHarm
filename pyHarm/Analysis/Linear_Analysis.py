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

from pyHarm.Analysis.ABCAnalysis import ABCAnalysis
from pyHarm.Solver import FirstSolution,SystemSolution
from pyHarm.Systems.ABCSystem import ABCSystem
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from scipy import linalg
import numpy as np


class Linear_Analysis(ABCAnalysis):
    """
    Modal analysis and linear forced response analysis.

    Performs a linear modal analysis and then proceeds to a linear FRF by using mode superposition
    
    Attributes:
        system (ABCSystem): ABCSystem associated with the analysis.
        analysis_options (dict): input dictionary completed with the default values if keywords are missing.
        flag_print (bool): if True, prints a message after each Solve method.
        SolList (list[SystemSolution]): list of SystemSolution stored.
        eigensol (dict): output dictionary containing the eigenfrequencies and the eigenvectors
    """
    
    factory_keyword : str = "linear_analysis"
    """str: keyword that is used to call the creation of this class in the system factory."""

    name = "Linear modal and FRF analysis"

    default={"puls_inf":.1,
             "puls_sup":1.0,
             "num_linear_puls": 100,
             "verbose":True,
             "damping": {"modal": {"xi":  0.001}}}

    """dict: default dictionary for the analysis."""
    
    def __init__(self, inputData:dict, System:ABCSystem):
        self.system = System
        self.analysis_options = getCustomOptionDictionary(inputData,self.default)
        self.flag_print = self.analysis_options["verbose"]
        self.SolList = []
        self.eigensol = {'eigenfrequencies': None,
                         'eigenvectors': None}
    
    def initialise(self):
        """
        Retrieves the mass and stiffness matrix of the assembled system
        
        Returns:
            K_global (np.ndarray): full stiffness matrix of the system
            M_global (np.ndarray): full mass matrix of the system
        """
        x0 = np.concatenate([np.zeros(self.system.ndofs_solve), np.array([0.])])
        M_assembled = self.system._get_assembled_mass_matrix(x0)
        if len(self.system.LE_nonlinear_dlft) != 0 :
            Rlin = np.zeros(self.system.ndofs)
            Rlin += self.system._residual(self.system.LE_extforcing,x0)
            Rlin += self.system._residual(self.system.LE_linear,x0)
            K_assembled = self.system._get_assembled_stiffness_matrix(x0,**{"Rglin":Rlin,"dJgdxlin":None,"dJgdomlin":None})
        else:
           K_assembled = self.system._get_assembled_stiffness_matrix(x0) 
        size_K = np.shape(K_assembled)[0] // (2*self.system.nh+1)
        K_global =  K_assembled[size_K:2*size_K, size_K:2*size_K]
        M_global =  M_assembled[size_K:2*size_K, size_K:2*size_K]
        return K_global, M_global

    def modal_analysis(self, K, M):
        """
        Eigenvalue analysis leading to the eigenfrequencies and the normalized right eigenvectors
        
        Args:
            K (np.ndarray): full stiffness matrix of the system
            M (np.ndarray): full mass matrix of the system

        Returns:
            omega (np.ndarray): eigenfrequencies in rad/s
            phi (np.ndarray): normalized to unity right eigenvectors
        """
        w, phi = linalg.eig(K, M)
        omega = np.sort(np.sqrt(np.absolute(w)))
        freq_Hz = omega[:10] / (2 * np.pi)
        if self.flag_print:
            print('First ten eigenfrequencies of the linear stuck system\n'+'\n'.join([f"{valeur:.2f} Hz".rjust(12) for valeur in freq_Hz]))
        return omega, phi

    def compute_linear_FRF(self, K, M, phi):
        """
        Linear frequency response function by means of mode superposition
        
        Args:
            K (np.ndarray): full stiffness matrix of the system
            M (np.ndarray): full mass matrix of the system
            phi (np.ndarray): normalized to unity right eigenvectors
        """
        damping = self.analysis_options['damping']

        # Generalized mass and stiffness matrices
        Mg = phi.T @ M @ phi
        Kg = phi.T @ K @ phi

        # Damping matrix
        # Option 1: Rayleigh
        if 'Rayleigh' in damping.keys():
            C = damping['Rayleigh']['coef_K'] * K + damping['Rayleigh']['coef_M'] * M
            Cg = phi.T @ C @ phi
        # Option 2: Modal damping
        elif 'modal' in damping.keys():
            Cg = 2 * damping['modal']['xi'] * np.sqrt(np.diag(np.diag(Kg))) @ np.sqrt(np.diag(np.diag(Mg)))

        # External forcing
        sub = list(self.system.LE_extforcing[0].data['connect'].keys())[0]
        node_num = self.system.LE_extforcing[0].data['connect'][sub][0]
        dof_num = self.system.LE_extforcing[0].data['dirs'][0]
        expl_dofs = self.system.expl_dofs
        dof_ex = expl_dofs[(expl_dofs['harm']==0) & (expl_dofs['sub']==sub) & (expl_dofs['node_num']==node_num) & (expl_dofs['dof_num']==dof_num)].index[0]
        F = np.zeros((len(K), 1))
        F[dof_ex] = self.system.LE_extforcing[0].data['amp']

        # Forced response
        om = np.linspace(self.analysis_options['puls_inf'], self.analysis_options['puls_sup'], self.analysis_options['num_linear_puls'])
        for omega in om:
            Z = Kg - omega**2 * Mg + (1j) * omega * Cg # Inverse of FRF or transfer function
            Q = np.linalg.inv(Z) @ (phi.T @ F)
            X = phi @ np.reshape(Q, (len(Q),))
            X = np.concatenate((np.abs(X).reshape(-1,1), np.asarray(omega).reshape(-1,1)))
            isol = FirstSolution(X)
            self.SolList.append(isol)

    def Solve(self, x0=None, **kwargs):
        """
        Solving step of the analysis.
        """
        K, M = self.initialise()
        omega, phi = self.makeStep(K,M)
        self.eigensol['eigenfrequencies'] = omega / (2 * np.pi)
        self.eigensol['eigenvectors'] = phi
    
    def makeStep(self,K,M):
        """
        Makes a whole step of the analysis.
        
        Args:
            K (np.ndarray): full stiffness matrix of the system
            M (np.ndarray): full mass matrix of the system

        Returns:
            omega (np.ndarray): eigenfrequencies in rad/s
            phi (np.ndarray): normalized to unity right eigenvectors
        """
        omega, phi = self.modal_analysis(K,M)
        self.compute_linear_FRF(K, M, phi)
        return omega, phi
    