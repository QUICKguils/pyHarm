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
import pandas as pd

# 1.1 correction functions : 
def ElementaryMatrixResidual_corr(x:np.ndarray,om:float,kronAe:np.ndarray,dto:int) -> np.ndarray:
    R = (om**dto * kronAe) @ x
    return R

def ElementaryMatrixJacobian_corr(x:np.ndarray,om:float,kronAe:np.ndarray,dto:int) -> tuple[np.ndarray,np.ndarray]:
    dJdx = (om**dto * kronAe)
    dJdom = (dto * om**(dto-1) * kronAe) @ x
    return dJdx,dJdom


# 1.1 correction functions : 
def add_EM_connectors_corr(INP:dict, number_elements:int, Ae:np.ndarray, dto:int, prefix_name:str) -> dict:
    """
    Add ElementaryMatrix for each element.
    
    Args:
        inp (dict): input dictionary.
        number_elements (dict): number of elements.
        Ae (dict): Elementary matrix.
        dto (dict): order of time derivative.
        
    Returns:
        dict: modified input dictionary.
    """
    for k in range(number_elements):
        name_connector = f"_{k:02d}_{(k+1):02d}"
        INP["connectors"][prefix_name+name_connector] = {
        "type":"ElementaryMatrix",
        "connect":{"beam":[k],"INTERNAL":[(k+1)]},
        "dirs":[0,1],
        "A":Ae,
        "dto":dto
        }
    return INP


# 1.2.4. correction functions : 
def generate_C_phys_corr(om2:np.ndarray,phi:np.ndarray,xi:float,M:np.ndarray) -> np.ndarray:
    """
    Generates the modal equivalent damping matrix in the physical space
    
    Args: 
        om2 (np.ndarray): eigenvalues (square angular frequency)
        phi (np.ndarray): eigenvectors
        xi (float): modal damping value
        M (np.ndarray): mass matrix
        
    Returns:
        np.ndarray: modal damping matrix expressed in the physical space
    """
    C_mod = np.diag(2*np.sqrt(om2)*xi) # build modal damping matrix
    C_phys = (M@phi) @ C_mod @ (M@phi).T # project in physical base
    # Add the lines and column corresponding to clamped node
    C_phys = np.block([
        [np.zeros((2,2)),np.zeros((2,C_phys.shape[1]))],
        [np.zeros((C_phys.shape[0],2)),C_phys]
    ])
    return C_phys


# 1.3.1. correction functions : 
def rearange_matrices_corr(M:np.ndarray,C:np.ndarray,K:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    Arange_Mat = []
    for A in (M, C, K):
        A_part = np.block([
            [A[:2,:2],A[:2,-2:],A[:2,2:-2]],
            [A[-2:,:2],A[-2:,-2:],A[-2:,2:-2]],
            [A[2:-2,:2],A[2:-2,-2:],A[2:-2,2:-2]]
        ])
        Arange_Mat.append(A_part)
    return tuple(Arange_Mat)


# 1.3.2. correction functions : 
def make_modal_analysis(Mcond:np.ndarray, Kcond:np.ndarray) -> tuple[np.ndarray,np.ndarray] : 
    """
    Makes the modal analysis of the provided mass and rigidity matrices
    
    Args: 
        Mcond (np.ndarray): Mass matrix
        Kcond (np.ndarray): Rigidity matrix
        
    Returns:
        np.ndarray: sorted vector of square angular frequency values
        np.ndarray: sorted matrix of eigen modes
    """
    Minv = np.linalg.inv(Mcond)
    om2,phi = np.linalg.eig(Minv@Kcond)
    ii_sorted = np.argsort(om2)
    om2 = om2[ii_sorted]
    phi = phi[:,ii_sorted]
    scaler = np.sqrt(np.diag(1/np.diag(phi.T @ Mcond @ phi))) # the vectors are scaled according to mass matrix
    phi = (phi@scaler) # rescaled eigenvectors
    return om2, phi

def get_phi_dyn_corr(Mp,Kp):
    M_a, K_a = Mp[4:,4:], Kp[4:,4:]
    om2_a, phi_a = make_modal_analysis(M_a, K_a)
    return om2_a, phi_a

# 1.3.3. correction functions : 
def get_phi_stat_corr(Kp):
    K_ii, K_ib = Kp[4:,4:], Kp[4:,:4]
    phi_stat = - np.linalg.inv(K_ii) @ K_ib
    return phi_stat



# 1.3.4. correction functions : 
def get_phi_cb_corr(phy_dyn,phy_stat):
    phi = np.block([
        [np.eye(phy_stat.shape[1]),np.zeros((phy_stat.shape[1],phy_dyn.shape[1]))],
        [phy_stat,phy_dyn]
    ])
    return phi


# 1.3.5. correction functions :
def get_gen_matrices_corr(Mp, Cp, Kp, phi_CB):
    list_A = []
    for A in (Mp, Cp, Kp) : 
        list_A.append(
        phi_CB.T @ A @ phi_CB
        )
    return tuple(list_A)


# 2.2. correction functions :
def add_local_coordinate_corr(INP, local_coordinates):
    for l in local_coordinates : 
        series = local_coordinates[l]
        coor = [
            eval(series.loc["Normal"]),
            eval(series.loc["Tangent_1"]),
            eval(series.loc["Tangent_2"])
        ]
        INP["coordinates"][l] = coor
    return INP



# 2.3. correction functions :
def add_forcing_corr(INP, forcing_name, forcing_sub, forcing_node, forcing_dir, forcing_amp=.1) : 
    INP["connectors"][forcing_name] = {
        "type":"CosinusForcing",
        "connect":{forcing_sub:[forcing_node]},
        "dirs":[forcing_dir],
        "amp":.1
    }
    return INP

# 2.4. correction functions :
def add_friction_connectors_corr(INP, pairing_nodes, **kwargs) : 
    type_con = kwargs["type_con"]
    mu_con = kwargs["mu"]
    N0_con = kwargs["N0"]
    g_con = kwargs["g"]
    jac_con = kwargs["jac"]
    eps_con = kwargs["eps"]
    dir_con = kwargs["dirs"]
    for i in pairing_nodes.index : 
        infos = pairing_nodes.loc[i]
        name_con = f"Con_Aube_{infos['Aube']:03d}_Disque_{infos['Disque']:03d}"
        INP["connectors"][name_con] = {
            "type":type_con,
            "connect":{"Aube":[infos['Aube']],"Disque":[infos['Disque']]},
            "dirs":dir_con,
            "coordinatesystem":infos["Coordinates"],
            "mu":mu_con,
            "N0":N0_con,
            "g":g_con,
            "eps":eps_con,
            "jac":jac_con,
        }
    return INP


























