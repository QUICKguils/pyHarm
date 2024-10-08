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
import copy



def get_sector_mass_mat_corr(m0,mb1,mb2) : 
    return np.diag(np.array([m0,mb1,mb2]))

def get_sector_rigid_mat_corr(kg,kb1,kb2) : 
    K = np.array([[kg+kb1, -kb1, 0 ],
                  [-kb1, kb1+kb2, -kb2],
                  [0, -kb2, kb2]])
    return K

    
def get_full_mass_mat_corr(M_sector, n_sector) : 
    M = np.kron(np.eye(n_sector), M_sector)
    return M
def get_full_rigid_mat_corr(K_sector, k0, n_sector) : 
    K = np.kron(np.eye(n_sector), K_sector)
    # make the link between sectors : 
    num_ii = 3*np.concatenate([np.arange(n_sector),np.zeros(1)]).astype(int)
    for k in range(n_sector) : 
        msgh = tuple(np.meshgrid(num_ii[k:k+2],num_ii[k:k+2],indexing="ij"))
        K[msgh] += np.array([[k0,-k0],
                             [-k0,k0]])
    return K


def get_M_inv_corr(M):
    M_inv = np.diag(1/np.diag(M))
    return M_inv


def make_eig_corr(M_inv, K) : 
    om2, phi = np.linalg.eig(M_inv@K)
    sorting = np.argsort(om2)
    om2, phi = om2[sorting], phi[:,sorting]
    f = np.sqrt(om2) / (2*np.pi)
    f, phi = np.real(f), np.real(phi)
    return f,phi


def get_m2_indexes_corr(node, explicit_dofs) : 
    indexes = explicit_dofs[explicit_dofs["node"]==node].index
    return indexes

def get_phi_m2_corr(phi, indexes_m2) : 
    phi_2 = phi[indexes_m2,:]
    return phi_2


def get_families_from_diam_corr(pd_diam) : 
    diam_num = pd_diam["diam_num"]
    loc_fam_cut = np.concatenate([np.array([-1]),np.where(np.sign(np.diff(diam_num))<0)[0],np.array([len(diam_num)])])
    list_fam = []
    for l in range(len(loc_fam_cut)-1) : 
        list_fam.append( pd_diam.iloc[loc_fam_cut[l]+1:loc_fam_cut[l+1]+1] )
    return list_fam


def get_damping_mat_corr(xi_damp, phi, pd_diam) : 
    C = np.zeros(phi.shape)
    C = phi @ (2*xi_damp*np.array(pd_diam["freq"])*(2*np.pi) * np.eye(phi.shape[0])) @ phi.T
    return C


def make_blisk_sub_corr(M,C,K) : 
    Linsys = {"M":M, "C":C, "K":K, "G":0*K}
    Blisk = {"matrix":Linsys, "ndofs":1}
    return Blisk


def make_system_input_corr(nh=1,nti=128):
    System = { 
        "type":"Base",
        "nh":nh,
        "nti":nti
    }
    return System


    
def make_analysis_corr(puls_inf, puls_sup, ds_min, ds_max, ds0, **kwargs) : 
    FRF = {
        "study":"frf",
        "puls_inf": puls_inf,
        "puls_sup": puls_sup,
        "ds_min":ds_min,
        "ds_max":ds_max,
        "ds0":ds0
    }
    for k,v in kwargs.items() : 
        FRF[k] = v
    return FRF

def get_dephase(num_sector, n_sector, n_diam): 
    dephase = 2*np.pi/n_sector * n_diam * num_sector
    return dephase


def make_forcing_corr(n_sector, n_diam, explicit_dofs) : 
    Forcing = dict()
    for n in range(n_sector):
        name_forcing = f"F_{n:03d}"
        dephase = get_dephase(n,n_sector,n_diam)
        applied_node = int(np.array(explicit_dofs[((explicit_dofs["sub"]==n)&
                                                   (explicit_dofs["node"]==2))].index))
        Forcing[name_forcing] = {
            "type":"GOForcing",
            "connect":{"blisk":[applied_node]},
            "dirs":[0],
            "amp":1e0,
            "dto":0,
            "ho":1,
            "phi":dephase
        }
    return Forcing


def make_ring_sector_sub_dict_corr(Mrs) : 
    matrix = {"M":Mrs, "C":0*Mrs, "K":0*Mrs, "G":0*Mrs}
    Sector_ring = {
        "matrix" : matrix,
        "ndofs": 1
    }
    return Sector_ring



def get_ring_sector_numerotation(n, n_sector) : 
    sector_str = f"{n:03d}" 
    sector_plus_one = (n+1)%n_sector
    sector_plus_one_str = f"{sector_plus_one:03d}"
    return sector_str, sector_plus_one_str
    
def get_blisk_node_to_attach(n, explicit_dofs) :
    node_blisk = np.array([explicit_dofs[((explicit_dofs["node"]==0) & 
                                (explicit_dofs["sub"]==n))].index])[0,0]
    return node_blisk

def add_ring_to_input_corr(inp, n_sector, explicit_dofs, kr, Sector_ring) : 
    for n in range(n_sector) :
        str_sn, str_np1 = get_ring_sector_numerotation(n, n_sector)
        node_blisk = get_blisk_node_to_attach(n, explicit_dofs)
        inp["substructures"]["ring_"+str_sn] = Sector_ring
        inp["connectors"]["LinkBlisk_"+str_sn] = {
            "type":"LinearSpring",
            "connect":{"ring_"+str_sn:[0], "blisk":[node_blisk]},
            "dirs":[0],
            "k":kr * 1e1
        }
    
        inp["connectors"]["Spring_"+str_sn+"_"+str_np1] = {
            "type":"LinearSpring",
            "connect":{"ring_"+str_sn:[0], "ring_"+str_np1:[0]},
            "dirs":[0],
            "k":kr
        }
    
        inp["connectors"]["SpringGround_"+str_sn] = {
            "type":"LinearSpring",
            "connect":{"ring_"+str_sn:[0]},
            "dirs":[0],
            "k":kr*1e-3
        }
    return inp 

    
def modify_inp_for_jenkins_corr(inp, mu, mr, radius_ring, omega_rot) : 
    for k,c in inp["connectors"].items() : 
        if "LinkBlisk" in k :
            c["type"] = "Jenkins"
            c["mu"] = mu
            c["N0"] = mr*radius_ring*omega_rot**2
    return inp