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

def get_resonance_corr(mass:float,rigidity:float) -> float:
    """functions that returns the natural frequency of the single mass problem
    defined by its rigidity and mass value"""
    return np.sqrt(rigidity/mass) / (2*np.pi)


def add_tuned_mass_corr(INP:dict,tuned_mass:dict[str,np.ndarray],k_link:float) -> dict:
    """
    Function that returns a copy on the input dictionnary containing the new substructure and 
    the connector to the main substructure
    - Add the new substructure into INP_new["substructures"] and name it "tunedMass"
    - Add the new connector beetween the new substructure and the original substructure
    """
    INP_new = copy.deepcopy(INP)
    ### modifications bellow
    INP_new["substructures"]["tunedMass"] = {
        "matrix":tuned_mass,
        "ndofs":1
    }
    INP_new["connectors"]["spring_link"] = {
        "type":"LinearSpring",
        "connect":{"sub1":[0],"tunedMass":[0]},
        "dirs":[0],
        "k":k_link,
    }
    ### end modifications
    return INP_new

def add_tuned_mass_spring_corr(INP:dict,k_link:float)->dict:
    """Function that copies the input dictionary and returns a new one with the added linear spring
    - Add the linear spring connector between the original substructure and the ground by defining a
    single key,value in the "connect" argument
    """
    INP_new = copy.deepcopy(INP)
    ### modifications bellow
    INP_new["connectors"]["spring_link"] = {
        "type":"LinearSpring",
        "connect":{"sub1":[0]},
        "dirs":[0],
        "k":k_link,
    }
    ### end modifications
    return INP_new

def smooth_friction_corr(xdot,mu,n0,epsilon) : 
    """Function that computes the nonlinear forces using 
    the tanh regularization of Coulomb friction model."""
    f_fric = -mu*n0*np.tanh(xdot/epsilon)
    return f_fric


def residual_regulfriction_corr(x, om, Pslave, Pmaster, nabo, DFT, DTF,
                          mu, n0, epsilon) : 
    """
    This is the residual contribution of the smooth coulomb friction model made previously.
    - Construct the relative harmonic speed between by applying (Pslave-Pmaster) operator on the dof vector and use the derivative operator nabo. 
    - Construct the relative time speed by applying the inverse Fourier operator
    - Apply the `smooth_friction` function coded earlier to the relative speed
    - Transform the force back into harmonic space
    - Apply the force onto the right dofs by using (Pslave-Pmaster).T
    """
    xdot_relative = (nabo@((Pslave-Pmaster)@x)) @ DFT
    force_fric_t = smooth_friction_corr(xdot_relative,mu,n0,epsilon)
    force_fric_h = (Pslave-Pmaster).T@(force_fric_t @ DTF)
    return force_fric_h

def add_tuned_mass_plus_friction_corr(INP,tuned_mass,k_link,mu,n0,epsilon,add_tuned_mass):
    """
    Function that adds the friction element to the input dictionary.
    - Use previous `add_tuned_mass` function
    - Complete the new input dictionary by adding the friction element
    """
    INP_new = add_tuned_mass(INP,tuned_mass,k_link)
    ### modifications bellow
    INP_new["connectors"]["friction"] = {
        "type":"regulfric",
        "connect":{"tunedMass":[0]},
        "dirs":[0],
        "mu":mu,
        "N0":n0,
        "eps": epsilon
    }
    ### end modifications
    return INP_new

def modify_forcing_corr(INP:dict,forcing_coef:float) -> dict:
    """
    Function that modifies the amplitude of the forcing into the input dictionary
    - Make a copy of the input dictionary
    - forcing_new = forcing_old * forcing_coef
    """
    ### modifications bellow
    INP_new = copy.deepcopy(INP)
    INP_new["connectors"]["loading"]["amp"] *= forcing_coef
    ### end modifications
    return INP_new
