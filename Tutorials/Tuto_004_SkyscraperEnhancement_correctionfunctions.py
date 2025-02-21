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

import copy
from typing import Callable

import numpy as np


def mass_per_floor_corr(rho: float, area: float, thickness: float) -> float:
    """Function that returns the mass per floor using the mass per volume, the area of the floor and
    the thickness of the slabs. 2 slabs are present per floor"""
    return rho * 2 * (area * thickness)


def build_mass_matrix_corr(
    mass_per_floor_value: float, mass_ground_value: float, N_floor: int
) -> np.ndarray:
    """Fills the function in order to return the mass matrix of the building.
    The mass matrix shall be ordered so that the ground is the last dof in the matrix and the top floor
    is the first dof. The slabs are also present at the ground floor.
    Use np.diag in order to generate the matrix easily."""
    M = np.diag([mass_per_floor_value] * N_floor)
    M[-1, -1] += mass_per_floor_value
    return M


def k_approx_corr(f: float, m: float) -> float:
    """Function that returns the value of the estimated rigidity out
    of the natural angular frequency of a single mass oscillator"""
    k = (2 * np.pi * f) ** 2 * m
    return k


def create_K_corr(N_floor, ke):
    """Creates the rigidity matrix from the number of floor and
    the elementary rigidity calculated previously"""
    K = np.diag(ke * np.ones(N_floor))
    for i in range(K.shape[0] - 1):
        if i != 0:
            K[i, i] += ke
        K[i + 1, i] -= ke
        K[i, i + 1] -= ke
    K[-1, -1] += 1e2 * ke
    return K


def get_modal_analysis_corr(
    M: np.ndarray, K: np.ndarray
) -> tuple[: np.ndarray, : np.ndarray]:
    """Function that returns natural frequencies and
    the mode shape based on the given mass matrix and rigidity matrix.
    Use np.linalg.eig function in order to solve the eigenvalue problem.
    Don't forget to transform the angular frequency into frequencies.
    You can sort out the frequencies if you want at the end, but don't forget to make the
    transformation to the columns of the mode shape as well."""
    Mi = np.linalg.inv(M)
    om2, phi = np.linalg.eig(Mi @ K)
    fp = np.sqrt(om2) / (2 * np.pi)
    return fp, phi


def add_loading_connector_corr(INP: dict, forcing_node: int, amp: float) -> dict:
    """Function that generates a new input dictionnary
    - copy the template input file
    - Add a CosinusForcing into the input file with the given amp and forcing_node"""
    INP_out = copy.deepcopy(INP)
    load = {
        "type": "CosinusForcing",
        "connect": {"sub1": [forcing_node]},
        "dirs": [0],
        "amp": amp,
    }
    INP_out["connectors"]["loading"] = load
    return INP_out


def normal_creation_corr(mean: float, std: float, energy: float = 1.0) -> Callable:
    """Function that generates a normal distribution law based on the given parameters"""

    def normal(om: np.ndarray) -> np.ndarray:
        """Based on the given parameters in the normal_creation function
        returns the distribution value"""
        log10_om = np.log10(om)
        log10_mean = np.log10(mean * 2 * np.pi)
        sigma = std**2
        f = energy * (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-1 / 2.0 * ((log10_om - log10_mean) / (sigma)) ** 2)
        )  # here to
        return f

    return normal


def estimate_energy_corr(amp, om_min, om_max):
    """Function that estimates the energy from the previous loading."""
    return amp * (om_max - om_min)


def LawResidual_corr(
    x: np.ndarray, om: np.ndarray, loadvec: np.ndarray, law: Callable
) -> np.ndarray:
    amplitude = law(om)
    R = -amplitude * loadvec
    return R


def add_loading_connector_law_corr(INP: dict, forcing_node: int, law: Callable) -> dict:
    """Function that generates a new input dictionnary
    - copy the template input file
    - Add a CosinusForcing into the input file with the given amp and forcing_node"""
    INP_out = copy.deepcopy(INP)
    load = {
        "type": "LawForcing",
        "connect": {"sub1": [forcing_node]},
        "dirs": [0],
        "law": law,
    }
    INP_out["connectors"]["loading"] = load
    return INP_out


def add_tunedMass_to_INP_corr(
    INP: dict,
    dict_of_matrices: dict[str, np.ndarray],
    spring_value: float,
    force_distribution,
    N_floor,
    tunedMass,
    max_response_f: float = 5e-1,
):
    INP_new = add_loading_connector_law_corr(INP, N_floor - 1, force_distribution)
    coef_range = 0.5
    INP_new["analysis"]["FRF"]["puls_inf"] = (
        (1.0 - coef_range) * max_response_f * (2 * np.pi)
    )
    INP_new["analysis"]["FRF"]["puls_sup"] = (
        (1.0 + coef_range) * max_response_f * (2 * np.pi)
    )
    INP_new["substructures"]["tunedMass"] = {"matrix": tunedMass, "ndofs": 1}
    INP_new["connectors"]["spring"] = {
        "type": "LinearSpring",
        "connect": {"sub1": [0], "tunedMass": [0]},
        "dirs": [0],
        "k": spring_value,
    }
    return INP_new


def add_dashpot_to_INP_corr(INP: dict, dashpot_value: float):
    INP_new = copy.deepcopy(INP)
    INP_new["connectors"]["dashpot"] = {
        "type": "LinearDamper",
        "connect": {"sub1": [0], "tunedMass": [0]},
        "dirs": [0],
        "k": dashpot_value,
    }
    return INP_new


def weibull_creation_corr(
    lam: float, k: float, energy: float = 1.0, height=1.0
) -> Callable:
    """Function that generates a weibull distribution law based on the given parameters"""

    def weibull_corr(om: np.ndarray) -> np.ndarray:
        """Based on the given parameters in the weibull_creation function
        returns the distribution value"""
        ### MODIFICATIONS ARE TO BE DONE BELLOW
        f = energy * (
            k / lam * (om / lam) ** (k - 1) * np.exp(-((om / lam) ** (k)))
        )  # here to
        ### END MODIFICATION
        f *= height  # linear dependence of the height of the floor
        return f

    return weibull_corr


def add_loading_connector_law_wind_corr(
    INP: dict,
    list_forcing_node_and_height: list[tuple[int, float]],
    law_constructor: Callable,
    max_response_f: float,
    LAMBDA_WIND,
    K_WIND,
    energy,
) -> dict:
    """Function that generates a new input dictionnary
    - copy the template input file
    - for each floor add a forcing generated using weibull_creation function with the right height"""
    INP_out = copy.deepcopy(INP)
    for forcing_node, height in list_forcing_node_and_height:
        ### MODIFICATIONS ARE TO BE DONE BELLOW
        law = weibull_creation_corr(LAMBDA_WIND, K_WIND, energy * 1e-6, height)
        load = {
            "type": "LawForcing",
            "connect": {"sub1": [forcing_node]},
            "dirs": [0],
            "law": law,
        }
        ### END MODIFICATIONS
        INP_out["connectors"][f"loading_h_{height:.2e}"] = load
    coef_range = 0.4
    INP_out["analysis"]["FRF"]["puls_inf"] = (
        (1.0 - coef_range) * max_response_f * (2 * np.pi)
    )
    INP_out["analysis"]["FRF"]["puls_sup"] = (
        (1.0 + coef_range) * max_response_f * (2 * np.pi)
    )
    return INP_out
