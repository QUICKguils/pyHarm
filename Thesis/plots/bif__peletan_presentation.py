"""Bifurcation: Peletan model with uncontrolled perturbations"""

import numpy as np

from pyHarm.DynamicOperator import compute_DFT
from pyHarm.Maestro import Maestro

# Define the Duffing oscillator
# m*q_ddot + c*q_dot + k*q + gamma*q^3 = f(t)
# with parameters from Peletan article
linsys = dict()
linsys["M"] = np.array([1])
linsys["C"] = np.array([1])
linsys["K"] = np.array([1000])
linsys["G"] = 0 * linsys["M"]
f = 2000
gamma = 10_000

DATA = {
    "analysis": {
        "duffing_frf": {
            "study": "frf",

            "puls_inf": 0.01,
            "puls_start": 0.1,
            "puls_sup": 190,
            "ds0": 1e-2,
            "ds_min": 1e-8,
            "ds_max": 5e-1,

            "stepsizer": "acceptance",
            "predictor": "tangent",
            "corrector": "arc_length",
            "bifurcator": "jump",
            "stopper": "bounds",
            "solver": "NewtonRaphson",
        },
    },
    "system": {
        "type": "Base",
        "nti": 2048,
        "adim": { "status": False, "lc": 1.0, "wc": 1.0, },
    },
    "substructures": {
        "duffing": {
            "matrix": linsys,
            "ndofs": 1,
        },
    },
    "connectors": {
        "loading": {
            "type": "CosinusForcing",
            "connect": {"duffing": [0]},
            "dirs": [0],
            "amp": f,
        },
        "cubic_spring": {
            "type": "CubicSpring",
            "connect": {"duffing": [0]},
            "dirs": [0],
            "k": gamma,
        },
    },
}


def main():
    nh_list = [12]

    freqs_list = []
    ampls_list = []
    for nh in nh_list:
        freqs, ampls = solve_one(nh)
        freqs_list.append(freqs)
        ampls_list.append(ampls)

    sol = (freqs_list, ampls_list)

    plot(*sol)

    return sol


def solve_one(nh):
    # Update problem definition
    DATA["system"]["nh"] = nh

    M = Maestro(DATA)

    # These are the two only methods defined for a Maestro object.
    ix_dof = M.getIndex("duffing", 0, 0)
    try:
        M.operate()
    except KeyboardInterrupt:
        pass

    # Retrieve solutions.
    sol_accepted = [sol for sol in M.nls["duffing_frf"].SolList if sol.flag_accepted]
    freqs = np.array([sol.x[-1] for sol in sol_accepted])  # Extract the Fourier coefficients
    n_sol = freqs.shape[0]

    # Discrete Fourier Transform Operator
    DFTO = compute_DFT(M.system.nti, M.system.nh)
    # organize the Fourier coefficients (cos/sin) of the solution,
    # each column representing a solution at a particular excitation frequency.
    cs_accepted = np.concatenate([sol.x.reshape(-1, 1) for sol in sol_accepted], axis=1)
    ampls = np.empty(n_sol)
    for ix_sol in range(n_sol):
        # Filter the cs of the desired dof,
        # and get the time domain representation, i.e., the displacement q.
        displ = cs_accepted[ix_dof, ix_sol] @ DFTO["ft"]
        # Find the maximum of the displacement amplitude
        # for each excitation frequencies, a.k.a. the FRF.
        # RMS value or other norm styles could have been used.
        ampls[ix_sol] = np.max(displ)

    return freqs, ampls


def plot(freqs_list, ampls_list) -> None:
    import matplotlib.pyplot as plt
    from ..mplrc import load_rcparams, REPORT_TW

    load_rcparams()
    fig, axs = plt.subplots(2, 1, figsize=(REPORT_TW, 0.8*REPORT_TW))

    for freqs, ampls in zip(freqs_list, ampls_list):
        axs[0].plot(freqs, ampls, marker=".")

        zoom_mask = freqs < 0.8 # rad/s
        freqs_zoom = freqs[zoom_mask]
        ampls_zoom = ampls[zoom_mask]

        axs[1].plot(freqs_zoom, ampls_zoom)

    for ax in axs:
        ax.set_xlabel("Frequency (rad/s)")
        ax.set_ylabel("Amplitude (m)")

    fig.show()
