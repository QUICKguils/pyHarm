"""Context: pyharm linear vs Duffing"""

import numpy as np

from pyHarm.Maestro import Maestro
from pyHarm.DynamicOperator import compute_DFT


# Define the Duffing oscillator
# m*q_ddot + c*q_dot + k*q + gamma*q^3 = f(t)
linsys = dict()
linsys["M"] = np.array([1])
# linsys["C"] = np.array([0.05])  # Lazarus
linsys["C"] = np.array([0.3])  # Tuto 1
linsys["K"] = np.array([1])
linsys["G"] = 0 * linsys["M"]
gamma = 1
f = 1

# Build the global dictionary containing all
# the data of the problem.
DATA = {
    "analysis": {
        "duffing_frf": {
            "study": "frf",
            "puls_inf": 0.01,
            "puls_sup": 3.0,
            "ds0": 5e-3,
            "ds_min": 1e-8,
            "ds_max": 2e-2,

            "sign_ds": 1,
            "verbose": True,
            "stepsizer": "acceptance",
            "predictor": "tangent",
            "corrector": "arc_length",
            "stopper": "bounds",
        },
    },
    "system": {
        "type": "Base",
        "nti": 2048,
        "adim": {
            "status": False,
            "lc": 1.0,
            "wc": 1.0,
        },
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

    # Linear case, and duffing with 1 and 12 harmonics.
    nh_list = [1, 1, 12]
    gamma_list = [1E-8, 1, 1]
    label_list = ["linear", r"$n_h=1$", r"$n_h=12$"]

    freqs_list = []
    ampls_list = []
    for nh, gamma in zip(nh_list, gamma_list):
        freqs, ampls = solve_one(nh, gamma)
        freqs_list.append(freqs)
        ampls_list.append(ampls)

    sol = (freqs_list, ampls_list, label_list)

    plot(*sol)

    return sol


def solve_one(nh, gamma):
    # Update problem definition
    DATA["system"]["nh"] = nh
    DATA["connectors"]["cubic_spring"]["k"] = gamma

    # Maestro is the main user interface to pyHarm.
    #
    # It receives the comprehensive problem definition
    # as a global dictionary, then build the system and
    # returns it as a pyHarm Maestro object.
    M = Maestro(DATA)

    # These are the two only methods defined for a Maestro object.
    #
    # getIndex() returns the explicit dofs of the system, for the desired
    # substructure name, node number and direction.
    # These explicit dofs can be consulted with `M.system.expl_dofs`.
    #
    # operate() run all the analysis that were specified in the
    # input dictionary, and stores all the results in the Maestro object.
    # operate() can be put in the shown try-except block.
    # This allows to abort the computations with <CTRL-C> while preserving
    # the already computed values.
    # This can be useful when pyHarm is stuck in a PC infinite loop.
    ix_dof = M.getIndex("duffing", 0, 0)
    try:
        M.operate()
    except KeyboardInterrupt:
        pass

    # Retrieve solutions.
    # `SolList` is the list of all `Solver.SystemSolution` comuputed during
    # the continuation of a prescribed frf analysis (here, "duffing_frf" analysis).
    # Each particular solution is flagged as accepted if the nonlinear solver
    # used for the corrector steps has converged.
    sol_accepted = [sol for sol in M.nls["duffing_frf"].SolList if sol.flag_accepted]
    # Field `x` contains all the balanced Fourier coefficients of the solution.
    # The frequency at wich the solution is computed is appended afterwards.
    freqs = np.array([sol.x[-1] for sol in sol_accepted])  # Extract the Fourier coefficients
    n_sol = freqs.shape[0]

    # Discrete Fourier Transform Operator
    # They are available for each system elements,
    # but are actually computed by pyHarm.DynamicOperator.compute_DFT().
    # DFTO = M.nls["duffing_frf"].system.LE_nonlinear_nodlft[0].D
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


def plot(freqs_list, ampls_list, label_list) -> None:
    import matplotlib.pyplot as plt
    from ..mplrc import load_rcparams, REPORT_TW

    load_rcparams()
    fig, ax = plt.subplots(figsize=(0.6*REPORT_TW, 0.6*REPORT_TW))

    for freqs, ampls, label in zip(freqs_list, ampls_list, label_list):
        ax.plot(freqs, ampls, label=label)

    ax.set_xlabel("Frequency (rad/s)")
    ax.set_ylabel("Amplitude (m)")
    ax.legend()
    fig.show()
