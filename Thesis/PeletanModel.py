"""PeletanModel -- Simple Diffing oscillator from Peletan"""

from typing import NamedTuple

import numpy as np

from pyHarm.Bifurcators.ABCBifurcator import BifurcationType
from pyHarm.DynamicOperator import compute_DFT
from pyHarm.Maestro import Maestro
from pyHarm.Solver import SystemSolution, get_last_solution

# Define the Duffing oscillator
# m*q_ddot + c*q_dot + k*q + gamma*q^3 = f(t)
# with parameters from Peletan article
linsys = dict()
linsys["M"] = np.array([1])
linsys["C"] = np.array([1])
linsys["K"] = np.array([-1000])
linsys["G"] = 0 * linsys["M"]
f = 2000
gamma = 10_000


MODEL = {
    "analysis": {},
    "system": {},
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

# The "nh" field need to be completed
# on the corresponding solving file.
SYSTEM = {
    "type": "Base",
    "nti": 2048,
    "adim": {
        "status": False,
        "lc": 1.0,
        "wc": 1.0,
    },
}

CHUNCK_ALL = {
    "analysis": {
        "cont": {
            "study": "frf",

            "puls_inf": 0.0,
            "puls_start": 0.001,  # starting from 0.1 gives craszy results
            "puls_sup": 190,
            "ds0": 2e-3,
            "ds_min": 1e-6,
            "ds_max": 1e-1,
            "sign_ds": 1,

            "reductors": [
                {
                    "type": "globalHarmonic",
                    "nh_start": np.array([0]),
                    "err_admissible": 1e10,
                    "h_always_kept": np.array([0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]),
                    "verbose": False,
                }
            ],
            "stepsizer": "acceptance",
            "predictor": "tangent",
            "corrector": "arc_length",
            "stopper": "bounds",
            "bifurcator": "jump",
            "bifurcator_options": {
                # "pert": -1E-2,
                # "interactive": True,
                # "vizu": [("duffing", 0, 0)],
            },
            "solver": "NewtonRaphson",
            "solver_options": {"max_iter": 200},
            # "solver_options": {"max_iter": 5, "pert": 1E-2},
        },
    },
}

CHUNCK_LIST = [CHUNCK_ALL]


class Continuation(NamedTuple):
    M: Maestro  # Maestro object created for the first continuation chunck.
    sol_list: list[SystemSolution]  # Gathered SolList of all the continuation chuncks.
    chunk_list: list[dict]  # "analysis" keys of all the continuation chuncks.


def _get_dof_idx(M: Maestro):
    """Get SystemSolution indexes of the required dof.

    Here, it just returns all the indexes, as there is only one DOF
    in this simple Duffing oscillator.
    """
    return M.getIndex("duffing", 0, 0)  # node 0, dir 0 of subsystem "duffing"


def _filter_harm(idx: np.ndarray[int], ih: list[int]) -> np.ndarray[int]:
    """Filter a SystemSolution index vector, to select only the desired harmonics."""
    expl_ih = np.array([[2 * i - 1, 2 * i] for i in ih], dtype=int).ravel()
    expl_ih = expl_ih[expl_ih >= 0]  # remove the -1 generated for potential 0-harm

    filtered = np.zeros(idx.shape, dtype=int)
    filtered[expl_ih] = idx[expl_ih]

    return filtered


def _spot_solution(cont: Continuation, om: float, ampl: float) -> SystemSolution:
    """Get a SystemSolution near the specified point (om, ampl) of the continuation curve."""
    om_list = [sol.x[-1] for sol in cont.sol_list]
    ampl_list = compute_amplitude(cont)
    curve_sdist = np.array(  # square distance from the continuation curve
                           [(om_i - om) ** 2 + (ampl_i - ampl) ** 2 for (om_i, ampl_i) in zip(om_list, ampl_list)]
                           )
    id = np.argmin(curve_sdist)

    return cont.sol_list[id]


def compute_amplitude(cont: Continuation, ih=None, pred=False) -> np.ndarray[float]:
    """Compute max. amplitude of given continuation, for specified harmonics."""
    DFTO = compute_DFT(cont.M.system.nti, cont.M.system.nh)
    idx = _get_dof_idx(cont.M)

    filtered = idx if ih is None else _filter_harm(idx, ih)

    def select_pred(sol: SystemSolution):
        return sol.x_pred if pred else sol.x

    return np.array([np.max(select_pred(sol)[filtered] @ DFTO["ft"]) for sol in cont.sol_list])


def extract_accepted(cont: Continuation):
    cont_accepted = Continuation(
        M=cont.M,
        sol_list=[sol for sol in cont.sol_list if sol.flag_accepted],
        chunk_list=cont.chunk_list,
    )
    cont_rejected = Continuation(
        M=cont.M,
        sol_list=[sol for sol in cont.sol_list if not sol.flag_accepted],
        chunk_list=cont.chunk_list,
    )
    return cont_accepted, cont_rejected


def extract_bifurcation(cont: Continuation):
    cont_bifurcation = Continuation(
        M=cont.M,
        sol_list=[sol for sol in cont.sol_list if sol.bifurcation_type is not None],
        chunk_list=cont.chunk_list,
    )
    cont_fold = Continuation(
        M=cont.M,
        sol_list=[
            sol for sol in cont_bifurcation.sol_list if sol.bifurcation_type is BifurcationType.FOLD
        ],
        chunk_list=cont.chunk_list,
    )
    cont_branching = Continuation(
        M=cont.M,
        sol_list=[
            sol for sol in cont_bifurcation.sol_list if sol.bifurcation_type is BifurcationType.BRANCHING
        ],
        chunk_list=cont.chunk_list,
    )
    cont_unknown = Continuation(
        M=cont.M,
        sol_list=[
            sol for sol in cont_bifurcation.sol_list if sol.bifurcation_type is BifurcationType.UNKNOWN
        ],
        chunk_list=cont.chunk_list,
    )

    return cont_bifurcation, cont_fold, cont_branching, cont_unknown


def continuation_plotter():
    import matplotlib.pyplot as plt
    from .mplrc import load_rcparams

    load_rcparams()
    fig, ax = plt.subplots()
    fold_style = {"color": "C1", "s": 50, "zorder": 2.5, "marker": "x"}
    branching_style = {"color": "C6", "s": 50, "zorder": 2.5, "marker": "x"}
    unknown_style = {"color": "C3", "s": 50, "zorder": 2.5, "marker": "x"}
    ax.scatter([], [], **fold_style, label="fold bifurcation")
    ax.scatter([], [], **branching_style, label="branching point")
    ax.scatter([], [], **unknown_style, label="other bif.")

    def plot(cont: Continuation, ih=None, pred=False):
        cont_accepted, cont_rejected = extract_accepted(cont)
        cont_bifurcation, cont_fold, cont_branching, cont_unknown = extract_bifurcation(cont_accepted)

        if len(cont_accepted) != 0:
            om_accepted = np.array([sol.x[-1] for sol in cont_accepted.sol_list])
            ampl_accepted = compute_amplitude(cont_accepted, ih)
            if pred:
                pred_style = {"color": "C7", "marker": "."}
                om_pred_accepted = np.array([sol.x_pred[-1] for sol in cont_accepted.sol_list[:-1]])
                ampl_pred_accepted = compute_amplitude(cont_accepted, ih, pred=True)[:-1]
                ax.scatter(om_pred_accepted, ampl_pred_accepted, **pred_style, label="prediction")
                for i in range(len(om_accepted) - 1):
                    ax.plot(
                        [om_pred_accepted[i], om_accepted[i]],
                        [ampl_pred_accepted[i], ampl_accepted[i]],
                        color="C7",
                        linewidth=0.8,
                    )
            if ih is None:
                label = f"nh = {cont.M.system.nh}"
            else:
                label = f"ih = {ih}"
            ax.plot(om_accepted, ampl_accepted, label=label, marker=".")

        if len(cont_fold) != 0:
            om_fold = [sol.x[-1] for sol in cont_fold.sol_list]
            ampl_fold = compute_amplitude(cont_fold, ih)
            ax.scatter(om_fold, ampl_fold, **fold_style)

        if len(cont_branching) != 0:
            om_branching = [sol.x[-1] for sol in cont_branching.sol_list]
            ampl_branching = compute_amplitude(cont_branching, ih)
            ax.scatter(om_branching, ampl_branching, **branching_style)

        if len(cont_unknown) != 0:
            om_unknown = [sol.x[-1] for sol in cont_unknown.sol_list]
            ampl_unknown = compute_amplitude(cont_unknown, ih)
            ax.scatter(om_unknown, ampl_unknown, **unknown_style)

        ax.set_xlabel(r"$\omega$ [rad/s]")
        ax.set_ylabel("Amplitude [m]")
        ax.legend()
        fig.show()

    return plot


def solve(nh_list: list[tuple[int, int]], chunck_list: list[dict]) -> list[Continuation]:
    cont_list = []
    for i, nh in enumerate(nh_list):
        x0 = None
        M_chunck_list = []
        hb_params = {"system": SYSTEM | {"nh": nh}}
        for chunck in chunck_list:
            M = Maestro(MODEL | hb_params | chunck)
            try:
                M.operate(x0)
            except KeyboardInterrupt:
                pass
            last_sol = get_last_solution(M.nls["cont"].SolList)
            x0 = last_sol.x[:-1]
            M_chunck_list.append(M)
        cont_list.append(
            Continuation(
                M=Maestro(MODEL | hb_params | chunck_list[0]),
                sol_list=[sol for M in M_chunck_list for sol in M.nls["cont"].SolList],
                chunk_list=chunck_list,
            )
        )
    return cont_list


def main():
    nh_list = [19]
    chunck_list = CHUNCK_LIST

    cont_list = solve(nh_list, chunck_list)

    plot_cont = continuation_plotter()
    for cont in cont_list:
        plot_cont(cont, pred=True)

    return locals()
