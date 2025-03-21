"""solveLazarusModel -- Solve the simple Duffing oscillator from Lazarus et al."""

from typing import NamedTuple

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from pyHarm.DynamicOperator import compute_DFT
from pyHarm.Maestro import Maestro
from pyHarm.Predictors.ABCPredictor import BifurcationType
from pyHarm.Solver import SystemSolution

from .mplrc import load_rcparams
from .LazarusModel import CHUNCK_LIST, MODEL, SYSTEM


class Continuation(NamedTuple):
    M: Maestro  # Maestro object created for the first continuation chunck.
    sol_list: list[SystemSolution]  # Gathered SolList of all the continuation chuncks.
    chunk_list: list[dict]  # "analysis" keys of all the continuation chuncks.


def _get_dof(M: Maestro):
    """Get indexes of relevant dofs and associated harmonics."""
    return M.getIndex("duffing", 0, 0)  # DOF 0, dir 0 of subsystem "duffing"


def _filter_harm(dof: np.ndarray[int], ih: list[int]) -> np.ndarray[int]:
    expl_ih = np.array([], dtype=int)
    for i in ih:
        expl_ih = np.append(expl_ih, [(2*i-1), 2*i])
    expl_ih = expl_ih[expl_ih>=0]  # filter -1 generated for 0-harm

    filtered = np.zeros(dof.shape, dtype=int)
    for i in expl_ih:
        filtered[i] = dof[i]

    return filtered


def compute_amplitude(cont: Continuation, ih=None) -> np.ndarray[float]:
    """Compute max. amplitude of given continuation, for specified harmonics."""
    DFTO = compute_DFT(cont.M.system.nti, cont.M.system.nh)
    dof = _get_dof(cont.M)

    if ih is None:
        filtered = dof
    else:
        filtered = _filter_harm(dof, ih)

    return np.array([np.max(sol.x[filtered] @ DFTO["ft"]) for sol in cont.sol_list])


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
        sol_list=[sol for sol in cont.sol_list if sol.flag_bifurcation],
        chunk_list=cont.chunk_list,
    )
    cont_fold = Continuation(
        M=cont.M,
        sol_list=[
            sol for sol in cont_bifurcation.sol_list
            if sol.bifurcation_type is BifurcationType.FOLD
        ],
        chunk_list=cont.chunk_list,
    )
    cont_branching = Continuation(
        M=cont.M,
        sol_list=[
            sol for sol in cont_bifurcation.sol_list
            if sol.bifurcation_type is BifurcationType.BRANCHING
        ],
        chunk_list=cont.chunk_list,
    )

    return cont_bifurcation, cont_fold, cont_branching


def continuation_plotter():
    fig, ax = plt.subplots()
    fold_style      = {"color": "C1", "s": 50, "zorder": 2.5, "marker": "x"}
    branching_style = {"color": "C6", "s": 50, "zorder": 2.5, "marker": "x"}
    ax.scatter([], [], **fold_style, label="fold bifurcation")
    ax.scatter([], [], **branching_style, label="branching point")

    def plot(cont: Continuation, ih=None):
        cont_accepted, cont_rejected = extract_accepted(cont)
        cont_bifurcation, cont_fold, cont_branching = extract_bifurcation(cont_accepted)

        if len(cont_accepted) != 0:
            om_accepted = [sol.x[-1] for sol in cont_accepted.sol_list]
            ampl_accepted = compute_amplitude(cont_accepted, ih)
            if ih is None:
                label = f"nh = {cont.M.system.nh}"
            else:
                label = f"ih = {ih}"
            ax.plot(om_accepted, ampl_accepted, label=label)

        if len(cont_fold) != 0:
            om_fold = [sol.x[-1] for sol in cont_fold.sol_list]
            ampl_fold = compute_amplitude(cont_fold, ih)
            ax.scatter(om_fold, ampl_fold, **fold_style)

        if len(cont_branching) != 0:
            om_branching = [sol.x[-1] for sol in cont_branching.sol_list]
            ampl_branching = compute_amplitude(cont_branching, ih)
            ax.scatter(om_branching, ampl_branching, **branching_style)

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
            M.operate(x0)
            x0 = (
                M.nls["cont"].SolList[-1].x[:-1]
            )  # TODO: take the last *valid* solution
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
    load_rcparams()

    nh_list = [17]
    chunck_list = CHUNCK_LIST

    cont_list = solve(nh_list, chunck_list)

    plot_cont = continuation_plotter()
    for cont in cont_list:
        plot_cont(cont)

    return locals()
