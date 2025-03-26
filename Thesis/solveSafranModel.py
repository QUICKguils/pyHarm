"""solveSafranModel -- Solve the Safran toy model."""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from pyHarm.DynamicOperator import compute_DFT
from pyHarm.Maestro import Maestro
from pyHarm.Predictors.ABCPredictor import BifurcationType
from pyHarm.Solver import SystemSolution

from .mplrc import load_rcparams
from .SafranModel import CHUNCK_LIST, MODEL, SYSTEM


class Continuation(NamedTuple):
    M: Maestro  # Maestro object created for the first continuation chunck.
    sol_list: list[SystemSolution]  # Gathered SolList of all the continuation chuncks.
    chunk_list: list[dict]  # "analysis" keys of all the continuation chuncks.


def _get_dof_idx(M: Maestro):
    """Get SystemSolution indexes of the required dofs.

    That is, the DOFs located at the gap bearing.
    DOFs 4 and 5 in pyHarm are DOFs 5 and 6 in the Safran model schematic.
    Nonlinear bearing is located between these two DOFs.
    """
    idx_51 = M.getIndex("linear_rotor", 4, 0)  # shaft node, X-dir
    idx_52 = M.getIndex("linear_rotor", 4, 1)  # shaft node, Y-dir
    idx_61 = M.getIndex("linear_rotor", 5, 0)  # bearing node, X-dir
    idx_62 = M.getIndex("linear_rotor", 5, 1)  # bearing node, Y-dir

    return idx_51, idx_52, idx_61, idx_62


def _filter_harm(idx: np.ndarray[int], ih: list[int]) -> np.ndarray[int]:
    """Filter a SystemSolution index vector, to select only the desired harmonics."""
    expl_ih = np.array([[2 * i - 1, 2 * i] for i in ih], dtype=int).ravel()
    expl_ih = expl_ih[expl_ih >= 0]  # remove the -1 generated for potential 0-harm

    filtered = np.zeros(idx.shape, dtype=int)
    filtered[expl_ih] = idx[expl_ih]

    return filtered


def _filter_harm_tuple(indexes: tuple, ih: list[int]) -> tuple:
    return tuple((_filter_harm(idx, ih) for idx in indexes))


def compute_amplitude(cont: Continuation, ih=None, pred=False) -> np.ndarray[float]:
    """Compute gap amplitude at the bearing, for specified harmonics."""
    # Discrete fourier transform operator
    DFTO = compute_DFT(cont.M.system.nti, cont.M.system.nh)
    # Get indexes of relevant dofs and associated harmonics.
    indexes = _get_dof_idx(cont.M)

    if ih is None:
        filt_51, filt_52, filt_61, filt_62 = indexes
    else:
        filt_51, filt_52, filt_61, filt_62 = _filter_harm_tuple(indexes, ih)

    def select_pred(sol: SystemSolution):
        return sol.x_pred if pred else sol.x

    return np.array(
        [
            np.linalg.norm(
                (
                    np.sqrt(
                        ((select_pred(sol)[filt_51] - select_pred(sol)[filt_61]) @ DFTO["ft"]) ** 2
                        + ((select_pred(sol)[filt_52] - select_pred(sol)[filt_62]) @ DFTO["ft"]) ** 2
                    )
                )
                @ DFTO["tf"]
            )
            for sol in cont.sol_list
        ]
    )


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
            sol
            for sol in cont_bifurcation.sol_list
            if sol.bifurcation_type is BifurcationType.BRANCHING
        ],
        chunk_list=cont.chunk_list,
    )

    return cont_bifurcation, cont_fold, cont_branching


def continuation_plotter():
    fig, ax = plt.subplots()
    fold_style = {"color": "C1", "s": 50, "zorder": 2.5, "marker": "x"}
    branching_style = {"color": "C6", "s": 50, "zorder": 2.5, "marker": "x"}
    ax.scatter([], [], **fold_style, label="fold bifurcation")
    ax.scatter([], [], **branching_style, label="branching point")

    def plot(cont: Continuation, ih=None, pred=False):
        cont_accepted, cont_rejected = extract_accepted(cont)
        cont_bifurcation, cont_fold, cont_branching = extract_bifurcation(cont_accepted)

        if len(cont_accepted) != 0:
            om_accepted = [sol.x[-1] for sol in cont_accepted.sol_list]
            ampl_accepted = compute_amplitude(cont_accepted, ih)
            if pred:
                pred_style = {"color": "C2", "zorder": 2.5, "marker": "."}
                om_pred_accepted = [sol.x_pred[-1] for sol in cont_accepted.sol_list]
                ampl_pred_accepted = compute_amplitude(cont_accepted, ih, pred)
                ax.scatter(om_pred_accepted, ampl_pred_accepted, **pred_style, label="prediction")
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

        ax.set_xlabel(r"$\omega$ [rad/s]")
        ax.set_ylabel(r"$\|\Delta x\|/g$")
        ax.legend()
        fig.show()

    return plot


def plot_orbit(cont: Continuation, **kwargs) -> None:
    """Orbital motion at the nonlinear bearing, at the desired curve point.

    Keyword args:
        id (int) or om, ampl (float, float):
          Either the index (id) of the solution in the provided list,
          or a point (om, ampl) at which the nearest solution on the curve will be used.
        scale (float):
          Optional scaling factor used to exacerbate the radius values.
    """
    # Discrete fourier transform operator
    DFTO = compute_DFT(cont.M.system.nti, cont.M.system.nh)
    # Get indexes of relevant dofs and associated harmonics.
    idx_51, idx_52, idx_61, idx_62 = _get_dof_idx(cont.M)

    # Check inputs
    if len(cont.sol_list) == 0:
        print("Empty list of solution: unable to plot orbit")
        return
    if not kwargs:
        print("specify either an index `id` or an approximate curve point `om, ampl`")
        return

    # Get id, om, ampl
    om_list = [sol.x[-1] for sol in cont.sol_list]
    ampl_list = compute_amplitude(cont)
    if "id" in kwargs:
        id = kwargs["id"]
    else:
        om_target = kwargs["om"]
        ampl_target = kwargs["ampl"]
        curve_sdist = np.array(  # square distance from the curve
            [
                (om_i - om_target) ** 2 + (ampl_i - ampl_target) ** 2
                for (om_i, ampl_i) in zip(om_list, ampl_list)
            ]
        )
        id = np.argmin(curve_sdist)
    om = om_list[id]
    ampl = ampl_list[id]

    # Polar repr of the orbit at the defined id.
    dx = (cont.sol_list[id].x[idx_51] - cont.sol_list[id].x[idx_61]) @ DFTO["ft"]
    dy = (cont.sol_list[id].x[idx_52] - cont.sol_list[id].x[idx_62]) @ DFTO["ft"]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    if "scale" in kwargs:
        scale = kwargs["scale"]
        mean = np.mean(r)
        r_scaled = mean + (r - mean) * scale

    fig_orbit, ax_orbit = plt.subplots(subplot_kw={"projection": "polar"})
    ax_orbit.plot(
        np.linspace(0, 2 * np.pi, len(r)), 1 * np.ones_like(r),
        color="C7", linewidth=1, linestyle="--", label="Gap",
    )
    ax_orbit.plot(
        np.linspace(0, 2 * np.pi, len(r)), mean * np.ones_like(r),
        color="C2", linewidth=2, label="Mean",
    )
    ax_orbit.plot(
        theta, r_scaled,
        color="C0", linewidth=2, label="Rotor motion",
    )
    ax_orbit.scatter(
        theta[r > 1], r_scaled[r > 1],
        s=15, marker="o", zorder=2.5, color="C6", label="displacement > gap",
    )
    ax_orbit.set_title(f"Freq: {om:.4f} Hz, Mean ampl: {ampl:.4f}")
    ax_orbit.set_xticklabels([])
    ax_orbit.set_yticklabels([])
    ax_orbit.spines["polar"].set_visible(False)
    ax_orbit.legend(fontsize=10, loc=10)
    fig_orbit.show()


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
            x0 = M.nls["cont"].SolList[-1].x[:-1]  # TODO: take the last *valid* solution
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

    nh_list = [1]
    chunck_list = CHUNCK_LIST

    cont_list = solve(nh_list, chunck_list)

    plot_cont = continuation_plotter()
    for cont in cont_list:
        plot_cont(cont)

    return locals()
