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


def _get_gap_idx(M: Maestro):
    """Get indexes of relevant dofs and associated harmonics.

    DOF 4 and 5 in pyHarm are DOFs 5 and 6 in the Safran model schematic.
    Nonlinear bearing is located between these two DOFs.
    """
    idx_51 = M.getIndex("linear_rotor", 4, 0)
    idx_52 = M.getIndex("linear_rotor", 4, 1)
    idx_61 = M.getIndex("linear_rotor", 5, 0)
    idx_62 = M.getIndex("linear_rotor", 5, 1)

    return idx_51, idx_52, idx_61, idx_62


def compute_amplitude(cont: Continuation) -> np.ndarray[float]:
    # Discrete fourier transform operator
    DFTO = compute_DFT(cont.M.system.nti, cont.M.system.nh)
    # Get indexes of relevant dofs and associated harmonics.
    idx_51, idx_52, idx_61, idx_62 = _get_gap_idx(cont.M)

    return np.array([
        np.linalg.norm(
            (np.sqrt(
                ((sol.x[idx_51] - sol.x[idx_61]) @ DFTO["ft"]) ** 2
                + ((sol.x[idx_52] - sol.x[idx_62]) @ DFTO["ft"]) ** 2
            ))
            @ DFTO["tf"]
        )
        for sol in cont.sol_list
    ])


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


def nfrc_plotter():
    fig_nfrc, ax_nfrc = plt.subplots()

    def plot(cont: Continuation):
        cont_accepted, cont_rejected = extract_accepted(cont)
        cont_bifurcation, cont_fold, cont_branching = extract_bifurcation(cont_accepted)

        if len(cont_accepted) != 0:
            om_accepted = [sol.x[-1] for sol in cont_accepted.sol_list]
            ampl_accepted = compute_amplitude(cont_accepted)
            ax_nfrc.plot(om_accepted, ampl_accepted, label="NFRC")

        if len(cont_fold) != 0:
            om_fold = [sol.x[-1] for sol in cont_fold.sol_list]
            ampl_fold = compute_amplitude(cont_fold)
            ax_nfrc.scatter(
                om_fold, ampl_fold,
                s=50, zorder=2.5, marker="x", color="C1", label="fold bifurcation",
            )

        if len(cont_branching) != 0:
            om_branching = [sol.x[-1] for sol in cont_branching.sol_list]
            ampl_branching = compute_amplitude(cont_branching)
            ax_nfrc.scatter(
                om_branching, ampl_branching,
                s=50, zorder=2.5, marker="x", color="C6", label="branching point",
            )

        ax_nfrc.set_xlabel(r"$\omega$ [rad/s]")
        ax_nfrc.set_ylabel(r"$\|\Delta \tilde{x}(\omega)\|/$gap")
        ax_nfrc.legend()
        fig_nfrc.show()

    return plot


def plot_orbit(cont: Continuation, **kwargs) -> None:
    """Orbital motion at the nonlinear bearing, at the desired curve point.

    Keyword args:
        id (int) or om, ampl (float, float):
          Either the index (id) of the solution in the provided list,
          or a point (om, ampl) at which the nearest solution on the curve
          will be used.
        scale (float):
          Optional scaling factor used to exacerbate the radius values.
    """
    # Discrete fourier transform operator
    DFTO = compute_DFT(cont.M.system.nti, cont.M.system.nh)
    # Get indexes of relevant dofs and associated harmonics.
    idx_51, idx_52, idx_61, idx_62 = _get_gap_idx(cont.M)

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
        r = mean + (r - mean) * scale

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
        theta, r,
        color="C0", linewidth=2, label="Rotor motion",
    )
    ax_orbit.scatter(
        theta[r > 1], r[r > 1],
        s=15, marker="o", zorder=2.5, color="C6", label="displacement > gap",
    )
    ax_orbit.set_title(f"Freq: {om:.4f} Hz, Mean ampl: {ampl:.4f}")
    ax_orbit.set_xticklabels([])
    ax_orbit.set_yticklabels([])
    ax_orbit.spines["polar"].set_visible(False)
    ax_orbit.legend(fontsize=10, loc=10)
    fig_orbit.show()


def solve(hb_list: list[tuple[int, int]], chunck_list: list[dict]) -> list[Continuation]:
    cont_list = []
    for i, (nh, nti) in enumerate(hb_list):
        x0 = None
        M_chunck_list = []
        hb_params = {"system": SYSTEM | {"nh": nh, "nti": nti}}
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

    hb_list = [(0, 1024)]
    # hb_list = [(1, 1024), (2, 1024), (3, 1024), (4, 1024), (5, 1024)]
    # hb_list = [(3, 1024), (5, 1024), (12, 1024)]
    chunck_list = CHUNCK_LIST

    # This try-except acts as an interactive StopCriterion on CTRL-C
    # FIX: pyHarm_plugin print warning each time a Maestro is created
    try:
        cont_list = solve(hb_list, chunck_list)
    except KeyboardInterrupt:
        pass

    plot_nfrc = nfrc_plotter()
    for cont in cont_list:
        plot_nfrc(cont)

    return locals()
