"""solveSafranModel -- Solve the Safran toy model."""

import matplotlib.pyplot as plt
import numpy as np

from pyHarm.Maestro import Maestro
from pyHarm.DynamicOperator import compute_DFT
from pyHarm.Predictors.ABCPredictor import BifurcationType
from pyHarm.Solver import SystemSolution

from .buildSafranModel import PROBLEM, CONT_LIST
from .mplrc import load_rcparams

M_BASE = Maestro(PROBLEM)

# Discrete fourier transform operator
DFTO = compute_DFT(M_BASE.system.nti, M_BASE.system.nh)

# Get indexes of relevant dofs and associated harmonics.
# DOF 4 and 5 in pyHarm are DOFs 5 and 6 in the Safran model schematic.
# Nonlinear bearing is located bw. these two DOFs.
idx_51 = M_BASE.getIndex("linear_rotor", 4, 0)
idx_52 = M_BASE.getIndex("linear_rotor", 4, 1)
idx_61 = M_BASE.getIndex("linear_rotor", 5, 0)
idx_62 = M_BASE.getIndex("linear_rotor", 5, 1)


def compute_amplitude(solutions: list[SystemSolution]):
    return np.array(
        [
            np.linalg.norm(
                (
                    np.sqrt(
                        ((sol.x[idx_51] - sol.x[idx_61]) @ DFTO["ft"]) ** 2
                        + ((sol.x[idx_52] - sol.x[idx_62]) @ DFTO["ft"]) ** 2
                    )
                )
                @ DFTO["tf"]
            )
            for sol in solutions
        ]
    )


def extract_solution(SolList: list[SystemSolution]):
    sol_accepted = [sol for sol in SolList if sol.flag_accepted]
    sol_rejected = [sol for sol in SolList if not sol.flag_accepted]

    return (sol_accepted, sol_rejected)


def extract_bifurcation(sol_accepted: list[SystemSolution]):
    sol_bifurcation = [sol for sol in sol_accepted if sol.flag_bifurcation]
    sol_fold = [sol for sol in sol_bifurcation if sol.bifurcation_type is BifurcationType.FOLD]
    sol_branching = [sol for sol in sol_bifurcation if sol.bifurcation_type is BifurcationType.BRANCHING]

    return (sol_bifurcation, sol_fold, sol_branching)


def plot_nfrc(sol_accepted, sol_fold, sol_branching) -> None:
    fig_nfrc, ax_nfrc = plt.subplots()

    if len(sol_accepted) != 0:
        om_accepted = [sol.x[-1] for sol in sol_accepted]
        ampl_accepted = compute_amplitude(sol_accepted)
        ax_nfrc.plot(om_accepted, ampl_accepted, label="NFRC")

    if len(sol_fold) != 0:
        om_fold = [sol.x[-1] for sol in sol_fold]
        ampl_fold = compute_amplitude(sol_fold)
        ax_nfrc.scatter(
            om_fold, ampl_fold,
            s=50, zorder=2.5, marker="x", color="C1", label="fold bifurcation",
        )

    if len(sol_branching) != 0:
        om_branching = [sol.x[-1] for sol in sol_branching]
        ampl_branching = compute_amplitude(sol_branching)
        ax_nfrc.scatter(
            om_branching, ampl_branching,
            s=50, zorder=2.5, marker="x", color="C6", label="branching point",
        )

    ax_nfrc.set_xlabel(r"$\omega$ [rad/s]")
    ax_nfrc.set_ylabel(r"$\|\Delta \tilde{x}(\omega)\|/$gap")
    ax_nfrc.legend()
    fig_nfrc.show()


def plot_orbit(SolList: list[SystemSolution], **kwargs) -> None:
    """Orbital motion at the nonlinear bearing, at the desired curve point."""
    # Check inputs
    if len(SolList) == 0:
        print("Empty list of solution: unable to plot orbit")
        return
    if not kwargs:
        print(
            "specify either an index `id` "
            "or an approximate curve point `om, ampl`"
        )
        return

    # Get id, om, ampl
    omList = [sol.x[-1] for sol in SolList]
    amplList = compute_amplitude(SolList)
    if "id" in kwargs:
        id = kwargs["id"]
    else:
        om_target = kwargs["om"]
        ampl_target = kwargs["ampl"]
        curve_sdist = np.array([  # square distance from the curve
            (om_i-om_target)**2 + (ampl_i-ampl_target)**2
            for (om_i, ampl_i) in zip(omList, amplList)
        ])
        id = np.argmin(curve_sdist)
    om = omList[id]
    ampl = amplList[id]

    # Polar repr of the orbit at the defined id.
    dx = (SolList[id].x[idx_51] - SolList[id].x[idx_61]) @ DFTO["ft"]
    dy = (SolList[id].x[idx_52] - SolList[id].x[idx_62]) @ DFTO["ft"]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    if "scale" in kwargs:
        scale = kwargs["scale"]
        # r *= (r > ampl) * scale + (r < ampl) / scale
        r += (r - ampl) * scale

    fig_orbit, ax_orbit = plt.subplots(subplot_kw={"projection": "polar"})
    ax_orbit.plot(
        np.linspace(0, 2 * np.pi, len(r)), 1 * np.ones_like(r),
        color="C7", linewidth=1, linestyle="--", label="Gap",
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


def main() -> Maestro:
    load_rcparams()

    # This try-except acts as an interactive StopCriterion
    # FIX: pyHarm_plugin print warning each time a Maestro is created
    try:
        M_LIST = [Maestro(PROBLEM | CONT) for CONT in CONT_LIST]
        x0 = None
        for M in M_LIST:
            M.operate(x0)
            x0 = M.nls["cont"].SolList[-1].x[:-1]  # TODO: take the last *valid* solution
    except KeyboardInterrupt:
        pass

    SolList = [sol for M in M_LIST for sol in M.nls["cont"].SolList]

    (sol_accepted, sol_rejected) = extract_solution(SolList)
    (sol_bifurcation, sol_fold, sol_branching) = extract_bifurcation(sol_accepted)

    plot_nfrc(sol_accepted, sol_fold, sol_branching)

    return locals()


if __name__ == "__main__":
    main()
