"""solveSafranModel -- Solve the Safran toy model."""

import matplotlib.pyplot as plt
import numpy as np

from pyHarm.Maestro import Maestro
from pyHarm.DynamicOperator import compute_DFT
from pyHarm.Predictors.ABCPredictor import BifurcationType
from pyHarm.Solver import SystemSolution

from .buildSafranModel import M
from .mplrc import load_rcparams

# Discrete fourier transform operator
DFTO = compute_DFT(M.system.nti, M.system.nh)

# Get indexes of relevant dofs and associated harmonics.
# DOF 4 and 5 in pyHarm are DOFs 5 and 6 in the Safran model schematic.
# Nonlinear bearing is located bw. these two DOFs.
idx_51 = M.getIndex("linear_rotor", 4, 0)
idx_52 = M.getIndex("linear_rotor", 4, 1)
idx_61 = M.getIndex("linear_rotor", 5, 0)
idx_62 = M.getIndex("linear_rotor", 5, 1)


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


def extract_solution(M: Maestro):
    solutions = [sol for sol in M.nls["nonlinear"].SolList]
    sol_accepted = [sol for sol in solutions if sol.flag_accepted]
    sol_rejected = [sol for sol in solutions if not sol.flag_accepted]

    return (sol_accepted, sol_rejected)


def extract_bifurcation(sol_accepted: list[SystemSolution]):
    sol_bifurcation = [sol for sol in sol_accepted if sol.flag_bifurcation]
    sol_fold = [sol for sol in sol_bifurcation if sol.bifurcation_type is BifurcationType.FOLD]
    sol_branching = [sol for sol in sol_bifurcation if sol.bifurcation_type is BifurcationType.BRANCHING]

    return (sol_bifurcation, sol_fold, sol_branching)


def plot_nfrc(sol_accepted, sol_fold, sol_branching) -> None:
    ampl_accepted = compute_amplitude(sol_accepted)
    ampl_fold = compute_amplitude(sol_fold)
    ampl_branching = compute_amplitude(sol_branching)
    om_accepted = [sol.x[-1] for sol in sol_accepted]
    om_fold = [sol.x[-1] for sol in sol_fold]
    om_branching = [sol.x[-1] for sol in sol_branching]

    fig_nfrc, ax_nfrc = plt.subplots()
    ax_nfrc.plot(om_accepted, ampl_accepted, label="NFRC")
    ax_nfrc.scatter(
        om_fold, ampl_fold,
        s=50, zorder=2.5, marker="x", color="C1", label="fold bifurcation",
    )
    ax_nfrc.scatter(
        om_branching, ampl_branching,
        s=50, zorder=2.5, marker="x", color="C6", label="branching point",
    )
    ax_nfrc.set_xlabel("$\\omega [rad/s]$")
    ax_nfrc.set_ylabel(r"$\|\Delta \tilde{x}(\omega)\|/gap$")
    ax_nfrc.legend()
    fig_nfrc.show()


def plot_orbit(sol_bifurcation, id=0) -> None:
    """Orbital motion at the nonlinear bearing, for the selected `id` bifurcation."""
    dx = (sol_bifurcation[id].x[idx_51] - sol_bifurcation[id].x[idx_61]) @ DFTO["ft"]
    dy = (sol_bifurcation[id].x[idx_52] - sol_bifurcation[id].x[idx_62]) @ DFTO["ft"]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    fig_orbit, ax_orbit = plt.subplots(subplot_kw={"projection": "polar"})
    ax_orbit.plot(
        theta, r,
        color="C0", linewidth=0.8, label="Rotor motion",
    )
    ax_orbit.plot(
        np.linspace(0, 2 * np.pi, len(r)), 1 * np.ones_like(r),
        color="C7", linewidth=0.5, linestyle="--", label="Gap",
    )
    ax_orbit.scatter(
        theta[r > 1], r[r > 1],
        s=10, marker="o", zorder=2.5, color="C6", label="displacement > gap",
    )
    ax_orbit.set_xticklabels([])
    ax_orbit.set_yticklabels([])
    ax_orbit.spines["polar"].set_visible(False)
    ax_orbit.legend(fontsize=10, loc=10)
    fig_orbit.show()


def main() -> Maestro:
    load_rcparams()

    try:
        M.operate()
    except KeyboardInterrupt:
        return M

    (sol_accepted, sol_rejected) = extract_solution(M)
    (sol_bifurcation, sol_fold, sol_branching) = extract_bifurcation(sol_accepted)

    plot_nfrc(sol_accepted, sol_fold, sol_branching)
    plot_orbit(sol_bifurcation, id=0)

    return M


if __name__ == "__main__":
    M = main()
