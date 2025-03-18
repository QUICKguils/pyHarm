"""solveLazarusModel -- Solve the simple Duffing oscillator from Lazarus et al."""

import matplotlib.pyplot as plt
import numpy as np

from pyHarm.Maestro import Maestro
from pyHarm.DynamicOperator import compute_DFT
from pyHarm.Predictors.ABCPredictor import BifurcationType
from pyHarm.Solver import SystemSolution

from .LazarusModel import PROBLEM, CONT_LIST
from .mplrc import load_rcparams

M_BASE = Maestro(PROBLEM)

# Discrete fourier transform operator
DFTO = compute_DFT(M_BASE.system.nti, M_BASE.system.nh)

# The way pyHarm sort the solution can be tricky for complex system,
# so we better have to use the getIndex util function.
idx = M_BASE.getIndex("duffing", 0, 0)  # DOF 0, dir 0 of subsystem "duffing"


def compute_amplitude(SolList: list[SystemSolution]):
    return np.array([np.max(sol.x[idx] @ DFTO["ft"])for sol in SolList])


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
        ampl_accepted = compute_amplitude(sol_accepted)
        om_accepted = [sol.x[-1] for sol in sol_accepted]
        ax_nfrc.plot(om_accepted, ampl_accepted, label="NFRC")

    if len(sol_fold) != 0:
        ampl_fold = compute_amplitude(sol_fold)
        om_fold = [sol.x[-1] for sol in sol_fold]
        ax_nfrc.scatter(
            om_fold, ampl_fold,
            s=50, zorder=2.5, marker="x", color="C1", label="fold bifurcation",
        )

    if len(sol_branching) != 0:
        ampl_branching = compute_amplitude(sol_branching)
        om_branching = [sol.x[-1] for sol in sol_branching]
        ax_nfrc.scatter(
            om_branching, ampl_branching,
            s=50, zorder=2.5, marker="x", color="C6", label="branching point",
        )

    ax_nfrc.set_xlabel(r"$\omega$ [rad/s]")
    ax_nfrc.set_ylabel(r"max($x$)/m")
    ax_nfrc.legend()
    fig_nfrc.show()


def main():
    load_rcparams()

    # This try-except acts as an interactive StopCriterion
    # FIX: pyHarm_plugin print warning each time a Maestro is created
    try:
        M_LIST = [Maestro(PROBLEM | CONT) for CONT in CONT_LIST]
        # x0 = np.zeros(25)
        # x0[1] = 1  # 1s = 1
        # x0[2] = 1  # 1c = 1
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
