"""solveSafranModel -- Solve the Safran toy model."""

import matplotlib.pyplot as plt
import numpy as np
from buildSafranModel import M
from mplrc import load_rcparams
from pyHarm import Maestro
from pyHarm.DynamicOperator import compute_DFT

load_rcparams()

# Discrete fourier transform operator
DFTO = compute_DFT(M.system.nti, M.system.nh)

# Get indexes of relevant dofs and associated harmonics.
# DOF 4 and 5 in pyHarm are DOFs 5 and 6 in the Safran model schematic.
# Nonlinear bearing is located bw. these two DOFs.
idx_51 = M.getIndex("linear_rotor", 4, 0)
idx_52 = M.getIndex("linear_rotor", 4, 1)
idx_61 = M.getIndex("linear_rotor", 5, 0)
idx_62 = M.getIndex("linear_rotor", 5, 1)


def compute_nfrc(M: Maestro):
    # Extract relevant parts of the computed solution
    valid_sol = [sol for sol in M.nls["nonlinear"].SolList if sol.flag_accepted]
    omega = np.array([sol.x[-1] for sol in valid_sol])
    amplitude = np.array(
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
            for sol in valid_sol
        ]
    )

    return (valid_sol, omega, amplitude)


def compute_bifurcation(valid_sol):
    # Extract bifurcation solutions
    valid_sol_bif = [sol for sol in valid_sol if sol.flag_bifurcation]
    omega_bif = np.array([sol.x[-1] for sol in valid_sol_bif])
    amplitude_bif = np.array(
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
            for sol in valid_sol_bif
        ]
    )

    return (valid_sol_bif, omega_bif, amplitude_bif)


def plot_nfrc_bif(omega, amplitude, omega_bif, amplitude_bif) -> None:
    # Plot the NFRC
    fig_nfrc, ax_nfrc = plt.subplots()
    ax_nfrc.plot(omega, amplitude, label="NFRC")
    ax_nfrc.scatter(
        omega_bif,
        amplitude_bif,
        marker="o",
        color="k",
        facecolors="none",
        label="bifurcation detected",
    )
    ax_nfrc.set_xlabel("$\\omega [rad/s]$")
    ax_nfrc.set_ylabel(r"$\|\Delta \tilde{x}(\omega)\|/gap$")
    ax_nfrc.legend()
    fig_nfrc.show()


def plot_orbits(valid_sol_bif) -> None:
    # Orbital motion at the nonlinear bearing,
    # at the selected (id_bif) bifurcation.
    id_bif = 0
    dx = (valid_sol_bif[id_bif].x[idx_51] - valid_sol_bif[id_bif].x[idx_61]) @ DFTO[
        "ft"
    ]
    dy = (valid_sol_bif[id_bif].x[idx_52] - valid_sol_bif[id_bif].x[idx_62]) @ DFTO[
        "ft"
    ]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Plot the orbital motion.
    fig_orbit, ax_orbit = plt.subplots(subplot_kw={"projection": "polar"})
    ax_orbit.plot(theta, r, label="Rotor motion")
    ax_orbit.plot(
        np.linspace(0, 2 * np.pi, len(r)), 1 * np.ones_like(r), "k--", label="Gap"
    )
    ax_orbit.plot(theta[r > 1], r[r > 1], "ro", label="displacement > gap")
    ax_orbit.set_xticklabels([])
    ax_orbit.set_yticklabels([])
    ax_orbit.spines["polar"].set_visible(False)
    ax_orbit.legend(fontsize=10, loc=10)
    fig_orbit.show()


if __name__ == "__main__":
    M.operate()
    (valid_sol, omega, amplitude) = compute_nfrc(M)
    (valid_sol_bif, omega_bif, amplitude_bif) = compute_bifurcation(valid_sol)

    plot_nfrc_bif(omega, amplitude, omega_bif, amplitude_bif)
    plot_orbits(valid_sol_bif)
