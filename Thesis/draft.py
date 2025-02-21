import matplotlib.pyplot as plt
import model
import numpy as np
from mplrc import load_rcparams

import pyHarm
from pyHarm.DynamicOperator import compute_DFT

load_rcparams()

LinearSystem = model.load_jeffcott_model("asym")

# Specification of the problem dictionary
problem = {
    "analysis": {
        # "linear": {"study": "linear_analysis"},
        "nonlinear": {
            "study": "frf",
            "puls_inf": 1.0,
            "puls_start": 1.0,
            "puls_sup": 400.0,
            "ds0": 1e-1,
            "ds_min": 1e-12,
            "ds_max": 1e-1,
            "sign_ds": 1,
            "verbose": True,
            "stepsizer": "acceptance",
            "predictor": "tangent",
            "reductors": [
                {
                    "type": "globalHarmonic",
                    "nh_start": np.array([1]),
                    "err_admissible": 1e10,
                    "h_always_kept": np.array([1]),
                    "verbose": False,
                },
            ],
            "corrector": "arc_length",
            "preconditioner": "allgower",
            "stopper": "bounds",
            "solver": "scipyroot",
        },
    },
    "system": {
        "type": "Base",
        "nh": 1,
        "nti": 1024,
        "adim": {
            "status": True,
            "lc": 1.5e-4,
            "wc": 1.0,
        },
    },
    "substructures": {
        "linear_rotor": {  # Unforced, linear rotor toy model from Safran
            "matrix": LinearSystem,
            "ndofs": 4,
        }
    },
    "connectors": {
        # Unbalance force can be modelled as m*d*Om^2*cos(Om*t)
        # See MDoT, rotordynamics 1.
        # To model a rotating unbalance, specify sinusoid unbalance
        # loading in the two transverse directions with a dephase of
        # pi/2 between them.
        "unbalance_dir0": {
            "connect": {"linear_rotor": [1]},
            "dirs": [0],
            "type": "GOForcing",
            "amp": 50e-3,
            "ho": 1,  # cos(Om*t) term
            "dto": 2,  # Om^2 term
            "phi": 0,  # Phase lag
        },
        "unbalance_dir1": {
            "connect": {"linear_rotor": [1]},
            "dirs": [1],
            "type": "GOForcing",
            "amp": 50e-3,
            "ho": 1,  # cos(Om*t) term
            "dto": 2,  # Om^2 term
            "phi": np.pi / 2,  # Phase lag
        },
        "bearing_gap": {
            "connect": {"linear_rotor": [4], "INTERNAL": [5]},
            "dirs": [0, 1],
            "type": "PenaltyBilateralGap",
            "g": 0.15e-3,
            "k": 1e8,
        },
    },
}

# Build the problem dataframe through Maestro
M = pyHarm.Maestro(problem)

# Solve
M.operate()

# Get indexes of relevant dofs and associated harmonics.
idx_51 = M.getIndex("linear_rotor", 4, 0)
idx_52 = M.getIndex("linear_rotor", 4, 1)
idx_61 = M.getIndex("linear_rotor", 5, 0)
idx_62 = M.getIndex("linear_rotor", 5, 1)

# Discrete fourier transform operator
DFTO = compute_DFT(M.system.nti, M.system.nh)

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


# Plot the NFRC
fig_nfrc, ax_nfrc = plt.subplots()
ax_nfrc.plot(omega, amplitude, label="NFRC")
ax_nfrc.scatter(
    omega_bif,
    amplitude_bif,
    marker="o",
    color="k",
    facecolors="none",
    label="Fold bifurcations",
)
ax_nfrc.set_xlabel("$\\omega [rad/s]$")
ax_nfrc.set_ylabel(r"$\|\Delta \tilde{x}(\omega)\|/gap$")
ax_nfrc.legend()
fig_nfrc.show()


# Orbital motion at the nonlinear bearing,
# at the selected (id_bif) bifurcation.
id_bif = 0
dx = (valid_sol_bif[id_bif].x[idx_51] - valid_sol_bif[id_bif].x[idx_61]) @ DFTO["ft"]
dy = (valid_sol_bif[id_bif].x[idx_52] - valid_sol_bif[id_bif].x[idx_62]) @ DFTO["ft"]
r = np.sqrt(dx**2 + dy**2)
theta = np.arctan2(dy, dx)

# Plot the orbital motion.
fig_orbit, ax_orbit = plt.subplots(subplot_kw={'projection': 'polar'})
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
