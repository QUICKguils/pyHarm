## This block makes the mandatory imports and configure matplotlib for figures
# import sys

import time

import matplotlib.pyplot as plt
import numpy as np

import pyHarm

# ------------------ SYSTEM CONSTRUCTION ------------------ #

m = 1.0  # mass
delta = 0.3  # damping
alpha = 1  # stiffness
beta = 1  # cubic term stiffness
gamma = 1.0  # forcing amplitude

MassMatrix = np.array([[m]])
DampMatrix = np.array([[delta]])
RigidMatrix = np.array([[alpha]])
LinSys = dict()
LinSys["M"] = MassMatrix
LinSys["C"] = DampMatrix
LinSys["K"] = RigidMatrix
LinSys["G"] = 0 * MassMatrix

# ------------------ INPUT CONSTRUCTION ------------------ #
INP = {
    "analysis": {
        "FRF": {
            "study": "frf",
            "puls_inf": 0.01,
            "puls_sup": 2.5,
            "ds0": 0.01,
            "ds_min": 1e-12,
            "ds_max": 0.01,
            "sign_ds": 1,
            "verbose": False,
            "stepsizer": "acceptance",
            "predictor": "tangent",
            "predictor_options": {"verbose": False},
            "corrector": "arc_length",
            "preconditioner": "nopreconditioner",
            "stopper": "bounds",
            "solver": "scipyroot",
        }
    },
    "system": {
        "type": "Base",
        "nh": 7,
        "nti": 2048,
    },
    "substructures": {"sub1": {"matrix": LinSys, "ndofs": 1}},
    "connectors": {
        "loading": {
            "type": "CosinusForcing",
            "connect": {"sub1": [0]},
            "dirs": [0],
            "amp": gamma,
        },
        "cubic": {
            "type": "CubicSpring",
            "connect": {"sub1": [0]},
            "dirs": [0],
            "k": beta,
        },
    },
}


# ------------------ COMPUTE SOLUTIONS ------------------ #

omegas = dict()
amplitudes = dict()
times = dict()
sols = dict()
list_nh = np.arange(1, 15, 2)
for nh in list_nh:
    INP["system"]["nh"] = nh
    NRB_NH = pyHarm.Maestro(INP)
    indices_selection = ("sub1", 0, 0)
    indexH = NRB_NH.getIndex(*indices_selection)

    # Solve
    start = time.time()
    NRB_NH.operate()
    elapsed_time = time.time() - start

    # Retrieve solutions
    SA = [sol for sol in NRB_NH.nls["FRF"].SolList if sol.flag_accepted]
    om = np.array([sol.x[-1] for sol in SA])

    elem = NRB_NH.nls["FRF"].system.LE_nonlinear_nodlft[0]
    SA_X = np.concatenate([sol.x.reshape(-1, 1) for sol in SA], axis=1)
    a_max = []
    for i in range(len(om)):
        X_TIME = SA_X[indexH, i] @ elem.D["ft"]
        a_max.append(np.max(X_TIME))

    # Store solutions
    sols[nh] = SA_X
    times[nh] = elapsed_time
    omegas[nh] = om
    amplitudes[nh] = a_max

# ------------------ PLOT SOLUTIONS ------------------ #

for nh in list_nh:
    plt.plot(omegas[nh], amplitudes[nh])

_ = plt.legend(["nh_" + str(i) for i in list_nh], fontsize=12)
_ = plt.xlabel("Frequency [rad/s]", fontsize=12)
_ = plt.ylabel("Max amplitude over one period [m]", fontsize=12)
_ = plt.xticks(fontsize=12)
_ = plt.yticks(fontsize=12)

plt.show()
