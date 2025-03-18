"""LazarusModel -- Simple Diffing oscillator from Lazarus et al."""

import numpy as np

from .StepSizeMyAcceptance import StepSizeMyAcceptance

# Build the linear part of the Duffing oscillator
linsys = dict()
linsys["M"] = np.array([1])
linsys["C"] = np.array([0.05])
# linsys["C"] = np.array([0.3])  # Damping used in tuto1
linsys["K"] = np.array([1])
linsys["G"] = 0 * linsys["M"]

PROBLEM = {
    "plugin": [StepSizeMyAcceptance],
    "analysis": {},
    "system": {
        "type": "Base",
        "nh": 17,
        # NOTE: bcs of cubic nl, nti should be > 2(3*nh)+1 = 73
        "nti": 2048,
        "adim": {
            "status": False,
            "lc": 1.0,
            "wc": 1.0,
        }
    },
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
            "amp": 1,
        },
        "cubic_spring": {
            "type": "CubicSpring",
            "connect": {"duffing": [0]},
            "dirs": [0],
            "k": 1,
        },
    },
}

CONT = {
    "analysis": {
        "cont": {
            "study": "frf",
            "puls_inf": 0.01,
            "puls_start": 0.1,
            "puls_sup": 5.0,
            "ds0": 1e-3,
            "ds_min": 1e-12,
            "ds_max": 5e-3,
            "sign_ds": 1,
            "verbose": True,
            "stepsizer": "myacceptance",
            "predictor": "tangent",
            # "reductors": [
            #     {
            #         "type": "globalHarmonic",
            #         "nh_start": np.array([1]),
            #         "err_admissible": 1e10,
            #         "h_always_kept": np.array([1]),
            #         "verbose": False,
            #     },
            #     {
            #         "type": "AllgowerPreconditioner",
            #     },
            # ],
            "corrector": "arc_length",
            "stopper": "bounds",
            "solver": "scipyroot",
        },
    },
}

CONT_LIST = [CONT]
