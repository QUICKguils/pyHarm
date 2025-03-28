"""LazarusModel -- Simple Diffing oscillator from Lazarus et al."""

import numpy as np

from .StepSizeMyAcceptance import StepSizeMyAcceptance
from .PredictorMyTangent import PredictorMyTangent

# Build the linear part of the Duffing oscillator
linsys = dict()
linsys["M"] = np.array([1])
linsys["C"] = np.array([0.05])
linsys["K"] = np.array([1])
linsys["G"] = 0 * linsys["M"]

MODEL = {
    "plugin": [StepSizeMyAcceptance, PredictorMyTangent],
    "analysis": {},
    "system": {},
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

# The "nh" field need to be completed
# on the corresponding solving file.
SYSTEM = {
    "type": "Base",
    "nti": 2048,
    "adim": {
        "status": False,
        "lc": 1.0,
        "wc": 1.0,
    },
}

CHUNCK_ALL = {
    "analysis": {
        "cont": {
            "study": "frf",
            "puls_inf": 0.01,
            "puls_start": 0.1,
            # "puls_sup": 5.0,
            "puls_sup": 0.3,

            "ds0": 1e-2,
            "ds_min": 1e-8,
            "ds_max": 5e-2,

            # "ds0": 1e-1,
            # "ds_min": 1e-3,
            # "ds_max": 2e-1,

            "sign_ds": 1,
            "verbose": True,
            "stepsizer": "myacceptance",
            "predictor": "tangent",
            "corrector": "arc_length",
            "stopper": "bounds",
            "solver": "scipyroot",
        },
    },
}

CHUNCK_LIST = [CHUNCK_ALL]
