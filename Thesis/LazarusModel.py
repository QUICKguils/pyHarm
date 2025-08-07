"""LazarusModel -- Simple Diffing oscillator from Lazarus et al."""

import numpy as np

from .StepSizeMyAcceptance import StepSizeMyAcceptance
from .PredictorMyTangent import PredictorMyTangent
from .SolverMyNewtonRaphson import SolverMyNewtonRaphson

# Build the linear part of the Duffing oscillator
linsys = dict()
linsys["M"] = np.array([1])
linsys["C"] = np.array([0.05])
linsys["K"] = np.array([1])
linsys["G"] = 0 * linsys["M"]

MODEL = {
    "plugin": [StepSizeMyAcceptance, PredictorMyTangent, SolverMyNewtonRaphson],
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

            # # Quick test on first peak
            # "puls_inf": 0.01,
            # "puls_start": 0.1,
            # "puls_sup": 0.3,
            # "ds0": 1e-2,
            # "ds_min": 1e-8,
            # "ds_max": 5e-2,

            # # Coarse
            # "puls_inf": 0.01,
            # "puls_start": 0.1,
            # "puls_sup": 5.0,
            # "ds0": 5e-3,
            # "ds_min": 1e-8,
            # "ds_max": 5e-2,

            # # Less coarse (capture branch bifurcation)
            # "puls_inf": 0.01,
            # "puls_start": 0.1,
            # "puls_sup": 5.0,
            # "ds0": 1e-2,
            # "ds_min": 1e-8,
            # "ds_max": 2e-2,

            # Quite fine
            "puls_inf": 0.01,
            "puls_start": 0.1,
            "puls_sup": 5.0,
            "ds0": 5e-3,
            "ds_min": 1e-8,
            "ds_max": 5e-3,

            "sign_ds": 1,
            "verbose": True,
            "stepsizer": "acceptance",
            "predictor": "tangent",
            "corrector": "arc_length",
            "bifurcator": None,
            # "bifurcator": "perturbation",
            # "bifurcator_options": {"blind_spot: 30"},
            "stopper": "bounds",
            "solver": "MyNewtonRaphson",
            "solver_options": {"max_iter": 4, "pert": 1E-4},
        },
    },
}

CHUNCK_LIST = [CHUNCK_ALL]
