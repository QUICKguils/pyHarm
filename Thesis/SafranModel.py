"""buildSafranModel -- Build the Safran toy model."""

import pathlib

import numpy as np

from .StepSizeMyAcceptance import StepSizeMyAcceptance

MODEL_PATH = pathlib.Path(__file__).parent / "res"

# Problem specification as a phHarm dictionary
# NOTE: see SafranModel.py for the original problem specs.
MODEL = {
    "plugin": [StepSizeMyAcceptance],
    "analysis": {},
    "system": {},
    "substructures": {
        "linear_rotor": {  # Unforced, linear rotor toy model from Safran
            "filename": str(MODEL_PATH / "Jeffcott_asym.mat"),
            "ndofs": 4,
        },
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

SYSTEM = {
    "type": "Base",
    "nh": 1,
    "nti": 1024,
    "adim": {
        "status": True,
        "lc": 0.15e-3,  # Adim by the gap clearance
        "wc": 1.0,
    },
}

# Chained continuations
# CHUNCK_1 -> CHUNCK_2 -> CHUNCK_3    -> CHUNCK_4        -> CHUNCK_5  => CONT_LIST
# 1-70     -> 70-78    -> 78-loop-150 -> 150-s_shape-150 -> 150-300

CHUNCK_1 = {
    "analysis": {
        "cont": {
            "study": "frf",
            "puls_inf": 1.0,
            "puls_start": 1.0,
            "puls_sup": 72.0,
            "ds0": 1e-1,
            "ds_min": 1e-12,
            "ds_max": 3e-1,
            "sign_ds": 1,
            "verbose": True,
            "stepsizer": "myacceptance",
            "predictor": "tangent",
            "corrector": "arc_length",
            "stopper": "bounds",
            "solver": "scipyroot",
            # "solver": "NewtonRaphson",
        },
    },
}

CHUNCK_2 = {
    "analysis": {
        "cont": {
            "study": "frf",
            "puls_inf": 70.0,
            "puls_start": 72.0,
            "puls_sup": 76.0,
            "ds0": 5e-3,
            "ds_min": 1e-12,
            "ds_max": 1e-2,
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

CHUNCK_3 = {
    "analysis": {
        "cont": {
            "study": "frf",
            "puls_inf": 76.0,
            "puls_start": 76.0,
            "puls_sup": 150.0,
            "ds0": 1e-1,
            "ds_min": 1e-12,
            "ds_max": 3e-1,
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

CHUNCK_3 = {
    "analysis": {
        "cont": {
            "study": "frf",
            "puls_inf": 76.0,
            "puls_start": 76.0,
            "puls_sup": 150.0,
            "ds0": 1e-1,
            "ds_min": 1e-12,
            "ds_max": 3e-1,
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

CHUNCK_LIST = [CHUNCK_1, CHUNCK_2]
