"""model -- Build the Safran toy model."""

import pathlib

from scipy import io

MODEL_PATH = pathlib.Path(__file__).parent / "model"


def load_jeffcott_model(type: str):
    if type == "sym":
        jeffcott_data = io.loadmat(str(MODEL_PATH / "jeffcott"))
    elif type == "asym":
        jeffcott_data = io.loadmat(str(MODEL_PATH / "jeffcott_asym"))
    else:
        raise ValueError("type must be 'sym' or 'asym'")

    LinSys = dict()
    LinSys["M"] = jeffcott_data["M"]
    LinSys["C"] = jeffcott_data["C"]
    LinSys["K"] = jeffcott_data["K"]
    LinSys["G"] = jeffcott_data["G"]

    return LinSys
