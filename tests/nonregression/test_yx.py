import json

import numpy as np
import pandas as pd
import pytest

import pyHarm
from tests.nonregression.nonregression_helperfunctions import (
    chec_FRF_vs_ref,
    check_residuals,
    generate_arc_length,
)

NRProb_inputfolder = "./tests/nonregression/NonRegressionData/yx/"


def getFRF_yx(M: pyHarm.Maestro):
    SA = [sol for sol in M.nls["FRF"].SolList if sol.flag_accepted]
    indices_selection_51 = ("sub1", 4, 0)
    indices_selection_52 = ("sub1", 4, 1)
    indices_selection_61 = ("sub1", 5, 0)
    indices_selection_62 = ("sub1", 5, 1)
    indexH_51 = M.getIndex(*indices_selection_51)
    indexH_52 = M.getIndex(*indices_selection_52)
    indexH_61 = M.getIndex(*indices_selection_61)
    indexH_62 = M.getIndex(*indices_selection_62)
    from pyHarm.DynamicOperator import compute_DFT

    DFTO = compute_DFT(M.system.nti, M.system.nh)
    om = np.array([sol.x[-1] for sol in SA])
    ampH = np.array(
        [
            np.linalg.norm(
                (
                    np.sqrt(
                        ((sol.x[indexH_51] - sol.x[indexH_61]) @ DFTO["ft"]) ** 2
                        + ((sol.x[indexH_52] - sol.x[indexH_62]) @ DFTO["ft"]) ** 2
                    )
                )
                @ DFTO["tf"]
            )
            for sol in SA
        ]
    )
    return om, ampH


@pytest.mark.all
@pytest.mark.nonregression
def test_yx():
    with open(NRProb_inputfolder + "input_file.json", "r") as f:
        inp = json.load(f)
    M = pyHarm.Maestro(inp)
    M.operate("null")
    check_residuals(M)
    om, amp = getFRF_yx(M)
    lamb = generate_arc_length(om, amp)
    chec_FRF_vs_ref(lamb, om, amp, NRProb_inputfolder)
