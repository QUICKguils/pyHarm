import pytest
import json
import numpy as np
import pandas as pd
import pyHarm
from tests.nonregression.nonregression_helperfunctions import check_residuals,chec_FRF_vs_ref, generate_arc_length

NRProb_inputfolder = "./tests/nonregression/NonRegressionData/jenkins/"

def getFRF_basic(M:pyHarm.Maestro):
    SA = [sol for sol in M.nls['FRF'].SolList if sol.flag_accepted]
    indices_selection = ("sub1",0,0)
    indexH = M.getIndex(*indices_selection)
    om = np.array([sol.x[-1] for sol in SA])
    ampH = np.array([np.linalg.norm(sol.x[indexH]) for sol in SA])
    return om,ampH

@pytest.mark.all
@pytest.mark.nonregression
def test_jenkins():
    with open(NRProb_inputfolder + "input_file.json","r") as f :
        inp = json.load(f)
    M = pyHarm.Maestro(inp)
    M.operate("null")
    check_residuals(M)
    om, amp = getFRF_basic(M)
    lamb = generate_arc_length(om, amp)
    chec_FRF_vs_ref(lamb, om, amp, NRProb_inputfolder)