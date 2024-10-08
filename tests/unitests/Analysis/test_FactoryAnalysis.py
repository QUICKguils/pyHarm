import pandas as pd
from pyHarm.Analysis.ABCAnalysis import ABCAnalysis
from pyHarm.Analysis.FactoryNonLinearStudy import NonLinearStudy_kind, generateNonLinearAnalysis
import numpy as np
import pytest


@pytest.mark.all
@pytest.mark.unit
def test_NonLinearStudyKind():
    for k, v in NonLinearStudy_kind.items():
        assert issubclass(v,ABCAnalysis)
        assert k==v.factory_keyword

class fakeNLStudy(ABCAnalysis):
    factory_keyword='fakeNLStudy'
    def __init__(self, *args):
        pass
    def initialise(self, *args, **kwargs):
        pass
    def Solve(self, *args, **kwargs):
        pass
    def makeStep(self, *args, **kwargs):
        pass

NonLinearStudy_kind[fakeNLStudy.factory_keyword] = fakeNLStudy

def test_generateNonLinearAnalysis():
    fake_input = dict()
    fake_system = dict() #shall be a ABCSystem
    fakeNLStudy_instance = generateNonLinearAnalysis(
        fakeNLStudy.factory_keyword,
        fake_input,
        fake_system
        )
    assert isinstance(fakeNLStudy_instance,fakeNLStudy)