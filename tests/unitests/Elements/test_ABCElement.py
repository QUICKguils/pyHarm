import pandas as pd
from pyHarm.Elements.ABCElement import ABCElement
from pyHarm.CoordinateSystem import GlobalCoordinateSystem
import numpy as np
import pytest
from pytest import MonkeyPatch

NH = 1
NTI = 32
NAME = 'E'
DATA = dict()
CS = GlobalCoordinateSystem(1)

@pytest.fixture
def mock_E(monkeypatch:MonkeyPatch) -> ABCElement :
    monkeypatch.setattr(ABCElement, '__abstractmethods__', set())
    monkeypatch.setattr(ABCElement, 'factory_keyword', 'mockE')
    monkeypatch.setattr(ABCElement, 'adim', lambda self,lc,wc:None)
    monkeypatch.setattr(ABCElement, 'evalResidual', lambda self,x,om:None)
    monkeypatch.setattr(ABCElement, 'evalJacobian', lambda self,x,om:(None,None))
    monkeypatch.setattr(ABCElement, 'generateIndices', lambda self,expl_dofs:None)
    monkeypatch.setattr(ABCElement, '__str__', lambda self:None)
    monkeypatch.setattr(ABCElement, '__init_data__', lambda self, name, data, CS:None)
    return ABCElement(NH,NTI,NAME,DATA,CS)

@pytest.mark.all
@pytest.mark.unit
def test_ABCElement(mock_E:ABCElement):
    assert isinstance(mock_E, ABCElement)
    mock_E.name = NAME # shows a problem in the creation of ABCElements, name shall be written right away
    assert mock_E.__repr__() == f"{NAME}[ABCElement]"