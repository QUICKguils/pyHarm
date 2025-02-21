import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch

from pyHarm.Analysis.ABCAnalysis import ABCAnalysis


@pytest.fixture
def mock_A(monkeypatch: MonkeyPatch) -> ABCAnalysis:
    monkeypatch.setattr(ABCAnalysis, "__abstractmethods__", set())
    monkeypatch.setattr(ABCAnalysis, "factory_keyword", "mockA")
    monkeypatch.setattr(ABCAnalysis, "initialise", lambda: None)
    monkeypatch.setattr(ABCAnalysis, "Solve", lambda: None)
    monkeypatch.setattr(ABCAnalysis, "makeStep", lambda: None)
    return ABCAnalysis(dict(), dict(), 0)


@pytest.mark.all
@pytest.mark.unit
def test_ABCAnalysis(mock_A: ABCAnalysis):
    assert isinstance(mock_A, ABCAnalysis)
