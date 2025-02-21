import numpy as np
import pytest
from pytest import MonkeyPatch

from pyHarm.Correctors.ABCCorrector import ABCCorrector


@pytest.fixture
def mock_cor(monkeypatch: MonkeyPatch) -> ABCCorrector:
    monkeypatch.setattr(ABCCorrector, "__abstractmethods__", set())
    monkeypatch.setattr(ABCCorrector, "factory_keyword", "mock_cor")
    monkeypatch.setattr(ABCCorrector, "ClosureEquation", lambda *args: None)
    monkeypatch.setattr(ABCCorrector, "ClosureJacobian", lambda *args: None)
    return ABCCorrector()


@pytest.mark.all
@pytest.mark.unit
def test_ABCCorrector(mock_cor) -> None:
    pass
