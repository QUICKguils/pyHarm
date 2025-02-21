import numpy as np
import pytest
from pytest import MonkeyPatch

from pyHarm.Substructures.ABCSubstructure import ABCSubstructure
from pyHarm.Substructures.SubDataReader.ABCReader import ABCReader

## test ABCSubstructure


@pytest.fixture(autouse=True)
def mock_abcsubstructure(monkeypatch: MonkeyPatch) -> ABCSubstructure:
    monkeypatch.setattr(ABCSubstructure, "__abstractmethods__", set())
    monkeypatch.setattr(ABCSubstructure, "_add_connectors", lambda self, _: dict())
    monkeypatch.setattr(ABCSubstructure, "_add_kinematics", lambda self, _: dict())
    monkeypatch.setattr(
        ABCSubstructure, "factory_keyword", lambda self: "monkey_abcsubstructure"
    )


@pytest.mark.all
@pytest.mark.unit
def test_ABCSubstructure_01(mock_abcsubstructure) -> None:
    data = {"nnodes": 1, "ndofs": 1, "nmodes": 1, "matching": [0]}
    nh = 3
    name = "testsub"
    S = ABCSubstructure(nh, name, data)
    assert S.__repr__() == "testsub[ABCSubstructure]"
    assert S.nnodes == data["nnodes"]
    assert S.ndofs == data["ndofs"]
    assert S.nmodes == data["nmodes"]
    assert S.total_dofs == data["nnodes"] * data["ndofs"] + data["nmodes"]
    assert S.dofs_matching == data["matching"]
    assert S.nh == nh
    assert isinstance(S.substructure_reader, ABCReader)


from pyHarm.Substructures.FactorySubstructure import (
    SubstructureDico,
    generate_substructure,
)


@pytest.mark.all
@pytest.mark.unit
def test_FactorySubstruture() -> None:
    data = {
        "nnodes": 3,
        "ndofs": 2,
        "nmodes": 2,
        "matching": [3, 2],
        "matrix": {"M": np.array([[1]])},
    }
    nh = 3
    name = "testsub"

    for k, v in SubstructureDico.items():
        data["type"] = k
        S = generate_substructure(nh, name, data)
        assert isinstance(S, v)
        assert isinstance(S, ABCSubstructure)


from pyHarm.Substructures.OnlyDofs import OnlyDofs


@pytest.mark.all
@pytest.mark.unit
def test_OnlyDofs() -> None:
    data = {
        "type": "onlydofs",
        "nnodes": 3,
        "ndofs": 2,
        "nmodes": 2,
        "matching": [3, 2],
        "matrix": {"M": np.array([[1]])},
    }
    nh = 3
    name = "testsub"
    S = OnlyDofs(nh, name, data)
    assert S.connectors == dict()
    assert S.kinematics == dict()


from pyHarm.Substructures.Substructure import Substructure


@pytest.mark.all
@pytest.mark.unit
def test_Substructure() -> None:
    data = {
        "type": "substructure",
        "nnodes": 3,
        "ndofs": 2,
        "nmodes": 2,
        "matching": [3, 2],
        "matrix": {"M": np.array([[1]])},
    }
    nh = 3
    name = "testsub"
    S = Substructure(nh, name, data)
    assert S.connectors == {
        "testsub": {
            "type": "substructure",
            "connect": name,
            "matrix": {"M": np.array([[1]])},
        }
    }
    assert S.kinematics == dict()
