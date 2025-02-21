import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch

from pyHarm.CoordinateSystem import GlobalCoordinateSystem
from pyHarm.Elements.NodeToNodeElements.NodeToNodeElement import NodeToNodeElement

NH = 1
NTI = 32
CS = GlobalCoordinateSystem(2)
NAME_E = "testE"
DATA_E = {
    "type": "ntnE",
    "connect": {"sub": [0], "INTERNAL": [1]},
    "dirs": [1],
}


EXPL_DOFS = pd.DataFrame(
    {
        "sub": ["sub"] * 4,
        "node_num": [0] * 2 + [1] * 2,
        "dof_num": [0, 1] * 2,
    }
)


@pytest.fixture
def mock_ntnE(monkeypatch: MonkeyPatch) -> NodeToNodeElement:
    monkeypatch.setattr(NodeToNodeElement, "__abstractmethods__", set())
    monkeypatch.setattr(
        NodeToNodeElement, "evalResidual", lambda self, x, om: np.zeros(x.size)
    )
    monkeypatch.setattr(
        NodeToNodeElement,
        "evalJacobian",
        lambda self, x, om: (np.zeros((x.size, x.size)), np.zeros(x.size)),
    )
    monkeypatch.setattr(NodeToNodeElement, "adim", lambda self, lc, wc: None)
    monkeypatch.setattr(NodeToNodeElement, "factory_keyword", "ntnE")
    return NodeToNodeElement(NH, NTI, NAME_E, DATA_E, CS)


@pytest.mark.all
@pytest.mark.unit
def test_NodeToNodeElement__init_data__(mock_ntnE) -> None:
    assert mock_ntnE.name == NAME_E
    assert mock_ntnE.CS == CS
    assert mock_ntnE.subs == ["sub"] * 2
    assert mock_ntnE.nbSub == 2
    assert mock_ntnE.nodes == [[0], [1]]
    assert mock_ntnE.data == DATA_E


@pytest.mark.all
@pytest.mark.unit
def test_NodeToNodeElement_generateIndices(mock_ntnE) -> None:
    mock_ntnE.generateIndices(EXPL_DOFS)
    assert np.allclose(mock_ntnE.indices, np.array([0, 1, 2, 3]))


@pytest.mark.all
@pytest.mark.unit
def test_NodeToNodeElement_evalJaco_DF(mock_ntnE) -> None:
    mock_ntnE.generateIndices(EXPL_DOFS)
    drdx, drdom = mock_ntnE._evalJaco_DF(np.array([0] * 4), 0.0, 1e-5)
    assert np.allclose(drdx, np.zeros((4, 4)))
    assert np.allclose(
        drdom,
        np.zeros(
            4,
        ),
    )
