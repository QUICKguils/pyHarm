import numpy as np
import pytest

from pyHarm.DynamicOperator import compute_DFT, nabla


@pytest.mark.all
@pytest.mark.unit
def test_nabla() -> None:
    from scipy.linalg import block_diag

    nab2 = nabla(2)
    nab2_corr = block_diag(
        np.array([[0]]),
        np.array([[0, 1.0], [-1.0, 0]]),
        2 * np.array([[0, 1.0], [-1.0, 0]]),
    )
    assert np.isclose(nab2, nab2_corr).all()


@pytest.mark.all
@pytest.mark.unit
def test_compute_DFT() -> None:
    NH = 1
    NTI = 128
    t = np.linspace(0, 2 * np.pi, NTI + 1)[:-1]
    zerot = np.cos(0 * t)
    cost = np.cos(t)
    sint = np.sin(t)
    D = compute_DFT(NTI, NH)
    zeroh = zerot @ D["tf"]
    cosh = cost @ D["tf"]
    sinh = sint @ D["tf"]
    zerot_rebuilt = zeroh @ D["ft"]
    cost_rebuilt = cosh @ D["ft"]
    sint_rebuilt = sinh @ D["ft"]
    assert np.isclose(zeroh, np.array([1, 0, 0])).all()
    assert np.isclose(cosh, np.array([0, 1, 0])).all()
    assert np.isclose(sinh, np.array([0, 0, 1])).all()
    assert np.isclose(zerot_rebuilt, zerot).all()
    assert np.isclose(cost_rebuilt, cost).all()
    assert np.isclose(sint_rebuilt, sint).all()
