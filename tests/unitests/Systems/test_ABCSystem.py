import pandas as pd
from pyHarm.Systems.ABCSystem import ABCSystem
import numpy as np
import pytest
from pytest import MonkeyPatch
import copy
NH = 1
NTI = 32
INPUT = {
    'system':{'nh':NH, 'nti':NTI},
    'substructures':{
        'sub':{
            'type':'onlydofs',
            'ndofs':1,
            'nnodes':2,
        }
    },
    'connectors':dict(),
    'kinematics':dict()
}

INPUT_WITH_ELEMENTS = copy.deepcopy(INPUT)
INPUT_WITH_ELEMENTS['connectors']['E0']  =  {'type':'LinearSpring', 'connect':{'sub':[0]}, 'dirs':[0], 'k':1.0}
INPUT_WITH_ELEMENTS['connectors']['E1']  =  {'type':'LinearSpring', 'connect':{'sub':[0], 'INTERNAL':[1]}, 'dirs':[0], 'k':1.0}
INPUT_WITH_ELEMENTS['kinematics']['K0']  =  {'type':'GOdisplacement', 'connect':{'sub':[1]}, 'dirs':[0], 'amp':1.0, 'ho':1, 'dto':0}


@pytest.fixture
def mock_S_noinit(monkeypatch:MonkeyPatch) -> ABCSystem :
    monkeypatch.setattr(ABCSystem, '__abstractmethods__', set())
    monkeypatch.setattr(ABCSystem, 'factory_keyword', 'mockE')
    monkeypatch.setattr(ABCSystem, 'Residual', lambda self:None)
    monkeypatch.setattr(ABCSystem, 'Jacobian', lambda self:None)
    monkeypatch.setattr(ABCSystem, '__init__', lambda self,_:None)
    return ABCSystem(INPUT)

@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__maincarateristics(mock_S_noinit:ABCSystem):
    mock_S_noinit._maincarateristics(INPUT)
    assert mock_S_noinit.nh == NH
    assert mock_S_noinit.nti == NTI
    assert mock_S_noinit.adim == False
    assert mock_S_noinit.lc == 1.0
    assert mock_S_noinit.wc == 1.0
    input_with_adim = copy.deepcopy(INPUT)
    input_with_adim['system']['adim']  =  {'status':True, 'lc':2.0, 'wc':1/2.}
    mock_S_noinit._maincarateristics(input_with_adim)
    assert mock_S_noinit.adim == True
    assert mock_S_noinit.lc == 2.0
    assert mock_S_noinit.wc == 1/2.


@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__buildElements(mock_S_noinit:ABCSystem):
    mock_S_noinit._maincarateristics(INPUT)
    mock_S_noinit._buildElements(INPUT)
    assert mock_S_noinit.LC == list()
    assert mock_S_noinit.LE == list()
    assert len(mock_S_noinit.expl_dofs) == 6
    input_with_linearspring = copy.deepcopy(INPUT)
    input_with_linearspring['connectors']['E0']  =  {'type':'LinearSpring', 'connect':{'sub':[0]}, 'dirs':[0], 'k':1.0}
    input_with_linearspring['connectors']['E1']  =  {'type':'LinearSpring', 'connect':{'sub':[0], 'INTERNAL':[1]}, 'dirs':[0], 'k':1.0}
    input_with_linearspring['kinematics']['K0']  =  {'type':'GOdisplacement', 'connect':{'sub':[1]}, 'dirs':[0], 'amp':1.0, 'ho':1, 'dto':0}
    mock_S_noinit._buildElements(input_with_linearspring)
    assert len(mock_S_noinit.LE) == 2
    assert len(mock_S_noinit.LC) == 1
    from pyHarm.Elements.NodeToNodeElements.LinearSpring import LinearSpring
    from pyHarm.KinematicConditions.GODisplacement import GODisplacement
    assert isinstance(mock_S_noinit.LE[0], LinearSpring)
    assert isinstance(mock_S_noinit.LC[0], GODisplacement)
    
@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__complete_expl_dofs(mock_S_noinit:ABCSystem):
    mock_S_noinit._maincarateristics(INPUT_WITH_ELEMENTS)
    mock_S_noinit._buildElements(INPUT_WITH_ELEMENTS)
    mock_S_noinit.ndofs = len(mock_S_noinit.expl_dofs)
    mock_S_noinit.LE_extforcing = [e for e in mock_S_noinit.LE if ((not e.flag_nonlinear) and (e.flag_extforcing))]
    mock_S_noinit.LE_linear = [e for e in mock_S_noinit.LE if ((not e.flag_nonlinear) and (not e.flag_extforcing))]
    mock_S_noinit.LE_nonlinear_dlft = [e for e in mock_S_noinit.LE if ((e.flag_nonlinear) and (e.flag_DLFT))]
    mock_S_noinit.LE_nonlinear_nodlft = [e for e in mock_S_noinit.LE if ((e.flag_nonlinear) and not (e.flag_DLFT))]
    mock_S_noinit._complete_expl_dofs()
    kc_dofs = np.array(mock_S_noinit.expl_dofs[mock_S_noinit.expl_dofs['KC']==1].index)
    print(kc_dofs)
    assert (kc_dofs == np.array([1,3,5])).all()


@pytest.fixture
def mock_S(monkeypatch:MonkeyPatch) -> ABCSystem :
    monkeypatch.setattr(ABCSystem, '__abstractmethods__', set())
    monkeypatch.setattr(ABCSystem, 'factory_keyword', 'mockE')
    monkeypatch.setattr(ABCSystem, 'Residual', lambda self:None)
    monkeypatch.setattr(ABCSystem, 'Jacobian', lambda self:None)
    return ABCSystem(INPUT_WITH_ELEMENTS)

@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__get_expl_dofs_into_solver(mock_S:ABCSystem):
    df_noKC = mock_S._get_expl_dofs_into_solver()
    assert len(df_noKC) == 3
    assert (df_noKC == mock_S.expl_dofs[mock_S.expl_dofs['KC']!=1].reset_index(drop=True)).all().all()


@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__expand_q(mock_S:ABCSystem):
    x = mock_S._expand_q(np.array([1., 2., 3.,100.]))
    assert (x[np.array([0,2,4])] == np.array([1., 2., 3.])).all()
    assert (x[np.array([1,3,5])] == np.array([0., 0., 0.])).all()

    
@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__complete_x(mock_S:ABCSystem):
    x = mock_S._expand_q(np.array([1., 2., 3.,100.]))
    x += mock_S._complete_x(mock_S.LC, x)
    assert (x[np.array([0,2,4])] == np.array([1., 2., 3.])).all()
    assert (x[np.array([1,3,5])] == np.array([0., 1., 0.])).all()


@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem_get_full_disp(mock_S:ABCSystem):
    x = mock_S.get_full_disp(np.array([1., 2., 3.,100.]))
    assert (x[np.array([0,2,4])] == np.array([1., 2., 3.])).all()
    assert (x[np.array([1,3,5])] == np.array([0., 1., 0.])).all()

    
@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__complete_R(mock_S:ABCSystem):
    x = mock_S.get_full_disp(np.array([1., 2., 3.,100.]))
    R = np.zeros(x.shape[0]-1)
    Radd = mock_S._complete_R(mock_S.LC, R, x)
    assert Radd.shape == R.shape

@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__residual(mock_S:ABCSystem):
    R = mock_S._residual(mock_S.LE, np.zeros(7,))
    assert R.shape == (6,)

@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__jacobian(mock_S:ABCSystem):
    Jx,Jom = mock_S._jacobian(mock_S.LE, np.zeros(7,))
    assert Jx.shape == (6,6)
    assert Jom.shape == (6,1)

@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__get_assembled_stiffness_matrix(mock_S:ABCSystem):
    K = mock_S._get_assembled_stiffness_matrix(np.zeros(7,))
    assert K.shape == (6,6)
    assert (K == np.kron(np.eye(3), np.array([[2.,-1.],[-1.,1.]]))).all()
    

INPUT_WITH_MASS = copy.deepcopy(INPUT_WITH_ELEMENTS)
INPUT_WITH_MASS['connectors']['mass'] = {'type':'GOElement', 'connect':{'sub':[0]}, 'dirs':[0], 'dto':2, 'xo':1, 'k':1.}
@pytest.fixture
def mock_S_withmass(monkeypatch:MonkeyPatch) -> ABCSystem :
    monkeypatch.setattr(ABCSystem, '__abstractmethods__', set())
    monkeypatch.setattr(ABCSystem, 'factory_keyword', 'mockE')
    monkeypatch.setattr(ABCSystem, 'Residual', lambda self:None)
    monkeypatch.setattr(ABCSystem, 'Jacobian', lambda self:None)
    return ABCSystem(INPUT_WITH_MASS)

@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem__get_assembled_mass_matrix(mock_S_withmass:ABCSystem):
    M = mock_S_withmass._get_assembled_mass_matrix(np.zeros(7,))
    assert M.shape == (6,6)
    verif = np.kron(np.eye(3), np.array([[-1.,0.],[0.,0.]]))
    verif[0] = 0.
    assert (M == verif).all()

