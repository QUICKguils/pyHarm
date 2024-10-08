import pytest

from pyHarm.Systems.FactorySystem import System_dico, generateSystem
from pyHarm.Systems.ABCSystem import ABCSystem

NH = 1
NTI = 32
INPUT_MINIMAL = {
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

@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem_System_dico():
    for k,v in System_dico.items():
        assert issubclass(v, ABCSystem)
        assert isinstance(k,str)
        assert (v.factory_keyword == k)

@pytest.mark.unit
@pytest.mark.all
def test_ABCSystem_generateSystem():
    for k,v in System_dico.items():
        Sys = generateSystem(k, INPUT_MINIMAL)
        assert isinstance(Sys, ABCSystem)

