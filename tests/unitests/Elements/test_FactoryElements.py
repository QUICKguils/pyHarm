import pytest

from pyHarm.CoordinateSystem import GlobalCoordinateSystem
from pyHarm.Elements.ABCElement import ABCElement
from pyHarm.Elements.FactoryElements import ElementDictionary, L_Elem, generateElement

NH = 1
NTI = 32
NAME = "E"
DATA = dict()
DICT_CS = {"global": GlobalCoordinateSystem(1), "local": GlobalCoordinateSystem(1)}


@pytest.mark.all
@pytest.mark.unit
def test_L_Elem() -> None:
    for e in L_Elem:
        assert issubclass(e, ABCElement)


@pytest.mark.all
@pytest.mark.unit
def test_ElementDictionary() -> None:
    for key, e in ElementDictionary.items():
        assert e.factory_keyword == key


@pytest.mark.all
@pytest.mark.unit
def test_generateElement() -> None:
    class FakeE(ABCElement):
        factory_keyword = "fakeE"

        def __init_data__(self, *args):
            pass

        def __str__(self, *args):
            pass

        def generateIndices(self, *args):
            pass

        def adim(self, *args):
            pass

        def evalResidual(self, *args):
            pass

        def evalJacobian(self, *args):
            pass

    ElementDictionary[FakeE.factory_keyword] = FakeE
    DATA["type"] = FakeE.factory_keyword
    E = generateElement(NH, NTI, NAME, DATA, DICT_CS)
    assert isinstance(E, ABCElement)
    DATA["coordinatesystem"] = "local"
    E = generateElement(NH, NTI, NAME, DATA, DICT_CS)
    assert isinstance(E, ABCElement)
    with pytest.raises(KeyError):
        DATA["coordinatesystem"] = "BS"
        E = generateElement(NH, NTI, NAME, DATA, DICT_CS)
