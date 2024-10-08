import pytest 
from pyHarm.Predictors.FactoryPredictor import Predictor_dico, generatePredictor
from pyHarm.Predictors.ABCPredictor import ABCPredictor

@pytest.mark.all
@pytest.mark.unit
def test_Predictor_dico() -> None : 
    for key,predictor in Predictor_dico.items() : 
        assert predictor.factory_keyword == key

SIGN_DS = 1

@pytest.mark.all
@pytest.mark.unit
def test_generatePredictor() -> None :
    class FakePredictor(ABCPredictor) : 
        factory_keyword = 'fakepredictor'
        def predict(self, *args) :
            pass
    Predictor_dico['fakepredictor'] = FakePredictor
    P = generatePredictor('fakepredictor', SIGN_DS, dict())
    assert isinstance(P, FakePredictor)
    assert isinstance(P, ABCPredictor)