import numpy as np
import pytest
from pandas import DataFrame

from pyHarm.BaseUtilFuncs import (
    Dico_ABCClass_factory_keyword,
    getCustomOptionDictionary,
    getIndexfromExpldofs,
    pyHarm_plugin,
)


@pytest.mark.all
@pytest.mark.unit
def test_getCustomOptionDictionary() -> None:
    default = {"a": 1, "b": 2, "c": 3}
    custom = {"a": -1, "d": 4}
    expected_output = {"a": -1, "b": 2, "c": 3, "d": 4}
    result = getCustomOptionDictionary(custom_options=custom, default_options=default)
    for k, v in expected_output.items():
        assert result[k] == v


@pytest.mark.all
@pytest.mark.unit
def test_getIndexfromExpldofs() -> None:
    df = DataFrame(
        {"sub": ["a", "a", "b", "b"], "node_num": [0, 0, 0, 1], "dof_num": [0, 1, 0, 0]}
    )
    c_a_0_1 = ("a", 0, [1])
    c_b_0_1 = ("b", 0, [1])
    c_b_0_b = ("b", 0, [0])
    a_0_1 = np.array(
        [
            1,
        ]
    )
    b_0_1 = np.array([])
    b_0_0 = np.array(
        [
            2,
        ]
    )
    res_a_0_1 = getIndexfromExpldofs(df, [c_a_0_1])
    res_b_0_1 = getIndexfromExpldofs(df, [c_b_0_1])
    res_b_0_0 = getIndexfromExpldofs(df, [c_b_0_b])
    assert np.allclose(res_a_0_1, a_0_1)
    assert np.allclose(res_b_0_1, b_0_1)
    assert np.allclose(res_b_0_0, b_0_0)


class FakeClass:
    factory_keyword = "fake"

    def __init__(self):
        pass


class FakeDaughterClass(FakeClass):
    factory_keyword = "fakedaughter"

    def __init__(self):
        pass


FakeClassDico = dict()


@pytest.mark.all
@pytest.mark.unit
def test_pyHarm_plugin() -> None:
    Dico_ABCClass_factory_keyword[FakeClass] = FakeClassDico
    pyHarm_plugin(FakeDaughterClass)
    assert FakeClassDico[FakeDaughterClass.factory_keyword] == FakeDaughterClass
