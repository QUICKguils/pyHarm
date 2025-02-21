## Test ABCReader
import numpy as np
import pytest

from pyHarm.Substructures.SubDataReader.ABCReader import ABCReader
from pyHarm.Substructures.SubDataReader.FactoryReader import (
    SubstructureReaderDictionary,
    generate_subreader,
)
from pyHarm.Substructures.SubDataReader.GenericReader import GenericReader


@pytest.mark.all
@pytest.mark.unit
def test_ABCReader() -> None:
    class fakeReader(ABCReader):
        factory_keyword = "fakereader"

        def data_complete(self, data) -> dict:
            return data

    R = fakeReader()
    assert R.factory_keyword == "fakereader"
    fake_data = {"fake": 1}
    fake_data_complete = R.data_complete(fake_data)
    for k in fake_data_complete.keys():
        assert fake_data_complete[k] == fake_data[k]


@pytest.mark.all
@pytest.mark.unit
def test_FactoryReader() -> None:
    class fakeReader(ABCReader):
        factory_keyword = "fakereader"

        def data_complete(self, data) -> dict:
            return data

    data = {"reader": "fakereader"}
    with pytest.raises(ValueError):
        R = generate_subreader(data)

    SubstructureReaderDictionary[fakeReader.factory_keyword] = fakeReader
    R = generate_subreader(data)
    assert isinstance(R, fakeReader)


@pytest.mark.all
@pytest.mark.unit
def test_GenericReader() -> None:
    data = {"reader": "generic"}
    R = generate_subreader(data)
    with pytest.raises(ValueError):
        _ = R.data_complete(data)
    data["nnodes"] = 2
    data["ndofs"] = 2
    R = generate_subreader(data)
    data = R.data_complete(data)
    assert isinstance(R, GenericReader)
    assert data["nmodes"] == 0
    data.pop("nnodes")
    data["matrix"] = {"testMat": np.eye(1)}
    R = generate_subreader(data)
    with pytest.raises(ValueError):
        _ = R.data_complete(data)
    data["matrix"] = {"testMat": np.eye(2)}
    R = generate_subreader(data)
    data = R.data_complete(data)
    assert data["nnodes"] == 1
