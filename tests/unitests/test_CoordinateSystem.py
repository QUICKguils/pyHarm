from pyHarm.CoordinateSystem import CoordinateSystem, GlobalCoordinateSystem, generateCoordinateSystem
import numpy as np
import pytest

@pytest.mark.all
@pytest.mark.unit
def test_generateCoordinateSystem() -> None :
    dirs = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    C = generateCoordinateSystem(dirs)
    assert isinstance(C,CoordinateSystem)

@pytest.mark.all
@pytest.mark.unit
def test_CoordinateSystem() -> None : 
    dirs_notortho = [[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]]
    with pytest.raises(ValueError):
        C = CoordinateSystem(dirs_notortho)
    dirs = [[1.,0.,0.],[0.,0.5,-0.5],[0.,0.5,0.5]]
    C = CoordinateSystem(dirs)
    assert C.n_dirs == 3
    assert C.n_component == 3
        


@pytest.mark.all
@pytest.mark.unit
def test_GlobalCoordinateSystem() -> None : 
    ndirs = 3
    C = GlobalCoordinateSystem(ndirs)
    assert C.n_dirs == ndirs
    assert C.n_component == ndirs
    assert np.allclose(C.dirs,np.eye(ndirs))