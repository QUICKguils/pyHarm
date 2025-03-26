import numpy as np
import pytest

from pyHarm.Solver import FirstSolution, SystemSolution

ALL_FLAGS = [
    "flag_restart",
    "flag_accepted",
    "flag_bifurcation",
    "flag_solved",
    "flag_intosolver",
    "flag_R",
    "flag_J",
    "flag_J_qr",
    "flag_J_lu",
    "flag_J_f",
]
FLAGS_CHECKCOMPLETE = ["flag_R", "flag_J", "flag_intosolver"]


@pytest.mark.all
@pytest.mark.unit
def test_SystemSolution_checkcomplete() -> None:
    fake_disp_vector = np.random.uniform(-1, 1, 100)
    S = SystemSolution(fake_disp_vector)
    assert S.CheckComplete() == False
    for flag in FLAGS_CHECKCOMPLETE:
        setattr(S, flag, True)
    assert S.CheckComplete() == True


@pytest.mark.all
@pytest.mark.unit
def test_SystemSolution_SaveSolution() -> None:
    fake_disp_vector = np.random.uniform(-1, 1, 100)
    S = SystemSolution(fake_disp_vector)
    List_of_solutions = list()
    with pytest.raises(ValueError):
        assert S.save(List_of_solutions)
    assert len(List_of_solutions) == 0
    for flag in FLAGS_CHECKCOMPLETE:
        setattr(S, flag, True)
    S.save(List_of_solutions)
    assert len(List_of_solutions) == 1
    assert List_of_solutions[0] == S


@pytest.mark.all
@pytest.mark.unit
def test_SystemSolution_convertJacobian_full_to_lu() -> None:
    fake_disp_vector = np.random.uniform(-1, 1, 100)
    S = SystemSolution(fake_disp_vector)
    J_F = np.array(
        [
            [
                25.0,
                5.0,
                1.0,
            ],
            [64.0, 8.0, 1.0],
            [144.0, 12.0, 1.0],
        ]
    )
    S.J_f = J_F
    S.flag_J_f = True
    _ = S.convert_jacobian("full", "lu")
    assert S.flag_J_lu
    assert np.allclose(S.J_lu[1], np.tril(S.J_lu[1]))
    assert np.allclose(S.J_lu[2], np.triu(S.J_lu[2]))
    assert np.allclose(J_F, S.J_lu[0] @ S.J_lu[1] @ S.J_lu[2])


@pytest.mark.all
@pytest.mark.unit
def test_SystemSolution_convertJacobian_full_to_qr() -> None:
    fake_disp_vector = np.random.uniform(-1, 1, 100)
    S = SystemSolution(fake_disp_vector)
    J_F = np.array(
        [
            [
                25.0,
                5.0,
                1.0,
            ],
            [64.0, 8.0, 1.0],
            [144.0, 12.0, 1.0],
        ]
    )
    S.J_f = J_F
    S.flag_J_f = True
    _ = S.convert_jacobian("full", "qr")
    assert S.flag_J_qr
    assert np.allclose(S.J_qr[1], np.triu(S.J_qr[1]))
    assert np.allclose(S.J_qr[0].T @ S.J_qr[0], np.eye(3))
    assert np.allclose(J_F, S.J_qr[0] @ S.J_qr[1])


@pytest.mark.all
@pytest.mark.unit
def test_SystemSolution_getJacobian() -> None:
    fake_disp_vector = np.random.uniform(-1, 1, 100)
    S = SystemSolution(fake_disp_vector)
    J_F = np.array(
        [
            [
                25.0,
                5.0,
                1.0,
            ],
            [64.0, 8.0, 1.0],
            [144.0, 12.0, 1.0],
        ]
    )
    S.J_f = J_F
    S.flag_J_f = True
    assert S.get_jacobian("shallreturnnone") == None
    S.flag_J = True
    with pytest.raises(ValueError):
        _ = S.get_jacobian("shallraiseerror")
    _ = S.get_jacobian("qr")
    S.flag_J_f = False
    _ = S.get_jacobian("full")
    assert np.allclose(S.J_f, S.J_qr[0] @ S.J_qr[1])


@pytest.mark.all
@pytest.mark.unit
def test_init_FirstSolution() -> None:
    fake_disp_vector = np.random.uniform(-1, 1, 100)
    FS = FirstSolution(fake_disp_vector)
    S = SystemSolution(fake_disp_vector, FS)
    assert S.precedent_solution == FS
