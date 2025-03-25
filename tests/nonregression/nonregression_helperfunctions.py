import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

from pyHarm.Maestro import Maestro


def check_residuals(M: Maestro, max_residual_accepted=1e-6):
    SA = [sol for sol in M.nls["FRF"].SolList if sol.flag_accepted]
    for sol in SA:
        print(np.linalg.norm(sol.R_solver))
        assert np.isclose(np.linalg.norm(sol.R_solver), 0.0, atol=max_residual_accepted).all()


def generate_arc_length(om: np.ndarray, amp: np.ndarray):
    oo = np.concatenate([np.diff(om).reshape(-1, 1), np.diff(amp).reshape(-1, 1)], axis=1)
    lam = np.concatenate([np.zeros(1), np.cumsum(np.linalg.norm(oo, axis=1))])
    lam = lam / lam[-1]
    return lam


def concat_oa(om: np.ndarray, amp: np.ndarray):
    return np.concatenate([om.reshape(-1, 1), amp.reshape(-1, 1)], axis=1)


def normalize01(q: np.ndarray):
    max_val = np.max(q)
    min_val = np.min(q)
    factor = 1 / (max_val - min_val)
    q_norm = factor * (q - min_val)
    return q_norm, factor, min_val


def normalize(q: np.ndarray, factor: float, min_val: float):
    q_norm = factor * (q - min_val)
    return q_norm


def chec_FRF_vs_ref(
    lamb: np.ndarray,
    om: np.ndarray,
    amp: np.ndarray,
    reference_path="",
    max_dist=1e-2,
    kdtree_train_interpolation=int(1e6),
):
    # read reference results
    with open(reference_path + "ref_sol.csv", "r") as f:
        ref_results = pd.read_csv(f)
    om_ref_norm, om_factor, om_min_val = normalize01(ref_results["om"])
    amp_ref_norm, amp_factor, amp_min_val = normalize01(ref_results["amp"])
    om_obs_norm = normalize(om, om_factor, om_min_val)
    amp_obs_norm = normalize(amp, amp_factor, amp_min_val)
    lamd_f_om = interp1d(ref_results["lamb"], om_ref_norm, kind="cubic")
    lamd_f_amp = interp1d(ref_results["lamb"], amp_ref_norm, kind="cubic")
    lambda_train = np.linspace(0.0, 1.0, kdtree_train_interpolation)
    om_train = lamd_f_om(lambda_train)
    amp_train = lamd_f_amp(lambda_train)
    ref_tree = KDTree(concat_oa(om_train, amp_train), leafsize=30)
    check_dist, _ = ref_tree.query(concat_oa(om_obs_norm, amp_obs_norm), k=1)
    assert np.max(check_dist) <= max_dist
