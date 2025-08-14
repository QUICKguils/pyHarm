import numpy as np
from scipy import linalg

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator, BifurcationType
from pyHarm.NonLinearSolver.SolverNewtonRaphson import SolverNewtonRaphson
from pyHarm.Solver import FirstSolution, SystemSolution
from pyHarm.Predictors.FactoryPredictor import generatePredictor
from pyHarm.Systems.System import System


class BifurcatorABE(ABCBifurcator):
    """Localize and track alternative solution branches via the ABE."""

    factory_keyword: str = "abe"
    """str: Concrete class name used by the factory to instantiate it."""

    default_options = {
        "verbose": True,
        "blind_spot": 40,
    }
    """dict: dictionary containing the default options for this concrete Bifurcator."""

    def __post_init__(self):
        self.opts = getCustomOptionDictionary(self.opts, self.default_options)

    def detect(self, sol: SystemSolution) -> bool:
        """yee"""

        J = sol.get_jacobian("full")
        sol.det_Jf = linalg.det(J)
        sol.det_Jx = linalg.det(J[:-1, :-1])

        if isinstance(sol, FirstSolution):
            return

        prev_sol = sol.precedent_solution

        detected = False
        if np.sign(sol.det_Jx) != np.sign(prev_sol.det_Jx):
            if np.sign(sol.det_Jf) == np.sign(prev_sol.det_Jf):
            # if np.sign(sol.det_Jf) == np.sign(prev_sol.det_Jf) or np.sign(sol.det_Jf) != np.sign(sol.det_Jx):
                sol.bifurcation_type = BifurcationType.FOLD
            else:
                sol.bifurcation_type = BifurcationType.BRANCHING
                detected = True
        elif np.dot(prev_sol.dir, sol.dir) < -np.cos(np.deg2rad(self.opts["blind_spot"] / 2)):
            sol.bifurcation_type = BifurcationType.UNKNOWN
            detected = True

            if self.opts["verbose"]:
                print("A potential higher order bifurcation is detected")

        return detected

    def localize(self, sol: SystemSolution, solver: SolverNewtonRaphson, update_reductor, system: System):
        """
        This basically consist of a mini local analysis, except that
        the intermediate secant solution are not stored in the SolList.

        Algorithms based on Allgower, p.87 and p.93.
        """

        # Make sure to have the right predictor with its default options,
        # independently of the one choosed for the main analysis.
        predictor = generatePredictor("tangent")

        # Assume that a bifurcation has been detected between
        # last_sol and sol.
        prev_sol = sol.precedent_solution

        ds_tol = 1e-5
        max_iter = 20
        ds = sol.ds
        sign_ds = sol.sign_ds

        for n_iter in range(1, max_iter + 1):

            # Evaluate the step size that approaches det(Jf) = 0
            phi = np.sign(sol.det_Jf) != np.sign(prev_sol.det_Jf)
            if phi:
                ds = sol.det_Jf / (sol.det_Jf - prev_sol.det_Jf) * ds
                sign_ds *= -1
            else:
                ds /= 2

            # Check if the step length tolerance is satisfied.
            if ds <= ds_tol:
                break

            # Prediction step
            predictor.compute_dir(sol)  # Get the tangent: sol.dir
            predictor.predict(sol)  # Get the prediction: sol.x_pred

            # Prepare a new solution
            xpred_red, _, _ = update_reductor(sol.x_pred, sol)
            new_sol = SystemSolution(xpred_red, sol)

            # Corrector step
            solver.solve(new_sol)

            # Else, update the solutions for next secant iteration
            prev_sol = sol
            sol = new_sol

        if n_iter > max_iter:
            print("WARN: branching: unable to properly localize a bifurcation.")

        return sol

    def track(self, solver: SolverNewtonRaphson, sol: SystemSolution):
        """In this case, give the sol.dir of the alternative branch."""

        J = solver.jacobian(sol.x[:-1], sol)[:-1, :] # H' of allgower

        # Computing the null space of J = span{tau_1, tau_2}
        ns_limit = 30
        rcond = max(J) * 1E-4
        ns_count = 1
        while True:
            ns = linalg.null_space(J, rcond=rcond)
            if ns.shape[1] < 2:
                rcond *= 2
            elif ns.shape[1] > 2:
                rcond *= 2
            else:
                break
            if ns_count == ns_limit:
                print("unable to compute null space of jacobian.")
                break
            ns_count += 1
        tau_1 = ns[:, 1]
        tau_2 = ns[:, 2]

        # Computing the null space of J.T = span{e}
        rcond = max(J) * 1E-4
        ns_count = 1
        while True:
            e = linalg.null_space(J.T, rcond=rcond)
            if ns.shape[1] < 1:
                rcond *= 2
            elif ns.shape[1] > 1:
                rcond *= 2
            else:
                break
            if ns_count == ns_limit:
                print("unable to compute null space of transpose jacobian.")
                break
            ns_count += 1

        def g(xi_1, xi_2):
            return np.dot(e.T, solver.residual(sol.x[:-1] + xi_1*tau_1 + xi_2*tau_2, sol)[:-1])


        # Approximation of the second derivatives of Jx via central FD scheme
        # TODO: look after automatic differentiation, like in JAX

        eps = np.finfo(float).eps ** (1/3)

        dd1 = eps **(-2) * (g(eps, 0) - 2 * g(0, 0) + g(-eps, 0))
        dd2 = eps ** (-2) * (g(0, eps) - 2 * g(0, 0) + g(0, -eps))
        d1d2 = 1 / 4 * eps ** (-2) * (g(eps, eps) + g(-eps, -eps) - g(eps, -eps) - g(-eps, eps))

        def abe_approx(xi_1, xi_2):
            return dd1 * xi_1**2 + 2 * d1d2 * xi_1 * xi_2 + dd2 * xi_2**2
