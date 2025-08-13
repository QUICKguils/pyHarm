import numpy as np

from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
from pyHarm.NonLinearSolver.SolverNewtonRaphson import SolverNewtonRaphson
from pyHarm.Solver import SystemSolution


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
        """True if a bifurcation has been detected between two solution points.

        Folds are flagged as bif, but not detected by this method.
        """
        pass

    def localize(self, sol: SystemSolution, solver: SolverNewtonRaphson):
        """This basically operated a bisection to find the bifurcation point.

        Bisection is performed between the two solution point where the
        bifurcation has been detected.
        """

        # see allgower, p.87
        prev_sol = sol.precedent_solution

        bsol = sol
        atol = 1e-4  # Tol on the determinant of the Jacobian
        converged = False
        max_iter = 8
        bounds = (0.0, 1.0)

        def affine_combination(alpha):
            return alpha * sol.x + (1 - alpha) * prev_sol.x

        for n_iter in range(1, max_iter + 1):
            alpha = (bounds[0] + bounds[-1]) / 2
            # bisection gives the prediction of the next solution
            x_pred = affine_combination(alpha)

            # Create the new predicted solution
            xpred_red, _, output_expl_dofs = self._update_reductor(x_pred, bsol)
            bsol = SystemSolution(xpred_red, bsol)

            residual = solver.solve(bsol)

            # Check tolerances
            atol_respected = atol > abs(residual)
            if atol_respected:
                break

            # Update bounds
            if residual > 0:
                bounds = (bounds[0], alpha)
            else:
                bounds = (alpha, bounds[-1])

            # Assess the convergence
            if atol_respected:
                break




    def track(self, sol: SystemSolution):
        """In this case, give the sol.dir of the alternative branch."""

        J_full = sol.get_jacobian("full")
        J = J_full[:-1, :]  # H' of allgower
        J_a = np.vstack((J, sol.dir))  # Augmented Jacobian

        # Approximation of the second derivatives of Jx via central FD scheme
        # TODO: look after automatic differentiation, like in JAX

        eps = 1.0e-7  # should be approx eps_machine**(1/3)

        dd1 = eps - 2 * (J(eps, 0) - 2 * J(0, 0) + J(-eps, 0))
        dd2 = eps ** (-2) * (J(0, eps) - 2 * J(0, 0) + J(0, -eps))
        d1d2 = 1 / 4 * eps ** (-2) * (J(eps, eps) + J(-eps, -eps) - J(eps, -eps) - J(-eps, eps))

        def abe_approx(xi_1, xi_2):
            return dd1 * xi_1**2 + 2 * d1d2 * xi_1 * xi_2 + dd2 * xi_2**2
