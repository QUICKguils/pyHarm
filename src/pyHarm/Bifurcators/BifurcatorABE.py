from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
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
        pass

    def localize(self,  sol: SystemSolution):
        # see allgower, p.87
        pass

    def track(self, sol: SystemSolution):
        """In this case, give the sol.dir of the alternative branch."""

        J = sol.get_jacobian("full")

        # Approximation of the second derivatives of Jx via central FD scheme
        # TODO: look after automatic differentiation, like in JAX

        eps = 1.0e-7  # should be approx eps_machine**(1/3)

        dd1 = eps - 2*(J(eps, 0) - 2*J(0, 0) + J(-eps, 0))
        dd2 = eps**(-2) * (J(0, eps) - 2*J(0, 0) + J(0, -eps))
        d1d2 = 1/4 * eps**(-2) * (J(eps, eps) + J(-eps, -eps) - J(eps, -eps) - J(-eps, eps))

        def abe_approx(xi_1, xi_2):
            return dd1 * xi_1**2 + 2*d1d2 * xi_1 * xi_2 + dd2 * xi_2 **2
