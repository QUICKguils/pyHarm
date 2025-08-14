import numpy as np
from scipy import linalg

from pyHarm import DynamicOperator
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator, BifurcationType
from pyHarm.NonLinearSolver.SolverNewtonRaphson import SolverNewtonRaphson
from pyHarm.Predictors.FactoryPredictor import generatePredictor
from pyHarm.Solver import FirstSolution, SystemSolution, get_last_solution
from pyHarm.Systems.System import System


class BifurcatorABE(ABCBifurcator):
    """Localize and track alternative solution branches via the ABE."""

    factory_keyword: str = "abe"
    """str: Concrete class name used by the factory to instantiate it."""

    default_options = {
        "verbose": True,
        "interactive": False,
        "vizu": None,
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

    def localize(
        self,
        sol_list: list[SystemSolution],
        solver: SolverNewtonRaphson,
        update_reductor,
        system: System,
    ):
        """
        This basically consist of a mini local analysis, except that
        the intermediate secant solution are not stored in the SolList.

        Returns a solution point closed to a detected bifurcation,
        up to a certain tolerance.

        Algorithms based on Allgower, p.87 and p.93.
        """

        # Make sure to have the right predictor with its default options,
        # independently of the one choosed for the main analysis.
        predictor = generatePredictor("tangent", dict())

        # Assume that a bifurcation has been detected between
        # last_sol and sol.
        sol = get_last_solution(sol_list)
        prev_sol = sol.precedent_solution

        ds_tol = 1e-5
        max_iter = 20
        ds = sol.ds
        sign_ds = sol.sign_ds

        for n_iter in range(1, max_iter + 1):
            J = sol.get_jacobian("full")
            sol.det_Jf = linalg.det(J)
            # Evaluate the step size that approaches det(Jf) = 0
            if np.sign(sol.det_Jf) != np.sign(prev_sol.det_Jf):
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
            new_sol.sign_ds = sol.sign_ds

            # Corrector step
            solver.solve(new_sol)

            # Else, update the solutions for next secant iteration
            prev_sol = sol
            sol = new_sol

        if n_iter > max_iter:
            print("WARN: branching: unable to properly localize a bifurcation.")

        self.in_operation = True

        return sol

    def track(
        self,
        sol_list: list[SystemSolution],
        solver: SolverNewtonRaphson,
        system: System,
        b_sol: SystemSolution,
    ):
        """Compute the sol.dir of the alternative branch.

        If interactive, it proposes which branch to follow on each
        detected branching bifurcation.
        If not interactive, it follows the main branch, and stores
        the computed tangents in the underlying solution: sol.branches_dir.

        Overwrite the sol.dir property with the desired tangent.
        """

        J = solver.jacobian(b_sol.x, b_sol)[:-1, :]  # H' of allgower

        ## Computing the null space of J = span{tau_1, tau_2}

        rcond = max(J.shape) * 1e-6  # following scipy docu, guesstimate
        ns_limit = 30
        ns_count = 1

        while True:
            ns = linalg.null_space(J, rcond=rcond)
            if ns.shape[1] < 2:
                rcond *= 2
            elif ns.shape[1] > 2:
                rcond /= 2
            else:
                break
            if ns_count == ns_limit:
                print("unable to compute null space of jacobian.")
                break
            ns_count += 1

        tau_1 = ns[:, 0]
        tau_2 = ns[:, 1]

        # ## Computing the null space of J.T = span{e}
        #
        # rcond = max(J) * 1E-4
        # ns_count = 1
        #
        # while True:
        #     e = linalg.null_space(J.T, rcond=rcond)
        #     if ns.shape[1] < 1:
        #         rcond *= 2
        #     elif ns.shape[1] > 1:
        #         rcond *= 2
        #     else:
        #         break
        #     if ns_count == ns_limit:
        #         print("unable to compute null space of transpose jacobian.")
        #         break
        #     ns_count += 1
        #
        # ## Approximation of the Algebraic Bifurcation Equation
        #
        # def g(xi_1, xi_2):
        #     return np.dot(e.T, solver.residual(sol.x[:-1] + xi_1*tau_1 + xi_2*tau_2, sol)[:-1])
        #
        # # Approximation of the second derivatives of Jx via central FD scheme
        # # TODO: look after automatic differentiation, like in JAX
        # eps = np.finfo(float).eps ** (1/3)
        # dd1 = eps **(-2) * (g(eps, 0) - 2 * g(0, 0) + g(-eps, 0))
        # dd2 = eps ** (-2) * (g(0, eps) - 2 * g(0, 0) + g(0, -eps))
        # d1d2 = 1 / 4 * eps ** (-2) * (g(eps, eps) + g(-eps, -eps) - g(eps, -eps) - g(-eps, eps))
        #
        # def abe_approx(xi_1, xi_2):
        #     return dd1 * xi_1**2 + 2 * d1d2 * xi_1 * xi_2 + dd2 * xi_2**2

        ## Prepare outputs

        # Classify tangents
        # Lil convention: 0 -> main; 1,2 -> alt
        sol = get_last_solution(sol_list)
        if np.abs(np.dot(sol.dir, tau_1)) > np.abs(np.dot(sol.dir, tau_2)):
            tangents = (tau_1, tau_2, -tau_2)
        else:
            tangents = (tau_2, tau_1, -tau_1)

        # Save the computed tangents for the solution
        # corresponding to the located bifurcation
        b_sol.branches_dir = tangents
        sol_list.append(b_sol)

        # If interactive, propose which branch to follow
        if self.opts["interactive"] is True:
            if self.opts["vizu"] is not None:
                self.branching_view(sol_list, system)
            while True:
                branch_choice = int(input("Which branch to follow (integer, look at the graph) ? "))
                if branch_choice in {0, 1, 2}:
                    break
                print("Pls bro use this software correcly. Enter 0, 1 or 2.")
            b_sol.dir = tangents[branch_choice]

        # Otherwise just follow the main branch
        b_sol.dir = tangents[0]

        self.in_operation = False

    def branching_view(self, sol_list: list[SystemSolution], system: System):
        """Vizualize last computed solutions before a bifurcation detection."""
        import matplotlib.pyplot as plt

        sol_accepted = [sol for sol in sol_list if sol.flag_accepted]

        # Take a few of the last computed solutions
        n_last_sols_desired = 2000
        n_last_sols = min(len(sol_accepted), n_last_sols_desired)
        last_sols = sol_accepted[-n_last_sols:]

        freqs = np.array([sol.x[-1] for sol in last_sols])
        n_sol = freqs.shape[0]

        if len(self.opts["vizu"]) == 1:
            # NOTE: local dumplicate of Maestro.getIndex, to avoid circular deps
            # TODO: find a cleaner way to do that
            sub, node, dir_num = self.opts["vizu"][0]
            expl_dofs = system.expl_dofs
            submatch = expl_dofs["sub"] == sub
            nodematch = expl_dofs["node_num"] == node
            dof_match = expl_dofs["dof_num"] == dir_num
            ix_dof = np.sort(expl_dofs[submatch * nodematch * dof_match].index)
        # TODO: implement ampl difference bw two nodes
        # elif len(self.opts["vizu"]) == 2:
        #     ix_dof1 = Maestro.getIndex(self.opts["vizu"])
        #     ix_dof2 = Maestro.getIndex(self.opts["vizu"])
        else:
            print("Bad options for `bifurcator_options[vizu]`")
            print("Unable to show the solution near the bifurcation.")

        # DFTO = M.nls["duffing_frf"].system.LE_nonlinear_nodlft[0].D
        DFTO = DynamicOperator.compute_DFT(system.nti, system.nh)
        cs_accepted = np.concatenate([sol.x.reshape(-1, 1) for sol in last_sols], axis=1)
        ampls = np.empty(n_sol)
        for ix_sol in range(n_sol):
            displ = cs_accepted[ix_dof, ix_sol] @ DFTO["ft"]
            ampls[ix_sol] = np.max(displ)

        fig, ax = plt.subplots()

        ax.plot(freqs, ampls, marker=".")
        b_sol = last_sols[-1]
        prev_sol = last_sols[-2]
        for ix_dir, dir in enumerate(b_sol.branches_dir):
            x_pred = b_sol.x + b_sol.sign_ds * prev_sol.ds * dir
            ampl_t = np.max(x_pred[:-1] @ DFTO["ft"])
            ax.plot([freqs[-1], x_pred[-1]], [ampls[-1], ampl_t])
            ax.text(x_pred[-1], ampl_t, ix_dir, ha="center", va="center")

        ax.set_xlabel("Frequency (rad/s)")
        ax.set_ylabel("Amplitude (m)")

        fig.show()
