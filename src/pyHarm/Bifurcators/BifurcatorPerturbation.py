import numpy as np
from scipy import linalg

from pyHarm import DynamicOperator
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator, BifurcationType
from pyHarm.NonLinearSolver.SolverNewtonRaphson import SolverNewtonRaphson
from pyHarm.Solver import FirstSolution, SystemSolution
from pyHarm.Systems.System import System


class BifurcatorPerturbation(ABCBifurcator):
    """Bifurcation handling through small perturbations."""

    factory_keyword: str = "perturbation"
    """str: Concrete class name used by the factory to instantiate it."""

    default_options = {
        "verbose": True,
        "interactive": False,
        "vizu": None,
        "blind_spot": 40,
        "pert_steps": 5,
        "pert": 1e-5,
    }
    """dict:
      Set of default parameters for this concrete Bifurcator if not given in the input argument.

    It contains a 'blind_spot' keyword that sets a blind angle to avoid the predictor to search in
    the inwards direction.
    It contains a 'pert_step' keyword that
    """

    def __post_init__(self):
        self.opts = getCustomOptionDictionary(self.opts, self.default_options)

    def detect(self, sol: SystemSolution):
        """Only raise flag_bifurcation for branching, no need to invert sign_ds"""

        J = sol.get_jacobian("full")
        sol.det_Jf = linalg.det(J)
        sol.det_Jx = linalg.det(J[:-1, :-1])

        if isinstance(sol, FirstSolution):
            return

        prev_sol = sol.precedent_solution

        detected = False
        if np.sign(sol.det_Jx) != np.sign(prev_sol.det_Jx):
            if np.sign(sol.det_Jf) == np.sign(prev_sol.det_Jf) or np.sign(sol.det_Jf) != np.sign(
                sol.det_Jx
            ):
                sol.flag_bifurcation = True
                sol.bifurcation_type = BifurcationType.FOLD
            else:
                sol.flag_bifurcation = True
                sol.bifurcation_type = BifurcationType.BRANCHING
                detected = True
        elif np.dot(prev_sol.dir, sol.dir) < -np.cos(np.deg2rad(self.opts["blind_spot"] / 2)):
            sol.flag_bifurcation = True
            sol.bifurcation_type = BifurcationType.UNKNOWN
            detected = True

        if self.opts["verbose"] and sol.flag_bifurcation:
            print(f"Warning: a {sol.bifurcation_type.value} was detected")

        return detected

    def localize(self, sol_list: list[SystemSolution], solver: SolverNewtonRaphson, system: System):
        """TBD"""
        if self.opts["interactive"]:
            if self.opts["vizu"] is not None:
                self.branching_view(sol_list, system)
            while True:
                branch_choice = input("Follow [m]ain branch or explore [a]lternative branch ? ")
                if branch_choice == "m":
                    sol_list[-1].sign_ds *= -1
                    return
                elif branch_choice == "a":
                    self.opts["pert"] = float(input("Input perturbation (float): "))
                    break
                print("Wrong options, please choose [m] or [a]")

        # Set Newton solver so that it now solves the perturbed equations.
        solver.solver_options["pert"] = self.opts["pert"]

        # Set tracking
        self.step_counter = self.opts["pert_steps"]
        self.in_operation = True

    def track(self, solver: SolverNewtonRaphson):
        """Here, the track method basically acts as a counter."""
        if self.step_counter == 0:
            # Reset Newton solver so that it solves the unperturbed equations.
            solver.solver_options["pert"] = 0.0
            self.in_operation = False
        self.step_counter -= 1

    def branching_view(self, sol_list: list[SystemSolution], system: System):
        """Vizualize last computed solutions before a bifurcation detection."""
        import matplotlib.pyplot as plt

        sol_accepted = [sol for sol in sol_list if sol.flag_accepted]

        # Take a few of the last computed solutions
        n_last_sols_desired = 5
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

        ax.set_xlabel("Frequency (rad/s)")
        ax.set_ylabel("Amplitude (m)")
        ax.legend()

        fig.show()
