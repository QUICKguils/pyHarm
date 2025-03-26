# Copyright 2024 SAFRAN SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import numpy as np
import scipy.linalg as spl


class SystemSolution:
    """Class that represents a solution of the system to be solved.

    This class is the main object that transits in the analysis process while solving a problem.
    The object contains information about its starting point, the previous SystemSolution it is
    linked to and the actual point the solver is studying.
    Once the solver has converged, the values of the residual and the solution point are stored in
    some of the attributes.

    Args:
        xs (np.ndarray): Array representing the starting point.
        last_solution_pointer: Pointer to the last converged solution object.
        **kwargs: Additional keyword arguments.

    Attributes:
        flag_restart (bool): Raised whenever (last_solution_point != index-1).
        flag_accepted (bool): Raised by the nonlinear solver, if the solution is considered as valid.
        flag_bifurcation (bool): Raised by the predictor, if a bifurcation has been detected.
        flag_solved (bool): Raised by the `save` method, if the solution is considered as solved.
        flag_intosolver (bool): Raised when the solution went through the solver.
        flag_R (bool): Presence of a Residual result.
        flag_J (bool): Presence of a Jacobian result.
        flag_J_qr (bool): Jacobian available with qr formalism of scipy.linalg.qr = [Q, R].
        flag_J_lu (bool): Jacobian available with lu formalism of scipy.linalg.lu = [P, L, U].
        flag_J_f (bool): Jacobian available at full-size.
        index_insolve (int): Index of the solution within the solver.
        niter (int): Number of iterations preformed by the solver.
        ds (float): Sstep size for the continuation.
        sign_ds (int): Sign of the step size.
        x_start (np.ndarray): Array representing the starting point.
        x (np.ndarray): Array representing the current solution point.
        x_pred (np.ndarray): Array representing the prediction point during continuation.
        R: Residual values.
        J_f: Full-size Jacobian.
        J_lu: Jacobian with LU decomposition.
        J_qr: Jacobian with QR decomposition.
        precedent_solution: A pointer to the last converged solution.
    """

    def __init__(self, xs: np.ndarray, last_solution_pointer=None, **kwargs):
        self.flag_restart = False
        self.flag_accepted = False
        self.flag_bifurcation = False
        self.flag_solved = False
        self.flag_intosolver = False
        self.flag_R = False
        self.flag_J = False
        self.flag_J_qr = False
        self.flag_J_lu = False
        self.flag_J_f = False
        self.index_insolve = 0
        self.niter = 0
        self.ds = 0.0
        self.sign_ds = 1
        self.x_start = xs
        self.x = copy.deepcopy(xs)
        self.x_pred = np.zeros_like(xs)
        self.R = None
        self.J_f = None
        self.J_lu = None
        self.J_qr = None
        self.precedent_solution = last_solution_pointer

    def save(self, SolList: list):
        """Saves the SystemSolution object in the provided list, if it is complete.

        Args:
            SolList (list[SystemSolution]): A list to save the SystemSolution object.

        Writes:
            SolList: The SystemSolution is appended to this list.
            flag_solved: This flag is set to True if the solution is considered as solved.

        Raises:
            ValueError: If the SystemSolution is not complete.
        """
        if self.flag_R and self.flag_J and self.flag_intosolver:
            self.flag_solved = True
            SolList.append(self)
        else:
            raise ValueError("The SystemSolution is not complete and thus cannot be saved")

    def get_jacobian(self, format="full", dump=False) -> np.ndarray:
        """Returns the Jacobian in the specified format.

        Args:
            format (str): The format of the Jacobian. Options: "full", "qr", "lu" (default: "full").
            dump (bool): If True, erases the Jacobian result with the format_in (default: False).

        Returns:
            np.ndarray: The Jacobian in the requested format, or None if the Jacobian is not available.

        Raises:
            ValueError: If the format is not compatible.
        """
        format_poss = {
            "full": self.flag_J_f,
            "qr": self.flag_J_qr,
            "lu": self.flag_J_lu,
        }
        if not self.flag_J:
            return None
        elif format not in format_poss:
            raise ValueError("Format not compatible")
        elif format == "full" and self.flag_J_f:
            return self.J_f
        elif format == "qr" and self.flag_J_qr:
            return self.J_qr
        elif format == "lu" and self.flag_J_lu:
            return self.J_lu
        else:
            if format == "qr" and self.flag_J_f:
                self.convert_jacobian("full", "qr")
                return self.J_qr
            if format == "full" and self.flag_J_qr:
                self.convert_jacobian("qr", "full")
                return self.J_f

    def convert_jacobian(self, format_in, format_out, dump=False):
        """Converts the Jacobian format from format_in to format_out.

        Args:
            format_in (str): The current format of the Jacobian.
            format_out (str): The desired format of the Jacobian.
            dump (bool): If True, erases the Jacobian result with the format_in (default: False).

        Returns:
            tuple: The updated flags and Jacobian data.

        Raises:
            ValueError: If the format is not compatible.
        """
        format_poss = {
            "full": [self.flag_J_f, self.J_f],
            "qr": [self.flag_J_qr, self.J_qr],
            "lu": [self.flag_J_lu, self.J_lu],
        }

        # Converts the format_in for the Jacobian to the format out.
        # If dump=True, then erases the Jacobian result with the format_in.
        def from_qr_to_full():
            self.J_f = np.dot(self.J_qr[0], self.J_qr[1])
            self.flag_J_f = True

        def from_full_to_qr():
            self.J_qr = spl.qr(self.J_f, mode="full")
            self.flag_J_qr = True

        def from_full_to_lu():
            self.J_lu = spl.lu(self.J_f)
            self.flag_J_lu = True

        def from_lu_to_full():
            self.J_f = np.dot(self.J_lu[0], self.J_lu[1], self.J_lu[2])
            self.flag_J_f = True

        def from_qr_to_lu():
            from_qr_to_full()
            from_full_to_lu()

        def from_lu_to_qr():
            from_lu_to_full()
            from_full_to_qr()

        convert_func = {
            "full": {"qr": from_full_to_qr, "lu": from_full_to_lu},
            "qr": {"full": from_qr_to_full, "lu": from_qr_to_lu},
            "lu": {"full": from_lu_to_full, "qr": from_lu_to_qr},
        }
        convert_func[format_in][format_out]()
        if dump:
            format_poss[format_in][0] = False
            format_poss[format_in][1] = None
            if (format_in, format_out) == ("qr", "lu") or (format_in, format_out) == ("lu", "qr"):
                format_poss["full"][0] = False
                format_poss["full"][1] = None
        return format_poss[format_out]


class FirstSolution(SystemSolution):
    """
    Inherits from the SystemSolution class with one major difference:
    there is no previous SystemSolution point for this class.

    Args:
        xs (np.ndarray): An array representing the starting point.
    """

    def __init__(self, xs):
        super().__init__(xs, None)
        self.x = copy.deepcopy(self.x_start)
        self.x_pred = copy.deepcopy(self.x_start)
